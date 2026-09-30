"""High-level causal analysis orchestration."""

from __future__ import annotations

import logging
import shutil

import pandas as pd

from causal_pipeline.ate import (
    ATESensitivityAnalyzer,
    BaseATEEstimator,
    create_ate_estimator,
    estimator_supports_sensitivity,
)
from causal_pipeline.cate import BaseCATEEstimator, create_cate_estimator
from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import CausalDataset, DataSplitter
from causal_pipeline.diagnostics import DiagnosticsRunner
from causal_pipeline.evaluation import CATEEvaluator, ContrastEvaluation
from causal_pipeline.policy import PolicyService
from causal_pipeline.results import ResultStore
from causal_pipeline.utils import write_dataframe

logger = logging.getLogger(__name__)


class CausalPipeline:
    """End-to-end observational causal analysis pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        splitter: DataSplitter | None = None,
        diagnostics_runner: DiagnosticsRunner | None = None,
        sensitivity_analyzer: ATESensitivityAnalyzer | None = None,
        evaluator: CATEEvaluator | None = None,
        policy_service: PolicyService | None = None,
        results: ResultStore | None = None,
    ) -> None:
        self.config = config
        self.splitter = splitter or DataSplitter()
        self.diagnostics_runner = diagnostics_runner
        self.sensitivity_analyzer = sensitivity_analyzer
        self.evaluator = evaluator
        self.policy_service = policy_service
        self.results = results or ResultStore(config)

    def run(self, df_input: pd.DataFrame) -> None:
        logger.info("Starting causal pipeline run.")
        dataset = CausalDataset(data=self.config.data, df=df_input.copy())
        partitions = self.splitter.split(dataset, self.config)
        self.results.save_partitions(partitions)

        if self.config.diagnostics is not None:
            diagnostics_runner = self.diagnostics_runner or DiagnosticsRunner(
                self.config.diagnostics,
            )
            diagnostics = diagnostics_runner.run(
                partitions.train,
                results_root=str(self.results.root),
            )
            self.results.save_diagnostics(diagnostics)

        ate_models = self.fit_ate_estimators(partitions.train)
        ate_results = self.estimate_ate_models(ate_models)
        sensitivity_summaries = self.run_ate_sensitivity(ate_models, partitions.train)

        cate_models = self.fit_cate_estimators(partitions.train)
        cate_ate_results = self.estimate_from_cate_models(cate_models, partitions.validation)
        validation_predictions = self.predict_cate_models(cate_models, partitions.validation)

        evaluator = None
        if self.config.cate_evaluation is not None:
            evaluator = self.evaluator or CATEEvaluator(self.config.cate_evaluation)

        cate_evaluation: dict[str, list[ContrastEvaluation]] = {}
        if evaluator is not None and cate_models:
            validation_scores = evaluator.build_robust_scores(partitions.validation)
            cate_evaluation = evaluator.evaluate_cate_models(
                predictions=validation_predictions,
                robust_scores=validation_scores,
                results_root=self.results.root,
            )
            self.persist_cate_evaluation(cate_evaluation=cate_evaluation)

        policy_results = None
        if self.config.policy:
            if evaluator is None:
                raise ValueError(
                    "Policy methods require cate_evaluation so held-out rules can be scored."
                )
            policy_service = self.policy_service or PolicyService(self.config.policy)
            policies = policy_service.fit_policy_methods(
                cate_models=cate_models,
                validation=partitions.validation,
                predictions=validation_predictions,
            )
            test_scores = evaluator.build_robust_scores(partitions.test)
            policy_results = policy_service.evaluate_policies(
                policies=policies,
                test=partitions.test,
                test_scores=test_scores,
            )
            self.results.write_dataframe("policy/summary.csv", policy_results)

        ate_summary = self.build_ate_summary(ate_results, sensitivity_summaries)
        cate_summary = self.build_cate_summary(cate_ate_results, cate_evaluation)
        self.results.write_summaries(
            ate_summary=ate_summary,
            cate_summary=cate_summary,
            policy_summary=policy_results,
        )
        logger.info("Causal pipeline run completed.")

    def fit_ate_estimators(self, train: CausalDataset) -> dict[str, BaseATEEstimator]:
        models: dict[str, BaseATEEstimator] = {}
        if not self.config.ate_estimators:
            return models
        for spec in self.config.ate_estimators:
            logger.info("Fitting ATE estimator %s.", spec.kind.value)
            estimator = create_ate_estimator(spec=spec, data=train)
            estimator.fit(train)
            models[spec.kind.value] = estimator
        return models

    def estimate_ate_models(self, ate_models: dict[str, BaseATEEstimator]) -> dict[str, pd.DataFrame]:
        results = {}
        if not ate_models:
            return results
        for estimator_id, model in ate_models.items():
            results[estimator_id] = model.estimate()
            write_dataframe(
                self.results.ate_dir(estimator_id) / "estimate.csv",
                results[estimator_id],
            )
        summary = pd.concat(
            [
                df_ate.assign(estimator=estimator_id)
                for estimator_id, df_ate in results.items()
            ],
            ignore_index=True,
        )
        self.results.write_dataframe("ate/summary.csv", summary)
        return results

    def run_ate_sensitivity(
        self,
        ate_models: dict[str, BaseATEEstimator],
        train: CausalDataset,
    ) -> dict[str, pd.DataFrame]:
        if self.config.sensitivity is None or not ate_models:
            return {}
        analyzer = self.sensitivity_analyzer or ATESensitivityAnalyzer(
            self.config.sensitivity,
        )
        summaries = {}
        for estimator_id, model in ate_models.items():
            if not estimator_supports_sensitivity(model):
                logger.info(
                    "Skipping sensitivity for %s. DoubleML score bounds are the supported analysis.",
                    estimator_id,
                )
                self._clear_sensitivity_output(estimator_id)
                continue
            logger.info("Running sensitivity analysis for %s.", estimator_id)
            result = analyzer.analyze(
                estimator=model,
                training_data=train,
                estimator_id=estimator_id,
                results_root=str(self.results.root),
            )
            if result is None:
                self._clear_sensitivity_output(estimator_id)
                continue
            sensitivity_dir = self.results.ate_dir(estimator_id) / "sensitivity"
            write_dataframe(sensitivity_dir / "summary.csv", result.summary)
            write_dataframe(sensitivity_dir / "benchmarks.csv", result.benchmarks)
            summaries[estimator_id] = result.summary
        return summaries

    def _clear_sensitivity_output(self, estimator_id: str) -> None:
        """Drop a previous sensitivity folder so an unsupported estimator cannot keep an old bound."""
        sensitivity_dir = self.results.ate_dir(estimator_id) / "sensitivity"
        if sensitivity_dir.exists():
            shutil.rmtree(sensitivity_dir)

    def fit_cate_estimators(self, train: CausalDataset) -> dict[str, BaseCATEEstimator]:
        models: dict[str, BaseCATEEstimator] = {}
        if not self.config.cate_estimators:
            return models
        for spec in self.config.cate_estimators:
            logger.info("Fitting CATE estimator %s.", spec.kind.value)
            estimator = create_cate_estimator(spec=spec, data=train)
            estimator.fit(train)
            models[spec.kind.value] = estimator
        return models

    def estimate_from_cate_models(
        self,
        cate_models: dict[str, BaseCATEEstimator],
        validation: CausalDataset,
    ) -> dict[str, pd.DataFrame]:
        results = {}
        for estimator_id, model in cate_models.items():
            estimate = model.estimate(validation)
            results[estimator_id] = estimate
            write_dataframe(
                self.results.cate_dir(estimator_id) / "ate_estimate.csv",
                estimate,
            )
        return results

    def predict_cate_models(
        self,
        cate_models: dict[str, BaseCATEEstimator],
        validation: CausalDataset,
    ) -> dict[str, pd.DataFrame]:
        predictions = {}
        for estimator_id, model in cate_models.items():
            df_cate_effects = model.predict_effects(validation)
            predictions[estimator_id] = df_cate_effects
            write_dataframe(
                self.results.cate_dir(estimator_id) / "validation_predictions.parquet",
                df_cate_effects,
            )
        return predictions

    def persist_cate_evaluation(
        self,
        cate_evaluation: dict[str, list[ContrastEvaluation]],
    ) -> None:
        for estimator_id, contrasts in cate_evaluation.items():
            for contrast_result in contrasts:
                contrast_dir = (
                    self.results.cate_dir(estimator_id) / contrast_result.contrast
                )
                df_cate_evaluation = pd.DataFrame(
                    {
                        "eceth": [contrast_result.eceth],
                        "rate_autoc": [contrast_result.rate_autoc],
                        "rate_autoc_pvalue": [contrast_result.rate_autoc_pvalue],
                        "rate_qini": [contrast_result.rate_qini],
                        "rate_qini_pvalue": [contrast_result.rate_qini_pvalue],
                    }
                )
                write_dataframe(contrast_dir / "evaluation.csv", df_cate_evaluation)

    def build_ate_summary(
        self,
        ate_results: dict[str, pd.DataFrame],
        sensitivity_summaries: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        rows = []
        for estimator_id, estimates in ate_results.items():
            sensitivity = (
                sensitivity_summaries[estimator_id]
                if estimator_id in sensitivity_summaries
                else None
            )
            for _, estimate_row in estimates.iterrows():
                row = {
                    "estimator": estimator_id,
                    "contrast": estimate_row["contrast"],
                    "estimand": estimate_row["estimand"] if "estimand" in estimate_row else "ate",
                    "estimate": estimate_row["estimate"],
                    "ci_lower": estimate_row["ci_lower"],
                    "ci_upper": estimate_row["ci_upper"],
                    "robustness_value": None,
                    "benchmark_variable": None,
                    "benchmark_multiple_to_null": None,
                }
                if sensitivity is not None and not sensitivity.empty:
                    row["robustness_value"] = sensitivity["robustness_value"].iloc[0]
                    row["benchmark_variable"] = sensitivity["benchmark_variable"].iloc[0]
                    if "benchmark_multiple_to_null" in sensitivity.columns:
                        row["benchmark_multiple_to_null"] = sensitivity[
                            "benchmark_multiple_to_null"
                        ].iloc[0]
                rows.append(row)
        return pd.DataFrame(rows)

    def build_cate_summary(
        self,
        cate_ate_results: dict[str, pd.DataFrame],
        cate_evaluation: dict[str, list[ContrastEvaluation]],
    ) -> pd.DataFrame:
        rows = []
        for estimator_id, ate_table in cate_ate_results.items():
            contrast_evaluations = (
                cate_evaluation[estimator_id]
                if estimator_id in cate_evaluation
                else []
            )
            evaluation_by_contrast = {
                item.contrast: item for item in contrast_evaluations
            }
            for _, ate_row in ate_table.iterrows():
                contrast = ate_row["contrast"]
                evaluation = (
                    evaluation_by_contrast[contrast]
                    if contrast in evaluation_by_contrast
                    else None
                )
                rows.append(
                    {
                        "estimator": estimator_id,
                        "contrast": contrast,
                        "ate_estimate": ate_row["estimate"],
                        "ate_ci_lower": ate_row["ci_lower"],
                        "ate_ci_upper": ate_row["ci_upper"],
                        "eceth": evaluation.eceth if evaluation else None,
                        "rate_autoc": evaluation.rate_autoc if evaluation else None,
                        "rate_autoc_pvalue": evaluation.rate_autoc_pvalue
                        if evaluation
                        else None,
                        "rate_qini": evaluation.rate_qini if evaluation else None,
                        "rate_qini_pvalue": evaluation.rate_qini_pvalue
                        if evaluation
                        else None,
                    }
                )
        return pd.DataFrame(rows)
