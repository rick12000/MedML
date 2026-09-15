"""High-level causal analysis orchestration."""

from __future__ import annotations

import logging

import pandas as pd

from causal_pipeline.ate import ATESensitivityAnalyzer, BaseATEEstimator, create_ate_estimator
from causal_pipeline.cate import BaseCATEEstimator, create_cate_estimator
from causal_pipeline.config import PipelineConfig
from causal_pipeline.data import CausalDataset, DataSplitter
from causal_pipeline.diagnostics import DiagnosticsRunner
from causal_pipeline.evaluation import CATEEvaluator, ContrastEvaluation
from causal_pipeline.policy import PolicyService
from causal_pipeline.results import ResultStore

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
        self.diagnostics_runner = diagnostics_runner or DiagnosticsRunner(config)
        self.sensitivity_analyzer = sensitivity_analyzer or ATESensitivityAnalyzer(config)
        self.evaluator = evaluator or CATEEvaluator(config)
        self.policy_service = policy_service or PolicyService(config)
        self.results = results or ResultStore(config)

    def run(self, df_input: pd.DataFrame) -> None:
        logger.info("Starting causal pipeline run.")
        dataset = CausalDataset(data=self.config.data, df=df_input)
        partitions = self.splitter.split(dataset, self.config)
        self.results.save_partitions(partitions)

        diagnostics = self.diagnostics_runner.run(
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

        validation_scores = self.evaluator.build_robust_scores(partitions.validation)
        cate_evaluation = self.evaluator.evaluate_cate_models(
            predictions=validation_predictions,
            robust_scores=validation_scores,
            results_root=self.results.root,
        )

        policy_results = None
        if self.config.policy.enabled:
            policies = self.policy_service.fit_policy_methods(
                cate_models=cate_models,
                validation=partitions.validation,
                predictions=validation_predictions,
            )
            test_scores = self.evaluator.build_robust_scores(partitions.test)
            policy_results = self.policy_service.evaluate_policies(
                policies=policies,
                test=partitions.test,
                test_scores=test_scores,
            )
            self.results.write_dataframe("policy/summary.csv", policy_results)

        self.persist_cate_evaluation(cate_evaluation=cate_evaluation)

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
        for spec in self.config.ate_estimators:
            if not spec.enabled:
                continue
            logger.info("Fitting ATE estimator %s.", spec.kind.value)
            estimator = create_ate_estimator(
                spec=spec,
                data=train,
                sensitivity_outcome_learner=self.config.sensitivity.outcome_learner,
            )
            estimator.fit(train)
            models[spec.kind.value] = estimator
        return models

    def estimate_ate_models(self, ate_models: dict[str, BaseATEEstimator]) -> dict[str, pd.DataFrame]:
        results = {}
        for estimator_id, model in ate_models.items():
            results[estimator_id] = model.estimate()
            path = self.results.ate_dir(estimator_id) / "estimate.csv"
            results[estimator_id].to_csv(path, index=False)
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
        if not self.config.sensitivity.enabled:
            return {}
        summaries = {}
        for estimator_id, model in ate_models.items():
            logger.info("Running sensitivity analysis for %s.", estimator_id)
            result = self.sensitivity_analyzer.analyze(
                estimator=model,
                training_data=train,
                estimator_id=estimator_id,
                results_root=str(self.results.root),
            )
            sensitivity_dir = self.results.ate_dir(estimator_id) / "sensitivity"
            sensitivity_dir.mkdir(parents=True, exist_ok=True)
            result.summary.to_csv(sensitivity_dir / "summary.csv", index=False)
            result.benchmarks.to_csv(sensitivity_dir / "benchmarks.csv", index=False)
            summaries[estimator_id] = result.summary
        return summaries

    def fit_cate_estimators(self, train: CausalDataset) -> dict[str, BaseCATEEstimator]:
        models: dict[str, BaseCATEEstimator] = {}
        for spec in self.config.cate_estimators:
            if not spec.enabled:
                continue
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
            path = self.results.cate_dir(estimator_id) / "ate_estimate.csv"
            estimate.to_csv(path, index=False)
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
            path = self.results.cate_dir(estimator_id) / "validation_predictions.parquet"
            df_cate_effects.to_parquet(path)
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
                contrast_dir.mkdir(parents=True, exist_ok=True)
                df_cate_evaluation = pd.DataFrame(
                    {
                        "eceth": [contrast_result.eceth],
                        "rate_autoc": [contrast_result.rate_autoc],
                        "rate_autoc_pvalue": [contrast_result.rate_autoc_pvalue],
                        "rate_qini": [contrast_result.rate_qini],
                        "rate_qini_pvalue": [contrast_result.rate_qini_pvalue],
                    }
                )
                df_cate_evaluation.to_csv(contrast_dir / "evaluation.csv", index=False)

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
