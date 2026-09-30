"""High-level causal analysis orchestration."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable

import pandas as pd

from causal_pipeline.ate import (
    ATESensitivityAnalyzer,
    BaseATEEstimator,
    initialize_ate_estimator,
    estimator_supports_sensitivity,
)
from causal_pipeline.cate import create_cate_estimator, mean_cate_table
from causal_pipeline.config import (
    CATEEstimatorSpec,
    DEFAULT_PROPENSITY_CLIP,
    PipelineConfig,
)
from causal_pipeline.crossfit import cross_fit_predictions
from causal_pipeline.data import CausalDataset, DataSplitter
from causal_pipeline.diagnostics import DiagnosticsRunner
from causal_pipeline.evaluation import CATEEvaluator, ContrastEvaluation
from causal_pipeline.policy import PolicyService
from causal_pipeline.results import ResultStore
from causal_pipeline.selection import (
    plot_tipping_point,
    split_included_population,
    transport_included_effects,
    unseen_population_bound,
)
from causal_pipeline.utils import write_dataframe

logger = logging.getLogger(__name__)


def predict_cate_fold(
    train: CausalDataset,
    held_out: CausalDataset,
    spec: CATEEstimatorSpec,
) -> pd.DataFrame:
    estimator = create_cate_estimator(spec=spec, data=train)
    estimator.fit(data=train)
    return estimator.predict_effects(data=held_out)


def cate_fold_predictor(spec: CATEEstimatorSpec) -> Callable[[CausalDataset, CausalDataset], pd.DataFrame]:
    def predict_fold(train: CausalDataset, held_out: CausalDataset) -> pd.DataFrame:
        return predict_cate_fold(train=train, held_out=held_out, spec=spec)

    return predict_fold


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
        self.results = results or ResultStore(config=config)
        if diagnostics_runner is not None:
            self.diagnostics_runner = diagnostics_runner
        elif config.diagnostics is not None:
            self.diagnostics_runner = DiagnosticsRunner(diagnostics=config.diagnostics)
        else:
            self.diagnostics_runner = None
        if sensitivity_analyzer is not None:
            self.sensitivity_analyzer = sensitivity_analyzer
        elif config.sensitivity is not None:
            self.sensitivity_analyzer = ATESensitivityAnalyzer(sensitivity=config.sensitivity)
        else:
            self.sensitivity_analyzer = None
        if evaluator is not None:
            self.evaluator = evaluator
        elif config.cate_evaluation is not None:
            self.evaluator = CATEEvaluator(evaluation=config.cate_evaluation)
        else:
            self.evaluator = None
        if policy_service is not None:
            self.policy_service = policy_service
        elif config.policy is not None:
            self.policy_service = PolicyService(
                methods=config.policy,
                bootstrap_samples=config.policy_bootstrap_samples,
                random_state=config.policy_random_state,
            )
        else:
            self.policy_service = None

    def run(self, df_input: pd.DataFrame) -> None:
        logger.info("Starting causal pipeline run.")
        analysis_frame, excluded_frame = self.analysis_frame(df_input)
        dataset = CausalDataset(data=self.config.data, df=analysis_frame)
        partitions = self.splitter.split(dataset, self.config)
        self.results.save_partitions(partitions)

        estimation = partitions.estimation
        if self.config.diagnostics is not None:
            if self.diagnostics_runner is None:
                raise RuntimeError("Diagnostics are configured but no diagnostics runner is available.")
            diagnostics = self.diagnostics_runner.run(
                train=estimation,
                results_root=str(self.results.root),
            )
            self.results.save_diagnostics(diagnostics)

        ate_models = self.fit_ate_estimators(estimation)
        ate_results = self.estimate_ate_models(ate_models)
        sensitivity_summaries = self.run_ate_sensitivity(ate_models)
        self.run_selection_analyses(
            estimation=estimation,
            excluded=excluded_frame,
            ate_results=ate_results,
        )

        cate_predictions = self.cross_fit_cate_predictions(estimation)
        cate_ate_results = {
            estimator_id: mean_cate_table(predictions)
            for estimator_id, predictions in cate_predictions.items()
        }
        for estimator_id, estimate in cate_ate_results.items():
            write_dataframe(self.results.cate_dir(estimator_id) / "ate_estimate.csv", estimate)

        cate_evaluation: dict[str, list[ContrastEvaluation]] = {}
        estimation_scores = None
        if self.evaluator is not None and (cate_predictions or self.config.policy):
            estimation_scores = self.evaluator.build_robust_scores(dataset=estimation)
        if self.evaluator is not None and cate_predictions and estimation_scores is not None:
            cate_evaluation = self.evaluator.evaluate_cate_models(
                predictions=cate_predictions,
                robust_scores=estimation_scores,
                results_root=self.results.root,
            )
            self.persist_cate_evaluation(cate_evaluation=cate_evaluation)

        policy_results = None
        if self.config.policy:
            if self.evaluator is None or partitions.test is None:
                raise ValueError(
                    "Policy methods require cate_evaluation and a held-out test set."
                )
            if self.policy_service is None:
                raise RuntimeError("Policy methods are configured but no policy service is available.")
            policies = self.policy_service.fit_policy_methods(
                estimation=estimation,
                predictions=cate_predictions,
                robust_scores=estimation_scores,
            )
            test_scores = self.evaluator.build_robust_scores(dataset=partitions.test)
            test_cate_evaluation = self.evaluate_held_out_cate(
                estimation=estimation,
                test=partitions.test,
                test_scores=test_scores,
            )
            policy_results = self.policy_service.evaluate_policies(
                policies=policies,
                test=partitions.test,
                test_scores=test_scores,
            )
            self.results.write_dataframe("policy/summary.csv", policy_results)

        ate_summary = self.build_ate_summary(ate_results, sensitivity_summaries)
        cate_summary = self.build_cate_summary(
            cate_ate_results,
            cate_evaluation,
            test_cate_evaluation if self.config.policy else {},
        )
        self.results.write_summaries(
            ate_summary=ate_summary,
            cate_summary=cate_summary,
            policy_summary=policy_results,
        )
        logger.info("Causal pipeline run completed.")

    def analysis_frame(self, df_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        sensitivity = self.config.sensitivity
        if sensitivity is None or sensitivity.selection_column is None:
            return df_input.copy(), None
        data = self.config.data
        blocked = [data.outcome, data.treatment, *data.confounders, *data.effect_modifiers]
        included, excluded = split_included_population(
            df=df_input,
            selection_column=sensitivity.selection_column,
            blocked_columns=blocked,
        )
        logger.info(
            "Restricted the analysis to %s included patients and held out %s excluded patients.",
            len(included),
            len(excluded),
        )
        return included, excluded

    def run_selection_analyses(
        self,
        estimation: CausalDataset,
        excluded: pd.DataFrame | None,
        ate_results: dict[str, pd.DataFrame],
    ) -> None:
        sensitivity = self.config.sensitivity
        if sensitivity is None:
            return
        if (
            excluded is not None
            and len(excluded) > 0
            and sensitivity.sampling_learner is not None
            and sensitivity.propensity_learner is not None
        ):
            covariates = list(estimation.X_adjustment.columns)
            missing = [column for column in covariates if column not in excluded.columns]
            if missing:
                raise ValueError(f"Excluded patients are missing covariates: {missing}")
            transported = transport_included_effects(
                included_covariates=estimation.X_adjustment.copy(),
                included_treatment=estimation.treatment_series.copy(),
                included_outcome=estimation.outcome_series.copy(),
                excluded_covariates=excluded.loc[:, covariates].copy(),
                outcome_learner=sensitivity.outcome_learner,
                sampling_learner=sensitivity.sampling_learner,
                propensity_learner=sensitivity.propensity_learner,
                treatment_values=estimation.treatment_values,
                control_value=estimation.control_value,
                propensity_clip=DEFAULT_PROPENSITY_CLIP,
            )
            self.results.write_dataframe("selection/transport.csv", transported)
        if sensitivity.inclusion_fraction is None:
            return
        outcome_lower = None
        outcome_upper = None
        if sensitivity.outcome_bounds is not None:
            outcome_lower, outcome_upper = sensitivity.outcome_bounds
        for estimator_id, estimates in ate_results.items():
            rows = []
            for _, estimate_row in estimates.iterrows():
                bound = unseen_population_bound(
                    included_effect=float(estimate_row["estimate"]),
                    inclusion_fraction=sensitivity.inclusion_fraction,
                    outcome_lower=outcome_lower,
                    outcome_upper=outcome_upper,
                )
                rows.append(bound.model_dump())
                if outcome_lower is not None and outcome_upper is not None:
                    plot_tipping_point(
                        bound=bound,
                        outcome_lower=outcome_lower,
                        outcome_upper=outcome_upper,
                        save_path=str(
                            self.results.ate_dir(estimator_id)
                            / "selection"
                            / f"tipping_{estimate_row['contrast']}.png"
                        ),
                    )
            write_dataframe(
                self.results.ate_dir(estimator_id) / "selection" / "unseen.csv",
                pd.DataFrame(rows),
            )

    def fit_ate_estimators(self, train: CausalDataset) -> dict[str, BaseATEEstimator]:
        models: dict[str, BaseATEEstimator] = {}
        if not self.config.ate_estimators:
            return models
        for spec in self.config.ate_estimators:
            logger.info("Fitting ATE estimator %s.", spec.kind.value)
            estimator = initialize_ate_estimator(spec=spec, data=train)
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
    ) -> dict[str, pd.DataFrame]:
        if self.config.sensitivity is None or not ate_models:
            return {}
        if self.sensitivity_analyzer is None:
            raise RuntimeError("Sensitivity is configured but no sensitivity analyzer is available.")
        analyzer = self.sensitivity_analyzer
        summaries = {}
        for estimator_id, model in ate_models.items():
            if not estimator_supports_sensitivity(model):
                logger.info(
                    "Skipping sensitivity for %s. DoubleML score bounds are the supported analysis.",
                    estimator_id,
                )
                self.clear_sensitivity_output(estimator_id)
                continue
            logger.info("Running sensitivity analysis for %s.", estimator_id)
            result = analyzer.analyze(
                estimator=model,
                estimator_id=estimator_id,
                results_root=str(self.results.root),
            )
            if result is None:
                self.clear_sensitivity_output(estimator_id)
                continue
            sensitivity_dir = self.results.ate_dir(estimator_id) / "sensitivity"
            write_dataframe(sensitivity_dir / "summary.csv", result.summary)
            write_dataframe(sensitivity_dir / "benchmarks.csv", result.benchmarks)
            summaries[estimator_id] = result.summary
        return summaries

    def clear_sensitivity_output(self, estimator_id: str) -> None:
        """Drop a previous sensitivity folder so an unsupported estimator cannot keep an old bound."""
        sensitivity_dir = self.results.ate_dir(estimator_id) / "sensitivity"
        if sensitivity_dir.exists():
            shutil.rmtree(sensitivity_dir)

    def cross_fit_cate_predictions(self, estimation: CausalDataset) -> dict[str, pd.DataFrame]:
        predictions: dict[str, pd.DataFrame] = {}
        if not self.config.cate_estimators:
            return predictions
        for spec in self.config.cate_estimators:
            estimator_id = spec.kind.value
            logger.info("Cross-fitting CATE estimator %s.", estimator_id)
            df_effects = cross_fit_predictions(
                dataset=estimation,
                n_folds=self.config.cate_crossfit_folds,
                random_state=self.config.split.random_state,
                predict_fold=cate_fold_predictor(spec),
            )
            predictions[estimator_id] = df_effects
            write_dataframe(
                self.results.cate_dir(estimator_id) / "crossfit_predictions.parquet",
                df_effects,
            )
        return predictions

    def evaluate_held_out_cate(
        self,
        estimation: CausalDataset,
        test: CausalDataset,
        test_scores: pd.DataFrame,
    ) -> dict[str, list[ContrastEvaluation]]:
        """Score CATE models fit on the estimation sample against test-set robust scores."""
        if self.evaluator is None or not self.config.cate_estimators:
            return {}
        predictions = {
            spec.kind.value: predict_cate_fold(train=estimation, held_out=test, spec=spec)
            for spec in self.config.cate_estimators
        }
        evaluation = self.evaluator.evaluate_cate_models(
            predictions=predictions,
            robust_scores=test_scores,
            results_root=self.results.root,
            held_out=True,
        )
        self.persist_cate_evaluation(cate_evaluation=evaluation, table_name="test_evaluation.csv")
        return evaluation

    def persist_cate_evaluation(
        self,
        cate_evaluation: dict[str, list[ContrastEvaluation]],
        table_name: str = "evaluation.csv",
    ) -> None:
        for estimator_id, contrasts in cate_evaluation.items():
            for contrast_result in contrasts:
                contrast_dir = (
                    self.results.cate_dir(estimator_id) / contrast_result.contrast
                )
                df_cate_evaluation = pd.DataFrame(
                    {
                        "eceth": [contrast_result.eceth],
                        "eceth_standard_error": [contrast_result.eceth_standard_error],
                        "eceth_pvalue": [contrast_result.eceth_pvalue],
                        "eceth_tolerance": [contrast_result.eceth_tolerance],
                        "rate_autoc": [contrast_result.rate_autoc],
                        "rate_autoc_pvalue": [contrast_result.rate_autoc_pvalue],
                        "rate_qini": [contrast_result.rate_qini],
                        "rate_qini_pvalue": [contrast_result.rate_qini_pvalue],
                    }
                )
                write_dataframe(contrast_dir / table_name, df_cate_evaluation)

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
                    "estimand": estimate_row["estimand"],
                    "estimate": estimate_row["estimate"],
                    "ci_lower": estimate_row["ci_lower"],
                    "ci_upper": estimate_row["ci_upper"],
                    "robustness_value_point": None,
                    "robustness_value_interval": None,
                    "benchmark_variable": None,
                    "benchmark_multiple_to_null": None,
                    "benchmark_multiple_to_null_interval": None,
                }
                if sensitivity is not None and not sensitivity.empty:
                    row["robustness_value_point"] = sensitivity["robustness_value_point"].iloc[0]
                    row["robustness_value_interval"] = sensitivity["robustness_value_interval"].iloc[0]
                    row["benchmark_variable"] = sensitivity["benchmark_variable"].iloc[0]
                    row["benchmark_multiple_to_null"] = sensitivity["benchmark_multiple_to_null"].iloc[0]
                    row["benchmark_multiple_to_null_interval"] = sensitivity[
                        "benchmark_multiple_to_null_interval"
                    ].iloc[0]
                rows.append(row)
        return pd.DataFrame(rows)

    def build_cate_summary(
        self,
        cate_ate_results: dict[str, pd.DataFrame],
        cate_evaluation: dict[str, list[ContrastEvaluation]],
        test_cate_evaluation: dict[str, list[ContrastEvaluation]] | None = None,
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
            held_out_by_contrast = {
                item.contrast: item
                for item in (test_cate_evaluation or {}).get(estimator_id, [])
            }
            for _, ate_row in ate_table.iterrows():
                contrast = ate_row["contrast"]
                evaluation = (
                    evaluation_by_contrast[contrast]
                    if contrast in evaluation_by_contrast
                    else None
                )
                held_out = held_out_by_contrast[contrast] if contrast in held_out_by_contrast else None
                rows.append(
                    {
                        "estimator": estimator_id,
                        "contrast": contrast,
                        "mean_crossfit_cate": ate_row["estimate"],
                        "eceth": evaluation.eceth if evaluation else None,
                        "eceth_standard_error": evaluation.eceth_standard_error if evaluation else None,
                        "eceth_pvalue": evaluation.eceth_pvalue if evaluation else None,
                        "eceth_tolerance": evaluation.eceth_tolerance if evaluation else None,
                        "rate_autoc": evaluation.rate_autoc if evaluation else None,
                        "rate_autoc_pvalue": evaluation.rate_autoc_pvalue
                        if evaluation
                        else None,
                        "rate_qini": evaluation.rate_qini if evaluation else None,
                        "rate_qini_pvalue": evaluation.rate_qini_pvalue
                        if evaluation
                        else None,
                        "test_eceth": held_out.eceth if held_out else None,
                        "test_rate_autoc": held_out.rate_autoc if held_out else None,
                        "test_rate_qini": held_out.rate_qini if held_out else None,
                    }
                )
        return pd.DataFrame(rows)
