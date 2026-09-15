"""Policy trees, virtual twins, MOB, and test-set subgroup evaluation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from econml.policy import DRPolicyTree, PolicyTree
from pydantic import BaseModel, ConfigDict, Field
from sklearn.tree import DecisionTreeRegressor

from causal_pipeline.cate import BaseCATEEstimator
from causal_pipeline.config import JsonValue, PipelineConfig, build_sklearn_learner
from causal_pipeline.data import CausalDataset

if TYPE_CHECKING:
    from rpy2.robjects import RObject

    PolicyFitModel = PolicyTree | DRPolicyTree | DecisionTreeRegressor | RObject
else:
    PolicyFitModel = PolicyTree | DRPolicyTree | DecisionTreeRegressor

logger = logging.getLogger(__name__)


class PolicyRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str = Field(min_length=1)
    source_cate_model: str | None
    rule_id: int = Field(ge=0)
    rule: str = Field(min_length=1)
    recommended_treatment: JsonValue
    validation_n: int = Field(ge=0)
    subgroup_leaf: int | None = None


class FittedPolicy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    method: str = Field(min_length=1)
    source_cate_model: str | None
    model: PolicyFitModel
    rules: list[PolicyRule]
    feature_names: list[str] = Field(min_length=1)


class PolicyService:
    """Train policy/subgroup models on validation and evaluate on test."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def fit_policy_methods(
        self,
        cate_models: dict[str, BaseCATEEstimator],
        validation: CausalDataset,
        predictions: dict[str, pd.DataFrame],
    ) -> list[FittedPolicy]:
        X_mod = validation.X_effect_modifiers
        policies: list[FittedPolicy] = []

        if self.config.policy.standard_policy_tree:
            for estimator_id, df_cate_predictions in predictions.items():
                policies.append(
                    self.fit_standard_policy_tree(
                        estimator_id=estimator_id,
                        dataset=validation,
                        X_mod=X_mod,
                        predictions=df_cate_predictions,
                    )
                )

        if self.config.policy.dr_policy_tree:
            policies.append(self.fit_dr_policy_tree(validation))

        if self.config.policy.virtual_twins:
            for estimator_id, model in cate_models.items():
                effects = model.predict_effects(validation)
                policies.append(
                    self.fit_virtual_twins(
                        estimator_id=estimator_id,
                        dataset=validation,
                        X_mod=X_mod,
                        effects=effects,
                    )
                )

        if self.config.policy.mob:
            policies.append(self.fit_mob(validation))

        return policies

    def fit_standard_policy_tree(
        self,
        estimator_id: str,
        dataset: CausalDataset,
        X_mod: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> FittedPolicy:
        logger.info("Fitting standard PolicyTree for %s.", estimator_id)
        reward = np.zeros((len(X_mod), len(dataset.treatment_values)))
        control_index = dataset.treatment_values.index(
            dataset.control_value,
        )
        for column_index, column in enumerate(predictions.columns):
            arm_index = control_index + column_index + 1
            if arm_index < reward.shape[1]:
                reward[:, arm_index] = predictions[column].values
        tree = PolicyTree(**dict(self.config.policy.policy_tree_params))
        tree.fit(X_mod, reward)
        rules = self.extract_policy_tree_rules(
            tree=tree,
            X_mod=X_mod,
            method="policy_tree",
            source_cate_model=estimator_id,
            dataset=dataset,
        )
        return FittedPolicy(
            method="policy_tree",
            source_cate_model=estimator_id,
            model=tree,
            rules=rules,
            feature_names=list(X_mod.columns),
        )

    def fit_dr_policy_tree(self, validation: CausalDataset) -> FittedPolicy:
        logger.info("Fitting DRPolicyTree.")
        params = dict(self.config.policy.dr_policy_tree_params)
        model_regression = build_sklearn_learner(
            self.config.policy.dr_policy_regression_learner,
        )
        model_propensity = build_sklearn_learner(
            self.config.policy.dr_policy_propensity_learner,
        )
        tree = DRPolicyTree(
            model_regression=model_regression,
            model_propensity=model_propensity,
            **params,
        )
        tree.fit(
            Y=validation.outcome_series,
            T=validation.treatment_series,
            X=validation.X_effect_modifiers,
            W=validation.X_confounders,
        )
        rules = self.extract_policy_tree_rules(
            tree=tree.policy_tree_,
            X_mod=validation.X_effect_modifiers,
            method="dr_policy_tree",
            source_cate_model=None,
            dataset=validation,
        )
        return FittedPolicy(
            method="dr_policy_tree",
            source_cate_model=None,
            model=tree,
            rules=rules,
            feature_names=list(validation.X_effect_modifiers.columns),
        )

    def fit_virtual_twins(
        self,
        estimator_id: str,
        dataset: CausalDataset,
        X_mod: pd.DataFrame,
        effects: pd.DataFrame,
    ) -> FittedPolicy:
        logger.info("Fitting virtual twins for %s.", estimator_id)
        tree = DecisionTreeRegressor(**dict(self.config.policy.virtual_twins_params))
        tree.fit(X_mod, effects.values)
        rules = self.extract_sklearn_tree_rules(
            tree=tree,
            X_mod=X_mod,
            effects=effects,
            method="virtual_twins",
            source_cate_model=estimator_id,
            dataset=dataset,
        )
        return FittedPolicy(
            method="virtual_twins",
            source_cate_model=estimator_id,
            model=tree,
            rules=rules,
            feature_names=list(X_mod.columns),
        )

    def fit_mob(self, dataset: CausalDataset) -> FittedPolicy:
        logger.info("Fitting MOB via partykit.")
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.packages import importr
        except ImportError as error:
            raise ImportError("MOB requires optional dependency rpy2.") from error

        pandas2ri.activate()
        partykit = importr("partykit")
        df_validation = dataset.df.copy()
        ro.globalenv["validation_data"] = pandas2ri.py2rpy(df_validation)
        if dataset.outcome_type.value == "binary":
            ro.r(
                f"""
                mob_fit <- partykit::glmtree(
                    {dataset.outcome} ~ {dataset.treatment} + {" + ".join(dataset.confounders)} |
                        {" + ".join(dataset.X_effect_modifiers.columns)},
                    data = validation_data,
                    family = binomial(),
                    control = partykit::mob_control()
                )
                """
            )
        else:
            ro.r(
                f"""
                mob_fit <- partykit::lmtree(
                    {dataset.outcome} ~ {dataset.treatment} + {" + ".join(dataset.confounders)} |
                        {" + ".join(dataset.X_effect_modifiers.columns)},
                    data = validation_data,
                    control = partykit::mob_control()
                )
                """
            )
        rules = [
            PolicyRule(
                method="mob",
                source_cate_model=None,
                rule_id=0,
                rule="mob_tree",
                recommended_treatment=dataset.treatment_values[-1],
                validation_n=len(df_validation),
                subgroup_leaf=None,
            )
        ]
        return FittedPolicy(
            method="mob",
            source_cate_model=None,
            model=ro.r["mob_fit"],
            rules=rules,
            feature_names=list(dataset.X_effect_modifiers.columns),
        )

    def extract_policy_tree_rules(
        self,
        tree: PolicyTree,
        X_mod: pd.DataFrame,
        method: str,
        source_cate_model: str | None,
        dataset: CausalDataset,
    ) -> list[PolicyRule]:
        leaf_ids = tree.apply(X_mod)
        rules = []
        for rule_id, leaf in enumerate(sorted(set(leaf_ids))):
            mask = leaf_ids == leaf
            recommended = int(np.argmax(tree.predict(X_mod.iloc[mask])[0]))
            rules.append(
                PolicyRule(
                    method=method,
                    source_cate_model=source_cate_model,
                    rule_id=rule_id,
                    rule=f"leaf_{leaf}",
                    recommended_treatment=dataset.treatment_values[recommended],
                    validation_n=int(mask.sum()),
                    subgroup_leaf=int(leaf),
                )
            )
        return rules

    def extract_sklearn_tree_rules(
        self,
        tree: DecisionTreeRegressor,
        X_mod: pd.DataFrame,
        effects: pd.DataFrame,
        method: str,
        source_cate_model: str | None,
        dataset: CausalDataset,
    ) -> list[PolicyRule]:
        leaf_ids = tree.apply(X_mod)
        rules = []
        for rule_id, leaf in enumerate(sorted(set(leaf_ids))):
            mask = leaf_ids == leaf
            mean_effects = effects.iloc[mask].mean(axis=0)
            best_index = int(np.argmax(mean_effects.values))
            recommended_arm = dataset.treatment_values[best_index + 1]
            rules.append(
                PolicyRule(
                    method=method,
                    source_cate_model=source_cate_model,
                    rule_id=rule_id,
                    rule=f"leaf_{leaf}",
                    recommended_treatment=recommended_arm,
                    validation_n=int(mask.sum()),
                    subgroup_leaf=int(leaf),
                )
            )
        return rules

    def subgroup_leaf_ids(
        self,
        policy: FittedPolicy,
        X_mod: pd.DataFrame,
    ) -> np.ndarray | None:
        if policy.method == "policy_tree":
            return np.asarray(policy.model.apply(X_mod))
        if policy.method == "dr_policy_tree":
            return np.asarray(policy.model.policy_tree_.apply(X_mod))
        if policy.method == "virtual_twins":
            return np.asarray(policy.model.apply(X_mod))
        return None

    def evaluate_policies(
        self,
        policies: list[FittedPolicy],
        test: CausalDataset,
        test_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        X_mod = test.X_effect_modifiers
        rows = []
        prevalence = len(test.df) / max(len(test.df), 1)
        for policy in policies:
            leaf_ids = self.subgroup_leaf_ids(policy=policy, X_mod=X_mod)
            for rule in policy.rules:
                if leaf_ids is None:
                    subgroup_scores = test_scores
                    test_n = len(test_scores)
                else:
                    if rule.subgroup_leaf is None:
                        continue
                    mask = leaf_ids == rule.subgroup_leaf
                    subgroup_scores = test_scores.iloc[mask]
                    test_n = int(mask.sum())
                if test_n == 0:
                    continue
                effect, lower, upper = self.bootstrap_subgroup_effect(
                    subgroup_scores=subgroup_scores,
                )
                rows.append(
                    {
                        "method": policy.method,
                        "source_cate_model": policy.source_cate_model,
                        "rule_id": rule.rule_id,
                        "rule": rule.rule,
                        "recommended_treatment": rule.recommended_treatment,
                        "test_n": test_n,
                        "test_prevalence": prevalence,
                        "effect_estimate": effect,
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
        return pd.DataFrame(rows)

    def bootstrap_subgroup_effect(
        self,
        subgroup_scores: pd.DataFrame,
    ) -> tuple[float, float, float]:
        column = subgroup_scores.columns[0]
        values = subgroup_scores[column].values
        point = float(np.mean(values))
        bootstrap = []
        rng = np.random.default_rng(self.config.cate_evaluation.random_state)
        for replicate in range(self.config.cate_evaluation.rate_bootstrap_samples):
            indices = rng.choice(len(values), size=len(values), replace=True)
            bootstrap.append(float(np.mean(values[indices])))
        lower = float(np.percentile(bootstrap, 2.5))
        upper = float(np.percentile(bootstrap, 97.5))
        return point, lower, upper
