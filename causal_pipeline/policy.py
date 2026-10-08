"""Policy trees, virtual twins, MOB, and test-set subgroup evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from econml.policy import DRPolicyTree, PolicyTree
from pydantic import BaseModel, ConfigDict, Field
from sklearn.tree import DecisionTreeRegressor

from causal_pipeline.config import (
    DRPolicyTreeMethodSpec,
    ExactPolicyTreeMethodSpec,
    JsonValue,
    MOBMethodSpec,
    OutcomeFavorability,
    PolicyKind,
    PolicyMethodSpec,
    PolicySearchMode,
    PolicyTreeMethodSpec,
    VirtualTwinsMethodSpec,
    clone_estimator,
)
from causal_pipeline.scaling import clone_scaled_estimator
from causal_pipeline.data import CausalDataset, contrast_columns
from causal_pipeline.policy_tree import (
    ExactPolicyTree,
    capital_reward_matrix,
    method_name,
    offset_treatment_rewards,
)
from causal_pipeline.virtual_twins import QUALIFYING_UNION_RULE, VIRTUAL_TWINS_METHOD, VirtualTwins

if TYPE_CHECKING:
    from rpy2.robjects import RObject

    PolicyFitModel = PolicyTree | DRPolicyTree | DecisionTreeRegressor | ExactPolicyTree | RObject
else:
    PolicyFitModel = PolicyTree | DRPolicyTree | DecisionTreeRegressor | ExactPolicyTree

logger = logging.getLogger(__name__)

BOOTSTRAP_CI_LOWER_PERCENTILE = 2.5
BOOTSTRAP_CI_UPPER_PERCENTILE = 97.5
INTERVAL_STATUS_DEFINITIONAL = "definitional"
INTERVAL_STATUS_BOOTSTRAP = "percentile_bootstrap"
MOB_ARM_COLUMN = "mob_non_control"


@dataclass(frozen=True)
class SubgroupInterval:
    estimate: float
    ci_lower: float
    ci_upper: float
    interval_status: str


class PolicyRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str = Field(min_length=1)
    source_cate_model: str | None
    rule_id: int = Field(ge=0)
    rule: str = Field(min_length=1)
    recommended_treatment: JsonValue
    learning_n: int = Field(ge=0)
    subgroup_leaf: int | None = None
    included_leaves: tuple[int, ...] | None = None
    learning_effect: float | None = None
    debiased_effect: float | None = None


class FittedPolicy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    method: str = Field(min_length=1)
    source_cate_model: str | None
    model: PolicyFitModel
    rules: list[PolicyRule]
    feature_names: list[str] = Field(min_length=1)


class PolicyService:
    """Train policy rules on cross-fit learning-sample rewards and evaluate them on test."""

    def __init__(
        self,
        methods: list[PolicyMethodSpec],
        bootstrap_samples: int,
        random_state: int,
    ) -> None:
        self.methods = methods
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state

    def fit_policy_methods(
        self,
        estimation: CausalDataset,
        predictions: dict[str, pd.DataFrame],
        robust_scores: pd.DataFrame | None = None,
    ) -> list[FittedPolicy]:
        X_mod = estimation.X_effect_modifiers.copy()
        policies: list[FittedPolicy] = []

        for method in self.methods:
            if method.kind == PolicyKind.POLICY_TREE:
                for estimator_id in predictions:
                    policies.append(
                        self.fit_standard_policy_tree(
                            spec=method,
                            estimator_id=estimator_id,
                            dataset=estimation,
                            X_mod=X_mod,
                            robust_scores=robust_scores,
                        )
                    )
            elif method.kind == PolicyKind.EXACT_POLICY_TREE:
                for estimator_id, df_cate_predictions in predictions.items():
                    policies.append(
                        self.fit_exact_policy_tree(
                            spec=method,
                            estimator_id=estimator_id,
                            dataset=estimation,
                            X_mod=X_mod,
                            predictions=df_cate_predictions,
                        )
                    )
            elif method.kind == PolicyKind.DR_POLICY_TREE:
                policies.append(self.fit_dr_policy_tree(spec=method, dataset=estimation))
            elif method.kind == PolicyKind.VIRTUAL_TWINS:
                for estimator_id, df_cate_predictions in predictions.items():
                    policies.append(
                        self.fit_virtual_twins(
                            spec=method,
                            estimator_id=estimator_id,
                            dataset=estimation,
                            X_mod=X_mod,
                            effects=df_cate_predictions,
                            robust_scores=robust_scores,
                        )
                    )
            elif method.kind == PolicyKind.MOB:
                policies.append(self.fit_mob(spec=method, dataset=estimation))

        return policies

    def fit_standard_policy_tree(
        self,
        spec: PolicyTreeMethodSpec,
        estimator_id: str,
        dataset: CausalDataset,
        X_mod: pd.DataFrame,
        robust_scores: pd.DataFrame | None,
    ) -> FittedPolicy:
        logger.info("Fitting standard PolicyTree for %s.", estimator_id)
        if robust_scores is None:
            raise ValueError("The policy tree reward is the cross-fit doubly robust score.")
        reward = reward_matrix(
            predictions=robust_scores,
            treatment_values=dataset.treatment_values,
            control_value=dataset.control_value,
            direction=reward_direction(dataset.data.outcome_favorability),
        )
        tree = clone_estimator(spec.tree)
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

    def fit_exact_policy_tree(
        self,
        spec: ExactPolicyTreeMethodSpec,
        estimator_id: str,
        dataset: CausalDataset,
        X_mod: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> FittedPolicy:
        logger.info("Fitting exact %s policy tree for %s.", spec.mode.value, estimator_id)
        control_action = dataset.treatment_values.index(dataset.control_value)
        direction = reward_direction(dataset.data.outcome_favorability)
        if spec.mode == PolicySearchMode.CAPITAL:
            contrast_name = contrast_columns(dataset.control_value, dataset.treatment_values)[0]
            rewards = capital_reward_matrix(
                effects=direction * predictions[contrast_name].to_numpy(dtype=float),
                treatment_values=dataset.treatment_values,
                control_value=dataset.control_value,
                clinical_threshold=spec.clinical_threshold,
                negative_effect_penalty=spec.negative_effect_penalty,
            )
        else:
            rewards = offset_treatment_rewards(
                rewards=reward_matrix(
                    predictions=predictions,
                    treatment_values=dataset.treatment_values,
                    control_value=dataset.control_value,
                    direction=direction,
                ),
                minimum_effect=spec.minimum_effect,
                control_action=control_action,
            )
        tree = ExactPolicyTree(
            mode=spec.mode,
            depth=spec.depth,
            min_node_size=spec.min_node_size,
            split_step=spec.split_step,
            feature_names=list(X_mod.columns),
        )
        tree.fit(features=X_mod, rewards=rewards, control_action=control_action)
        rules = self.extract_exact_policy_rules(
            tree=tree,
            X_mod=X_mod,
            method=method_name(spec.mode),
            source_cate_model=estimator_id,
            dataset=dataset,
        )
        return FittedPolicy(
            method=method_name(spec.mode),
            source_cate_model=estimator_id,
            model=tree,
            rules=rules,
            feature_names=list(X_mod.columns),
        )

    def fit_dr_policy_tree(
        self,
        spec: DRPolicyTreeMethodSpec,
        dataset: CausalDataset,
    ) -> FittedPolicy:
        logger.info("Fitting DRPolicyTree.")
        levels = control_first_arms(dataset.treatment_values, dataset.control_value)
        tree = DRPolicyTree(
            model_regression=clone_scaled_estimator(spec.outcome_learner),
            model_propensity=clone_scaled_estimator(spec.propensity_learner),
            categories=levels,
        )
        controls = dataset.X_controls.copy()
        groups = dataset.df[dataset.group_id] if dataset.group_id is not None else None
        tree.fit(
            Y=dataset.outcome_series.copy(),
            T=dataset.treatment_series.copy(),
            X=dataset.X_effect_modifiers.copy(),
            W=controls if controls.shape[1] else None,
            groups=groups,
        )
        rules = self.extract_policy_tree_rules(
            tree=tree.policy_model_,
            X_mod=dataset.X_effect_modifiers,
            method="dr_policy_tree",
            source_cate_model=None,
            dataset=dataset,
            action_levels=levels,
        )
        return FittedPolicy(
            method="dr_policy_tree",
            source_cate_model=None,
            model=tree,
            rules=rules,
            feature_names=list(dataset.X_effect_modifiers.columns),
        )

    def fit_virtual_twins(
        self,
        spec: VirtualTwinsMethodSpec,
        estimator_id: str,
        dataset: CausalDataset,
        X_mod: pd.DataFrame,
        effects: pd.DataFrame,
        robust_scores: pd.DataFrame | None,
    ) -> FittedPolicy:
        logger.info("Fitting virtual twins for %s.", estimator_id)
        tree = clone_estimator(spec.tree)
        if not isinstance(tree, DecisionTreeRegressor):
            raise TypeError("Virtual twins require a regression tree.")
        twins = VirtualTwins(
            tree=tree,
            minimum_effect=spec.minimum_effect,
            bootstrap_samples=self.bootstrap_samples,
            random_state=self.random_state,
        )
        twins.fit(features=X_mod, effects=effects, robust_scores=robust_scores)
        if twins.model is None:
            raise RuntimeError("Virtual twins did not fit a tree.")
        rules = self.extract_virtual_twin_rules(
            twins=twins,
            X_mod=X_mod,
            effects=effects,
            source_cate_model=estimator_id,
            dataset=dataset,
        )
        return FittedPolicy(
            method=VIRTUAL_TWINS_METHOD,
            source_cate_model=estimator_id,
            model=twins.model,
            rules=rules,
            feature_names=list(X_mod.columns),
        )

    def fit_mob(self, spec: MOBMethodSpec, dataset: CausalDataset) -> FittedPolicy:
        logger.info("Fitting MOB via partykit (%s).", spec.kind.value)
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.packages import importr
        except ImportError as error:
            raise ImportError("MOB requires optional dependency rpy2.") from error

        if len(dataset.treatment_values) != 2:
            raise NotImplementedError("MOB policies are implemented for binary treatment.")
        from rpy2.robjects.conversion import localconverter

        importr("partykit")
        learning_frame = dataset.df.copy()
        learning_frame[MOB_ARM_COLUMN] = (
            learning_frame[dataset.treatment] != dataset.control_value
        ).astype(int)
        formula = mob_formula(dataset)
        family = "family = binomial()," if dataset.outcome_type.value == "binary" else ""
        tree_function = "glmtree" if dataset.outcome_type.value == "binary" else "lmtree"
        with localconverter(pandas2ri.converter):
            ro.globalenv["learning_data"] = ro.conversion.py2rpy(learning_frame)
            ro.r(
                f"""
                mob_fit <- partykit::{tree_function}(
                    {formula},
                    data = learning_data,
                    {family}
                    control = partykit::mob_control()
                )
                mob_nodes <- predict(mob_fit, type = "node")
                mob_coef <- as.data.frame(coef(mob_fit))
                mob_coef$node <- as.integer(rownames(coef(mob_fit)))
                """
            )
            leaf_ids = np.asarray(ro.r("mob_nodes"), dtype=int)
            coefficients = ro.conversion.rpy2py(ro.r("mob_coef"))
        rules = mob_policy_rules(
            leaf_ids=leaf_ids,
            coefficients=coefficients,
            dataset=dataset,
            direction=reward_direction(dataset.data.outcome_favorability),
        )
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
        action_levels: list[JsonValue] | None = None,
    ) -> list[PolicyRule]:
        leaf_ids = np.asarray(tree.apply(X_mod))
        actions = np.asarray(tree.predict(X_mod))
        levels = dataset.treatment_values if action_levels is None else action_levels
        rules = []
        for rule_id, leaf in enumerate(sorted(set(leaf_ids))):
            mask = leaf_ids == leaf
            recommended = int(actions[mask][0])
            rules.append(
                PolicyRule(
                    method=method,
                    source_cate_model=source_cate_model,
                    rule_id=rule_id,
                    rule=f"leaf_{leaf}",
                    recommended_treatment=levels[recommended],
                    learning_n=int(mask.sum()),
                    subgroup_leaf=int(leaf),
                )
            )
        return rules

    def extract_exact_policy_rules(
        self,
        tree: ExactPolicyTree,
        X_mod: pd.DataFrame,
        method: str,
        source_cate_model: str | None,
        dataset: CausalDataset,
    ) -> list[PolicyRule]:
        leaf_ids = np.asarray(tree.apply(X_mod))
        actions = np.asarray(tree.predict(X_mod))
        descriptions = tree.leaf_descriptions()
        rules = []
        for rule_id, leaf in enumerate(sorted(set(leaf_ids.tolist()))):
            mask = leaf_ids == leaf
            recommended = int(actions[mask][0])
            rules.append(
                PolicyRule(
                    method=method,
                    source_cate_model=source_cate_model,
                    rule_id=rule_id,
                    rule=descriptions[int(leaf)],
                    recommended_treatment=dataset.treatment_values[recommended],
                    learning_n=int(mask.sum()),
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
        leaf_ids = np.asarray(tree.apply(X_mod))
        rules = []
        for rule_id, leaf in enumerate(sorted(set(leaf_ids))):
            mask = leaf_ids == leaf
            recommended_arm = best_treatment_arm(
                mean_effects=effects.iloc[mask].mean(axis=0),
                treatment_values=dataset.treatment_values,
                control_value=dataset.control_value,
                direction=reward_direction(dataset.data.outcome_favorability),
            )
            rules.append(
                PolicyRule(
                    method=method,
                    source_cate_model=source_cate_model,
                    rule_id=rule_id,
                    rule=f"leaf_{leaf}",
                    recommended_treatment=recommended_arm,
                    learning_n=int(mask.sum()),
                    subgroup_leaf=int(leaf),
                )
            )
        return rules

    def extract_virtual_twin_rules(
        self,
        twins: VirtualTwins,
        X_mod: pd.DataFrame,
        effects: pd.DataFrame,
        source_cate_model: str | None,
        dataset: CausalDataset,
    ) -> list[PolicyRule]:
        if twins.model is None:
            raise RuntimeError("Virtual twins did not fit a tree.")
        leaf_ids = np.asarray(twins.model.apply(X_mod))
        qualifying = set(twins.qualifying_leaves)
        rules = []
        for leaf in sorted(qualifying):
            mask = leaf_ids == leaf
            if not np.any(mask):
                continue
            recommended_arm = best_treatment_arm(
                mean_effects=effects.iloc[mask].mean(axis=0),
                treatment_values=dataset.treatment_values,
                control_value=dataset.control_value,
                direction=reward_direction(dataset.data.outcome_favorability),
            )
            rules.append(
                PolicyRule(
                    method=VIRTUAL_TWINS_METHOD,
                    source_cate_model=source_cate_model,
                    rule_id=len(rules),
                    rule=f"leaf_{leaf}",
                    recommended_treatment=recommended_arm,
                    learning_n=int(mask.sum()),
                    subgroup_leaf=int(leaf),
                )
            )
        debiasing = twins.debiasing
        if debiasing is None or debiasing.empty:
            return rules
        treated = [value for value in dataset.treatment_values if value != dataset.control_value]
        contrast_arms = dict(
            zip(
                contrast_columns(dataset.control_value, dataset.treatment_values),
                treated,
                strict=True,
            )
        )
        union_n = int(np.isin(leaf_ids, list(qualifying)).sum())
        for row in debiasing.itertuples(index=False):
            contrast = str(row.contrast)
            if contrast not in contrast_arms:
                continue
            rules.append(
                PolicyRule(
                    method=VIRTUAL_TWINS_METHOD,
                    source_cate_model=source_cate_model,
                    rule_id=len(rules),
                    rule=QUALIFYING_UNION_RULE,
                    recommended_treatment=contrast_arms[contrast],
                    learning_n=union_n,
                    included_leaves=twins.qualifying_leaves,
                    learning_effect=float(row.naive_effect),
                    debiased_effect=float(row.debiased_effect),
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
        if policy.method in {"exact_policy_welfare", "exact_policy_capital"}:
            return np.asarray(policy.model.apply(X_mod))
        if policy.method == "dr_policy_tree":
            return np.asarray(policy.model.policy_model_.apply(X_mod))
        if policy.method == "virtual_twins":
            return np.asarray(policy.model.apply(X_mod))
        if policy.method == "mob":
            return mob_leaf_ids(policy=policy, X_mod=X_mod)
        return None

    def evaluate_policies(
        self,
        policies: list[FittedPolicy],
        test: CausalDataset,
        test_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        X_mod = test.X_effect_modifiers
        rows = []
        n_test = len(test.df)
        for policy in policies:
            leaf_ids = self.subgroup_leaf_ids(policy=policy, X_mod=X_mod)
            for rule in policy.rules:
                if leaf_ids is None:
                    subgroup_scores = test_scores
                    test_n = len(test_scores)
                else:
                    if rule.included_leaves is not None:
                        mask = np.isin(leaf_ids, list(rule.included_leaves))
                    elif rule.subgroup_leaf is None:
                        continue
                    else:
                        mask = leaf_ids == rule.subgroup_leaf
                    subgroup_scores = test_scores.iloc[mask]
                    test_n = int(mask.sum())
                if test_n == 0:
                    continue
                interval = self.bootstrap_subgroup_effect(
                    subgroup_scores=subgroup_scores,
                    recommended_treatment=rule.recommended_treatment,
                    control_value=test.control_value,
                )
                rows.append(
                    {
                        "method": policy.method,
                        "source_cate_model": policy.source_cate_model,
                        "rule_id": rule.rule_id,
                        "rule": rule.rule,
                        "recommended_treatment": rule.recommended_treatment,
                        "test_n": test_n,
                        "test_prevalence": test_n / n_test,
                        "effect_estimate": interval.estimate,
                        "ci_lower_given_rule": interval.ci_lower,
                        "ci_upper_given_rule": interval.ci_upper,
                        "interval_status": interval.interval_status,
                        "learning_effect": np.nan if rule.learning_effect is None else rule.learning_effect,
                        "debiased_learning_effect": (
                            np.nan if rule.debiased_effect is None else rule.debiased_effect
                        ),
                    }
                )
        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary["n_rules_scored"] = len(summary)
        return summary

    def bootstrap_subgroup_effect(
        self,
        subgroup_scores: pd.DataFrame,
        recommended_treatment: JsonValue,
        control_value: JsonValue,
    ) -> SubgroupInterval:
        if recommended_treatment == control_value:
            return SubgroupInterval(
                estimate=0.0,
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                interval_status=INTERVAL_STATUS_DEFINITIONAL,
            )
        column = contrast_column_name(recommended_treatment, control_value, list(subgroup_scores.columns))
        values = subgroup_scores[column].to_numpy(dtype=float)
        point = float(np.mean(values))
        bootstrap = []
        rng = np.random.default_rng(self.random_state)
        for replicate in range(self.bootstrap_samples):
            indices = rng.choice(len(values), size=len(values), replace=True)
            bootstrap.append(float(np.mean(values[indices])))
        lower = float(np.percentile(bootstrap, BOOTSTRAP_CI_LOWER_PERCENTILE))
        upper = float(np.percentile(bootstrap, BOOTSTRAP_CI_UPPER_PERCENTILE))
        return SubgroupInterval(
            estimate=point,
            ci_lower=lower,
            ci_upper=upper,
            interval_status=INTERVAL_STATUS_BOOTSTRAP,
        )


def reward_direction(favorability: OutcomeFavorability) -> float:
    if favorability == OutcomeFavorability.LOWER_IS_BETTER:
        return -1.0
    return 1.0


def control_first_arms(
    treatment_values: list[JsonValue],
    control_value: JsonValue,
) -> list[JsonValue]:
    return [control_value] + [arm for arm in treatment_values if arm != control_value]


def reward_matrix(
    predictions: pd.DataFrame,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    direction: float = 1.0,
) -> np.ndarray:
    reward = np.zeros((len(predictions), len(treatment_values)))
    non_control = [value for value in treatment_values if value != control_value]
    for name, arm in zip(contrast_columns(control_value, treatment_values), non_control, strict=True):
        reward[:, treatment_values.index(arm)] = direction * predictions[name].to_numpy(dtype=float)
    return reward


def best_treatment_arm(
    mean_effects: pd.Series,
    treatment_values: list[JsonValue],
    control_value: JsonValue,
    direction: float = 1.0,
) -> JsonValue:
    scores = {control_value: 0.0}
    non_control = [value for value in treatment_values if value != control_value]
    for name, arm in zip(contrast_columns(control_value, treatment_values), non_control, strict=True):
        scores[arm] = direction * float(mean_effects[name])
    return max(scores, key=scores.get)


def contrast_column_name(
    recommended_treatment: JsonValue,
    control_value: JsonValue,
    columns: list[str],
) -> str:
    name = contrast_columns(control_value, [control_value, recommended_treatment])[0]
    if name not in columns:
        raise KeyError(f"Doubly robust scores have no column {name}.")
    return name


def mob_formula(dataset: CausalDataset) -> str:
    """Regress on every confounder. Partition only on modifiers that are not confounders."""
    confounders = set(dataset.confounders)
    regressors = [MOB_ARM_COLUMN] + list(dataset.confounders)
    partitions = [column for column in dataset.effect_modifiers if column not in confounders]
    if not partitions:
        raise ValueError(
            "MOB needs an effect modifier that is not also a confounder. "
            "Confounders stay in the leaf regression, including those listed as effect modifiers."
        )
    return f"{dataset.outcome} ~ {' + '.join(regressors)} | {' + '.join(partitions)}"


def mob_policy_rules(
    leaf_ids: np.ndarray,
    coefficients: pd.DataFrame,
    dataset: CausalDataset,
    direction: float = 1.0,
) -> list[PolicyRule]:
    coefficient_column = next(
        (
            column
            for column in coefficients.columns
            if column == MOB_ARM_COLUMN or str(column).startswith(MOB_ARM_COLUMN)
        ),
        None,
    )
    if coefficient_column is None or "node" not in coefficients.columns:
        raise RuntimeError("partykit did not return a treatment coefficient for each terminal node.")
    treated = next(value for value in dataset.treatment_values if value != dataset.control_value)
    rules = []
    for rule_id, leaf in enumerate(sorted(set(leaf_ids.tolist()))):
        node_rows = coefficients.loc[coefficients["node"] == int(leaf)]
        if node_rows.empty:
            raise RuntimeError(f"No MOB coefficient row for terminal node {leaf}.")
        recommend_treated = direction * float(node_rows.iloc[0][coefficient_column]) > 0.0
        rules.append(
            PolicyRule(
                method="mob",
                source_cate_model=None,
                rule_id=rule_id,
                rule=f"leaf_{leaf}",
                recommended_treatment=treated if recommend_treated else dataset.control_value,
                learning_n=int(np.sum(leaf_ids == leaf)),
                subgroup_leaf=int(leaf),
            )
        )
    return rules


def mob_leaf_ids(policy: FittedPolicy, X_mod: pd.DataFrame) -> np.ndarray:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter):
        ro.globalenv["mob_score_data"] = ro.conversion.py2rpy(X_mod)
        ro.globalenv["mob_fit"] = policy.model
        return np.asarray(ro.r('predict(mob_fit, newdata = mob_score_data, type = "node")'), dtype=int)
