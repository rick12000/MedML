"""Rubin's rules for scalars that already have a within-imputation variance."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm, t

from causal_pipeline.evaluation import ContrastEvaluation
from causal_pipeline.utils import read_dataframe, write_dataframe

logger = logging.getLogger(__name__)

WALD_SYMMETRY_RTOL = 1e-4
WALD_SYMMETRY_ATOL = 1e-6
POOLING_PARAMETER_COUNT = 1
ZERO_VARIANCE = 0.0


class PooledScalar(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimate: float
    within_variance: float
    between_variance: float
    total_variance: float
    degrees_of_freedom: float
    relative_increase: float
    fraction_missing_information: float
    ci_lower: float
    ci_upper: float
    imputations: int


def wald_variance(estimate: float, lower: float, upper: float, level: float) -> float:
    """Recover the sampling variance from a symmetric normal interval."""
    if not np.isfinite([estimate, lower, upper]).all():
        raise ValueError("A Wald interval needs a finite estimate and finite bounds.")
    span_below = estimate - lower
    span_above = upper - estimate
    if not np.isclose(span_below, span_above, rtol=WALD_SYMMETRY_RTOL, atol=WALD_SYMMETRY_ATOL):
        raise ValueError(
            "The interval is not symmetric about the estimate, so it is not a Wald variance."
        )
    half_width = (upper - lower) / 2.0
    if half_width == 0.0:
        return ZERO_VARIANCE
    quantile = float(norm.ppf(0.5 + level / 2.0))
    return float((half_width / quantile) ** 2)


def pool_scalar(
    estimates: np.ndarray,
    variances: np.ndarray,
    sample_size: int,
    level: float,
    parameter_count: int = POOLING_PARAMETER_COUNT,
) -> PooledScalar:
    """Barnard–Rubin t interval, matching mice::pool.scalar.

    A zero between-imputation variance uses the normal interval. A zero within
    variance uses Rubin's M - 1 degrees of freedom: the Barnard–Rubin adjustment
    collapses to zero and cannot form a t interval. Non-positive complete-data
    degrees of freedom likewise fall back to that Rubin denominator.
    """
    points = np.asarray(estimates, dtype=float)
    within = np.asarray(variances, dtype=float)
    imputation_count = int(points.size)
    if imputation_count < 2:
        raise ValueError("Rubin's rules need at least two imputations.")
    if within.shape != points.shape:
        raise ValueError("Estimates and variances must have the same shape.")
    if not np.isfinite(points).all() or not np.isfinite(within).all():
        raise ValueError("Estimates and variances must be finite.")
    if np.any(within < 0.0):
        raise ValueError("Variances must be non-negative.")
    estimate = float(np.mean(points))
    within_variance = float(np.mean(within))
    between_variance = float(np.var(points, ddof=1))
    total_variance = within_variance + (1.0 + 1.0 / imputation_count) * between_variance
    degrees_of_freedom, relative_increase, fraction_missing = barnard_rubin_degrees(
        within_variance=within_variance,
        between_variance=between_variance,
        total_variance=total_variance,
        imputation_count=imputation_count,
        sample_size=sample_size,
        parameter_count=parameter_count,
    )
    half_width = interval_half_width(total_variance, degrees_of_freedom, level)
    return PooledScalar(
        estimate=estimate,
        within_variance=within_variance,
        between_variance=between_variance,
        total_variance=total_variance,
        degrees_of_freedom=degrees_of_freedom,
        relative_increase=relative_increase,
        fraction_missing_information=fraction_missing,
        ci_lower=estimate - half_width,
        ci_upper=estimate + half_width,
        imputations=imputation_count,
    )


def barnard_rubin_degrees(
    within_variance: float,
    between_variance: float,
    total_variance: float,
    imputation_count: int,
    sample_size: int,
    parameter_count: int,
) -> tuple[float, float, float]:
    """Return degrees of freedom, relative variance increase, and fraction missing.

    The finite-sample branch is Barnard and Rubin (1999), as in mice::pool.scalar:
    dfcom is sample size minus parameter count. Zero between-imputation variance
    is the normal limit. Zero within variance, and non-positive dfcom, use
    Rubin's (M - 1) / lambda^2 degrees of freedom.
    """
    if between_variance == 0.0:
        return float("inf"), 0.0, 0.0
    inflation = 1.0 + 1.0 / imputation_count
    if within_variance == 0.0:
        degrees = float(imputation_count - 1)
        return degrees, float("inf"), 1.0
    relative_increase = inflation * between_variance / within_variance
    missing_fraction = inflation * between_variance / total_variance
    old_degrees = (imputation_count - 1) / missing_fraction**2
    complete_degrees = sample_size - parameter_count
    if complete_degrees <= 0 or not np.isfinite(complete_degrees):
        degrees = old_degrees
    else:
        observed_degrees = (
            (complete_degrees + 1.0)
            / (complete_degrees + 3.0)
            * complete_degrees
            * (1.0 - missing_fraction)
        )
        degrees = old_degrees * observed_degrees / (old_degrees + observed_degrees)
    fraction_missing_information = (relative_increase + 2.0 / (degrees + 3.0)) / (
        relative_increase + 1.0
    )
    return float(degrees), float(relative_increase), float(fraction_missing_information)


def interval_half_width(total_variance: float, degrees_of_freedom: float, level: float) -> float:
    if total_variance == 0.0:
        return 0.0
    scale = float(np.sqrt(total_variance))
    if np.isfinite(degrees_of_freedom):
        quantile = float(t.ppf(0.5 + level / 2.0, degrees_of_freedom))
    else:
        quantile = float(norm.ppf(0.5 + level / 2.0))
    return quantile * scale


def pooled_columns(result: PooledScalar) -> dict[str, float | int]:
    return {
        "estimate": result.estimate,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "within_variance": result.within_variance,
        "between_variance": result.between_variance,
        "total_variance": result.total_variance,
        "degrees_of_freedom": result.degrees_of_freedom,
        "relative_increase": result.relative_increase,
        "fraction_missing_information": result.fraction_missing_information,
        "imputations": result.imputations,
    }


def reference_sample_size(row_count: int, cluster_count: int | None) -> int:
    if cluster_count is None:
        return row_count
    return cluster_count


def align_on(frames: list[pd.DataFrame], keys: list[str]) -> list[pd.DataFrame]:
    indexed = [frame.set_index(keys, drop=False) for frame in frames]
    order = indexed[0].index
    return [frame.loc[order] for frame in indexed]


def sampling_variance(row: pd.Series, level: float) -> float:
    """Use a published standard error. A symmetric Wald interval is the fallback."""
    if "standard_error" in row.index and pd.notna(row["standard_error"]):
        return float(row["standard_error"]) ** 2
    return wald_variance(
        estimate=float(row["estimate"]),
        lower=float(row["ci_lower"]),
        upper=float(row["ci_upper"]),
        level=level,
    )


def pool_keyed_estimates(
    frames: list[pd.DataFrame],
    keys: list[str],
    level: float,
    sample_size: int,
) -> pd.DataFrame:
    """Pool rows that share `keys`. A standard error is the within variance when present."""
    if not frames:
        raise ValueError("Pooling needs one estimate table per imputation.")
    indexed = align_on(frames, keys)
    rows = []
    for key in indexed[0].index:
        members = [frame.loc[key] for frame in indexed]
        template = members[0]
        estimates = [float(row["estimate"]) for row in members]
        variances = [sampling_variance(row, level) for row in members]
        pooled = pool_scalar(
            estimates=np.asarray(estimates),
            variances=np.asarray(variances),
            sample_size=sample_size,
            level=level,
        )
        drop_labels = ["estimate", "ci_lower", "ci_upper"]
        if "standard_error" in template.index:
            drop_labels.append("standard_error")
        combined = template.drop(labels=drop_labels).to_dict()
        combined.update(pooled_columns(pooled))
        rows.append(combined)
    return pd.DataFrame(rows)


def average_aligned_frames(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positional mean and between-imputation standard deviation of numeric frames."""
    if len(frames) < 2:
        raise ValueError("Averaging imputations needs at least two frames.")
    column_lists = [list(frame.columns) for frame in frames]
    if any(columns != column_lists[0] for columns in column_lists):
        raise ValueError("Imputed prediction frames do not share columns.")
    if any(len(frame) != len(frames[0]) for frame in frames):
        raise ValueError("Imputed prediction frames do not share a row count.")
    reference_index = frames[0].index
    if any(not frame.index.equals(reference_index) for frame in frames):
        raise ValueError("Imputed prediction frames do not share a row index.")
    stack = np.stack([frame.to_numpy(dtype=float) for frame in frames], axis=0)
    mean = pd.DataFrame(np.mean(stack, axis=0), index=reference_index, columns=frames[0].columns)
    between = pd.DataFrame(
        np.std(stack, axis=0, ddof=1),
        index=reference_index,
        columns=frames[0].columns,
    )
    return mean, between


def pool_subgroup_table(
    frames: list[pd.DataFrame],
    level_column: str,
    sample_size: int,
) -> pd.DataFrame:
    """Pool effect, split-treatment, and group-outcome differences. Counts stay at their cross-imputation mean.

    ``effect_se`` and ``outcome_se`` become the square root of Rubin's total
    variance. The intervals and p-values use the Barnard–Rubin t reference.
    Rows with no cutoff are the overall ATE: only the effect contrast is pooled.
    """
    rows = []
    partitions = (
        (
            [frame.loc[frame["cutoff"].notna()] for frame in frames],
            ["estimator", "contrast", "column", "cutoff"],
            ("effect", "treatment", "outcome"),
        ),
        (
            [frame.loc[frame["cutoff"].isna()] for frame in frames],
            ["estimator", "contrast"],
            ("effect",),
        ),
    )
    for subset, keys, prefixes in partitions:
        if subset[0].empty:
            continue
        indexed = align_on(subset, keys)
        for key in indexed[0].index:
            members = [frame.loc[key] for frame in indexed]
            template = members[0].to_dict()
            level = float(members[0][level_column])
            for prefix in prefixes:
                pooled = pool_scalar(
                    estimates=np.asarray([float(row[f"{prefix}_difference"]) for row in members]),
                    variances=np.square([float(row[f"{prefix}_se"]) for row in members]),
                    sample_size=sample_size,
                    level=level,
                )
                template[f"{prefix}_difference"] = pooled.estimate
                template[f"{prefix}_se"] = float(np.sqrt(pooled.total_variance))
                template[f"{prefix}_ci_lower"] = pooled.ci_lower
                template[f"{prefix}_ci_upper"] = pooled.ci_upper
                template[f"{prefix}_pvalue"] = two_sided_tail(
                    estimate=pooled.estimate,
                    standard_error=float(np.sqrt(pooled.total_variance)),
                    degrees_of_freedom=pooled.degrees_of_freedom,
                )
                if "cutoff" in keys:
                    template[f"{prefix}_above"] = float(
                        np.mean([float(row[f"{prefix}_above"]) for row in members])
                    )
                    template[f"{prefix}_below"] = float(
                        np.mean([float(row[f"{prefix}_below"]) for row in members])
                    )
            if "cutoff" in keys:
                template["n_above"] = int(
                    round(float(np.mean([float(row["n_above"]) for row in members])))
                )
                template["n_below"] = int(
                    round(float(np.mean([float(row["n_below"]) for row in members])))
                )
            template["imputations"] = len(members)
            rows.append(template)
    return pd.DataFrame(rows)


def upper_tail(statistic: float, degrees_of_freedom: float) -> float:
    if np.isfinite(degrees_of_freedom):
        return float(t.sf(statistic, degrees_of_freedom))
    return float(norm.sf(statistic))


def lower_tail(statistic: float, degrees_of_freedom: float) -> float:
    if np.isfinite(degrees_of_freedom):
        return float(t.cdf(statistic, degrees_of_freedom))
    return float(norm.cdf(statistic))


def two_sided_tail(estimate: float, standard_error: float, degrees_of_freedom: float) -> float:
    if standard_error == 0.0:
        return 0.0 if estimate != 0.0 else 1.0
    return 2.0 * upper_tail(abs(estimate / standard_error), degrees_of_freedom)


def one_sided_tail(
    estimate: float,
    threshold: float,
    standard_error: float,
    degrees_of_freedom: float,
    lower: bool,
) -> float:
    """Barnard–Rubin tail. Infinite degrees of freedom is the normal tail."""
    if standard_error == 0.0:
        beyond = estimate < threshold if lower else estimate > threshold
        return 0.0 if beyond else 1.0
    statistic = (estimate - threshold) / standard_error
    if lower:
        return lower_tail(statistic, degrees_of_freedom)
    return upper_tail(statistic, degrees_of_freedom)


def pool_contrast_evaluation(
    evaluations: list[ContrastEvaluation],
    sample_size: int,
    level: float,
    tolerance: float,
) -> ContrastEvaluation:
    """Rubin-pool ECETH. AUTOC and Qini stay as means, with no pooled sampling result.

    The RATE standard error in this pipeline is the spread of half-samples, not a
    sampling variance, so Rubin's rules do not apply to it.
    """
    first = evaluations[0]
    if any(item.contrast != first.contrast for item in evaluations):
        raise ValueError("Contrast evaluations must share a contrast.")
    eceth = pool_scalar(
        estimates=np.asarray([item.eceth for item in evaluations]),
        variances=np.square([item.eceth_standard_error for item in evaluations]),
        sample_size=sample_size,
        level=level,
    )
    eceth_se = float(np.sqrt(eceth.total_variance))
    return ContrastEvaluation(
        contrast=first.contrast,
        eceth=eceth.estimate,
        eceth_standard_error=eceth_se,
        eceth_pvalue=one_sided_tail(
            estimate=eceth.estimate,
            threshold=tolerance,
            standard_error=eceth_se,
            degrees_of_freedom=eceth.degrees_of_freedom,
            lower=True,
        ),
        eceth_tolerance=tolerance,
        rate_autoc=float(np.mean([item.rate_autoc for item in evaluations])),
        rate_autoc_standard_error=float("nan"),
        rate_autoc_pvalue=float("nan"),
        rate_qini=float(np.mean([item.rate_qini for item in evaluations])),
        rate_qini_standard_error=float("nan"),
        rate_qini_pvalue=float("nan"),
        calibration_data_path=first.calibration_data_path,
        toc_data_path=first.toc_data_path,
    )


def pool_point_columns(
    frames: list[pd.DataFrame],
    keys: list[str],
    values: list[str],
) -> pd.DataFrame:
    """Mean and between-imputation SD for columns that have no sampling variance."""
    if not frames:
        raise ValueError("Point pooling needs one table per imputation.")
    stacked = pd.concat(frames, ignore_index=True)
    grouped = stacked.groupby(keys, dropna=False, sort=False)
    means = grouped[values].mean().reset_index()
    spread = grouped[values].std(ddof=1).reset_index()
    renamed = spread.rename(columns={column: f"{column}_between_sd" for column in values})
    return means.merge(renamed, on=keys, how="left")


def pool_curve(frames: list[pd.DataFrame], x_column: str, y_columns: list[str]) -> pd.DataFrame:
    """Pointwise mean and between-imputation SD on a shared grid."""
    grids = [frame[x_column].to_numpy(dtype=float) for frame in frames]
    if any(not np.array_equal(grid, grids[0]) for grid in grids):
        raise ValueError(f"Curve grids in {x_column} do not match across imputations.")
    averaged = frames[0][[x_column]].copy()
    for column in y_columns:
        stack = np.vstack([frame[column].to_numpy(dtype=float) for frame in frames])
        averaged[column] = np.mean(stack, axis=0)
        averaged[f"{column}_between_sd"] = np.std(stack, axis=0, ddof=1)
    return averaged


def pool_nuisance_screens(imputation_roots: list[Path], destination: Path) -> None:
    """Average screen tables. Column-drop intervals are left missing."""
    relatives = nuisance_tables(imputation_roots)
    if not relatives:
        return
    metric_rows: list[pd.DataFrame] = []
    for relative in relatives:
        frames = [
            read_dataframe(root / relative)
            for root in imputation_roots
            if (root / relative).exists()
        ]
        if len(frames) != len(imputation_roots):
            logger.warning("Skipping %s because an imputation did not write it.", relative)
            continue
        pooled = pool_nuisance_table(relative.name, frames)
        if pooled is None:
            continue
        write_dataframe(destination / relative, pooled)
        if relative.name == "metrics.csv":
            labeled = pooled.copy()
            labeled.insert(0, "screen", relative.parent.as_posix())
            metric_rows.append(labeled)
    if metric_rows:
        write_dataframe(
            destination / "nuisance_screen" / "metrics_summary.csv",
            pd.concat(metric_rows, ignore_index=True),
        )


def nuisance_tables(imputation_roots: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in imputation_roots:
        screen_root = root / "nuisance_screen"
        if not screen_root.exists():
            continue
        for path in screen_root.rglob("*.csv"):
            found.add(path.relative_to(root))
    return sorted(found, key=lambda path: path.as_posix())


def pool_nuisance_table(name: str, frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    if name == "metrics.csv":
        return pool_metric_rows(frames)
    if name == "column_drop_importance.csv":
        return pool_column_drop(frames)
    if name == "gain_importance.csv":
        return pool_point_columns(frames, keys=["feature"], values=["gain_importance"])
    if name == "correlation.csv":
        return pool_correlation(frames)
    if name == "shap_values.csv":
        return pool_shap(frames)
    return pool_numeric_mean(frames)


def pool_metric_rows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    pooled = pool_point_columns(frames, keys=["parameter"], values=["value"])
    pooled = pooled.rename(columns={"value": "estimate", "value_between_sd": "between_sd"})
    pooled["imputations"] = len(frames)
    return pooled


def pool_column_drop(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average column-drop means. Repeat standard errors are not Rubin's within variances."""
    logger.warning(
        "Column-drop intervals are left missing. "
        "A repeat standard error is not a sampling variance for Rubin's rules."
    )
    value_columns = [column for column in frames[0].columns if column != "feature"]
    pooled = pool_point_columns(frames, keys=["feature"], values=value_columns)
    interval_columns = [
        column
        for column in value_columns
        if column.startswith("sem_") or column.startswith("ci95_")
    ]
    pooled[interval_columns] = np.nan
    between_interval = [f"{column}_between_sd" for column in interval_columns]
    pooled = pooled.drop(columns=between_interval)
    order = next(column for column in pooled.columns if column.startswith("mean_"))
    return pooled.sort_values(order, ascending=False)


def pool_correlation(frames: list[pd.DataFrame]) -> pd.DataFrame:
    id_columns = ["alias", "variable"]
    value_columns = [column for column in frames[0].columns if column not in id_columns]
    pooled = pool_point_columns(frames, keys=id_columns, values=value_columns)
    between_columns = [column for column in pooled.columns if column.endswith("_between_sd")]
    return pooled.drop(columns=between_columns)


def pool_shap(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        logger.warning("SHAP tables differ in shape across imputations and were left unpooled.")
        return None
    numeric = [column for column in frames[0].columns if pd.api.types.is_numeric_dtype(frames[0][column])]
    keys = [column for column in frames[0].columns if column not in numeric]
    if not keys:
        stack = np.stack([frame[numeric].to_numpy(dtype=float) for frame in frames], axis=0)
        return pd.DataFrame(np.mean(stack, axis=0), columns=numeric)
    pooled = pool_point_columns(frames, keys=keys, values=numeric)
    between_columns = [column for column in pooled.columns if column.endswith("_between_sd")]
    return pooled.drop(columns=between_columns)


def pool_numeric_mean(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    numeric = [column for column in frames[0].columns if pd.api.types.is_numeric_dtype(frames[0][column])]
    keys = [column for column in frames[0].columns if column not in numeric]
    if keys and numeric:
        return pool_point_columns(frames, keys=keys, values=numeric)
    same_shape = all(
        frame.shape == frames[0].shape and list(frame.columns) == list(frames[0].columns)
        for frame in frames
    )
    if numeric and same_shape:
        stack = np.stack([frame[numeric].to_numpy(dtype=float) for frame in frames], axis=0)
        return pd.DataFrame(np.mean(stack, axis=0), columns=numeric, index=frames[0].index)
    logger.warning("A nuisance table has no shared key and was left unpooled.")
    return None
