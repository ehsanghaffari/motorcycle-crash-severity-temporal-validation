#!/usr/bin/env python3
"""
Rebuild two publication-quality fatal-class SHAP interaction figures used in
supporting analysis for the motorcycle crash-severity study:

1. Road geometry × primary crash cause
2. Seasonal timing (month sine + cosine) × lighting condition

The script is intended to run after ``motorcycle_crash_pipeline.py``. It uses
saved outputs from the primary analysis, verifies the pipeline hash and package
versions recorded in ``00_run_manifest.json``, reloads the fitted XGBoost
pipeline, reproduces the grouped interaction values, and exports detailed
category-level figures and validation tables.

Required primary-analysis outputs
---------------------------------
00_run_manifest.json
audit/motorcycle_cohort_analysis_ready.csv
audit/primary_feature_set.json
models/final_stable_feature_uncalibrated_pipeline.joblib
shap/fatal_shap_interaction_ranking.csv

The raw crash data are not distributed with this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import xgboost
import lightgbm
from scipy import sparse


FATAL_CLASS = 2
TEMPORAL_TEST_YEAR = 98
DEFAULT_SAMPLE_SIZE = 600
DEFAULT_SAMPLE_SEED = 42 + 91000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild publication-quality fatal SHAP interaction figures."
    )
    parser.add_argument(
        "--primary-output",
        type=Path,
        default=Path("outputs") / "main_analysis",
        help="Primary-analysis output directory (default: outputs/main_analysis).",
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=None,
        help="Path to motorcycle_crash_pipeline.py. Defaults to the script in this folder.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("outputs") / "interaction_figures",
        help="Directory for rebuilt figure products (default: outputs/interaction_figures).",
    )
    parser.add_argument(
        "--min-cell-count",
        type=int,
        default=5,
        help="Minimum observations required to display a category-pair heatmap cell.",
    )
    parser.add_argument(
        "--max-causes",
        type=int,
        default=12,
        help="Maximum primary-crash-cause categories displayed in the main heatmap.",
    )
    parser.add_argument(
        "--min-lighting-total",
        type=int,
        default=10,
        help="Minimum total observations required to display a lighting category.",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Allow execution despite package-version differences. Not recommended.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_required_files(output_dir: Path, pipeline_script: Path) -> None:
    required = [
        output_dir / "00_run_manifest.json",
        output_dir / "audit" / "motorcycle_cohort_analysis_ready.csv",
        output_dir / "audit" / "primary_feature_set.json",
        output_dir / "models" / "final_stable_feature_uncalibrated_pipeline.joblib",
        output_dir / "shap" / "fatal_shap_interaction_ranking.csv",
        pipeline_script,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n- " + "\n- ".join(missing)
        )


def current_versions() -> dict[str, str]:
    return {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "shap": shap.__version__,
        "joblib": joblib.__version__,
    }


def verify_provenance(
    manifest: dict[str, Any],
    pipeline_script: Path,
    allow_version_mismatch: bool,
) -> None:
    expected_hash = manifest.get("script_sha256")
    actual_hash = sha256_file(pipeline_script)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            "Pipeline-script hash mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            "Use the exact script from the primary run."
        )

    expected_python = str(manifest.get("python", "")).split()[0]
    actual_python = platform.python_version()
    expected_packages = manifest.get("packages", {})
    actual_packages = current_versions()

    mismatches: list[str] = []
    if expected_python and actual_python != expected_python:
        mismatches.append(f"Python: expected {expected_python}, found {actual_python}")
    for name, expected in expected_packages.items():
        actual = actual_packages.get(name)
        if actual is not None and str(actual) != str(expected):
            mismatches.append(f"{name}: expected {expected}, found {actual}")

    if mismatches and not allow_version_mismatch:
        raise RuntimeError(
            "Environment mismatch detected. Re-run in the primary analysis "
            "environment or recreate it with the recorded versions:\n- "
            + "\n- ".join(mismatches)
        )
    if mismatches:
        warnings.warn(
            "Proceeding with version mismatches:\n- " + "\n- ".join(mismatches),
            RuntimeWarning,
        )


def import_pipeline_module(script_path: Path):
    """Import the exact script under the module name required by joblib."""
    module_name = "motorcycle_crash_pipeline"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def recover_calendar_month(month_sin: pd.Series, month_cos: pd.Series) -> pd.Series:
    sin_values = pd.to_numeric(month_sin, errors="coerce")
    cos_values = pd.to_numeric(month_cos, errors="coerce")
    valid = sin_values.notna() & cos_values.notna()

    result = pd.Series(pd.NA, index=month_sin.index, dtype="Int64")
    angles = np.mod(
        np.arctan2(sin_values.loc[valid].to_numpy(), cos_values.loc[valid].to_numpy()),
        2 * np.pi,
    )
    month_zero_based = np.rint(angles / (2 * np.pi / 12)).astype(int) % 12
    result.loc[valid] = month_zero_based + 1
    return result


def compute_grouped_interactions(
    module,
    fitted_pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_features: list[str],
    numeric_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    sample_indices = module.stratified_sample_indices(
        y_test,
        max_samples=min(DEFAULT_SAMPLE_SIZE, len(y_test)),
        seed=DEFAULT_SAMPLE_SEED,
        include_all_class=FATAL_CLASS,
    )
    X_sample = X_test.iloc[sample_indices].copy()
    y_sample = y_test[sample_indices]

    transformed, encoded_names, encoded_groups = (
        module.transformed_matrix_and_metadata(
            fitted_pipeline,
            X_sample,
            categorical_features,
            numeric_features,
        )
    )
    dense = transformed.toarray() if sparse.issparse(transformed) else np.asarray(transformed)

    estimator = module.unwrap_tree_estimator(fitted_pipeline)
    explainer = shap.TreeExplainer(estimator)
    raw = explainer.shap_interaction_values(dense)
    standardized = module.standardize_interaction_values(
        raw,
        n_samples=dense.shape[0],
        n_features=dense.shape[1],
        n_classes=3,
    )
    fatal_interactions = standardized[:, :, :, FATAL_CLASS]
    group_matrix, group_names = module.grouping_matrix(encoded_groups)
    grouped = np.einsum(
        "fi,sfj,jg->sig",
        group_matrix,
        fatal_interactions,
        group_matrix,
    )

    metadata = pd.DataFrame(
        {
            "temporal_test_row_index": sample_indices,
            "observed_class": y_sample,
            "observed_severity": [
                module.INTERNAL_TO_NAME[int(value)] for value in y_sample
            ],
        }
    )
    return X_sample.reset_index(drop=True), grouped, group_names, metadata


def validate_against_reference_ranking(
    output_dir: Path,
    grouped: np.ndarray,
    group_names: list[str],
) -> pd.DataFrame:
    original = pd.read_csv(
        output_dir / "shap" / "fatal_shap_interaction_ranking.csv"
    )
    checks = []
    for feature_1, feature_2 in [
        ("Vaze_Hendese", "Elat_Tame"),
        ("month_sin", "Roshanaei"),
        ("month_cos", "Roshanaei"),
    ]:
        i = group_names.index(feature_1)
        j = group_names.index(feature_2)
        recomputed_abs = float(np.mean(np.abs(grouped[:, i, j])))
        recomputed_signed = float(np.mean(grouped[:, i, j]))

        match = original[
            (
                (original["feature_1"] == feature_1)
                & (original["feature_2"] == feature_2)
            )
            | (
                (original["feature_1"] == feature_2)
                & (original["feature_2"] == feature_1)
            )
        ]
        if match.empty:
            original_abs = np.nan
            original_signed = np.nan
        else:
            original_abs = float(
                match.iloc[0]["mean_absolute_fatal_interaction"]
            )
            original_signed = float(
                match.iloc[0]["mean_signed_fatal_interaction"]
            )

        checks.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "recomputed_mean_absolute": recomputed_abs,
                "original_mean_absolute": original_abs,
                "absolute_difference": abs(recomputed_abs - original_abs)
                if np.isfinite(original_abs)
                else np.nan,
                "recomputed_mean_signed": recomputed_signed,
                "original_mean_signed": original_signed,
                "signed_difference": recomputed_signed - original_signed
                if np.isfinite(original_signed)
                else np.nan,
            }
        )
    return pd.DataFrame(checks)


def geometry_cause_outputs(
    module,
    X_sample: pd.DataFrame,
    interaction_values: np.ndarray,
    output_dir: Path,
    min_cell_count: int,
    max_causes: int,
) -> None:
    geometry = X_sample["Vaze_Hendese"].fillna("Missing/Unreported").astype(str)
    cause = X_sample["Elat_Tame"].fillna("Missing/Unreported").astype(str)

    raw = pd.DataFrame(
        {
            "road_geometry": geometry,
            "primary_crash_cause": cause,
            "signed_fatal_interaction": interaction_values,
        }
    )
    raw.to_csv(
        output_dir / "road_geometry_by_primary_cause_row_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        raw.groupby(["road_geometry", "primary_crash_cause"], observed=True)
        .agg(
            n=("signed_fatal_interaction", "size"),
            mean_signed_interaction=("signed_fatal_interaction", "mean"),
            sd_signed_interaction=("signed_fatal_interaction", "std"),
            mean_absolute_interaction=(
                "signed_fatal_interaction",
                lambda values: float(np.mean(np.abs(values))),
            ),
        )
        .reset_index()
    )
    summary.to_csv(
        output_dir / "road_geometry_by_primary_cause_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cause_totals = raw["primary_crash_cause"].value_counts()
    displayed_causes = list(cause_totals.head(max_causes).index)

    geometry_order = list(
        module.ENGLISH_CODE_MAPS.get("Vaze_Hendese", {}).values()
    )
    observed_geometries = list(pd.unique(geometry))
    displayed_geometries = [
        value for value in geometry_order if value in observed_geometries
    ]
    displayed_geometries.extend(
        value
        for value in observed_geometries
        if value not in displayed_geometries
    )

    plotted = summary[
        summary["primary_crash_cause"].isin(displayed_causes)
        & summary["road_geometry"].isin(displayed_geometries)
    ].copy()

    mean_pivot = plotted.pivot(
        index="road_geometry",
        columns="primary_crash_cause",
        values="mean_signed_interaction",
    ).reindex(index=displayed_geometries, columns=displayed_causes)
    count_pivot = plotted.pivot(
        index="road_geometry",
        columns="primary_crash_cause",
        values="n",
    ).reindex(index=displayed_geometries, columns=displayed_causes)

    display_values = mean_pivot.where(count_pivot >= min_cell_count)
    matrix = np.ma.masked_invalid(display_values.to_numpy(dtype=float))
    finite_values = display_values.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    limit = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    if limit == 0:
        limit = 1.0

    fig, ax = plt.subplots(
        figsize=(max(11, 0.75 * len(displayed_causes)), max(6, 0.65 * len(displayed_geometries)))
    )
    image = ax.imshow(matrix, aspect="auto", vmin=-limit, vmax=limit)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean signed fatal-class SHAP interaction")

    ax.set_xticks(np.arange(len(displayed_causes)))
    ax.set_xticklabels(displayed_causes, rotation=70, ha="right")
    ax.set_yticks(np.arange(len(displayed_geometries)))
    ax.set_yticklabels(displayed_geometries)
    ax.set_xlabel("Primary crash cause")
    ax.set_ylabel("Road geometry")
    ax.set_title("Fatal-class interaction: road geometry × primary crash cause")

    for row_idx in range(len(displayed_geometries)):
        for col_idx in range(len(displayed_causes)):
            value = display_values.iloc[row_idx, col_idx]
            count = count_pivot.iloc[row_idx, col_idx]
            if pd.notna(value) and pd.notna(count):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}\n(n={int(count)})",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )

    figure_base = output_dir / "Figure_7a_geometry_by_primary_cause"
    save_figure(fig, figure_base)


def seasonal_lighting_outputs(
    X_sample: pd.DataFrame,
    combined_interaction: np.ndarray,
    output_dir: Path,
    min_lighting_total: int,
) -> None:
    month = recover_calendar_month(
        X_sample["month_sin"],
        X_sample["month_cos"],
    )
    lighting = X_sample["Roshanaei"].fillna("Missing/Unreported").astype(str)

    raw = pd.DataFrame(
        {
            "calendar_month": month,
            "lighting_condition": lighting,
            "signed_fatal_interaction": combined_interaction,
        }
    ).dropna(subset=["calendar_month"])
    raw["calendar_month"] = raw["calendar_month"].astype(int)
    raw.to_csv(
        output_dir / "seasonal_timing_by_lighting_row_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        raw.groupby(["calendar_month", "lighting_condition"], observed=True)
        .agg(
            n=("signed_fatal_interaction", "size"),
            mean_signed_interaction=("signed_fatal_interaction", "mean"),
            sd_signed_interaction=("signed_fatal_interaction", "std"),
        )
        .reset_index()
    )
    summary["standard_error"] = (
        summary["sd_signed_interaction"] / np.sqrt(summary["n"])
    )
    summary["ci95_lower"] = (
        summary["mean_signed_interaction"]
        - 1.96 * summary["standard_error"]
    )
    summary["ci95_upper"] = (
        summary["mean_signed_interaction"]
        + 1.96 * summary["standard_error"]
    )
    summary.loc[summary["n"] < 2, ["standard_error", "ci95_lower", "ci95_upper"]] = np.nan
    summary.to_csv(
        output_dir / "seasonal_timing_by_lighting_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    totals = raw["lighting_condition"].value_counts()
    displayed_lighting = list(totals[totals >= min_lighting_total].index)

    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in displayed_lighting:
        subset = summary[
            summary["lighting_condition"] == condition
        ].sort_values("calendar_month")
        ax.errorbar(
            subset["calendar_month"],
            subset["mean_signed_interaction"],
            yerr=1.96 * subset["standard_error"],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=f"{condition} (n={int(totals[condition])})",
        )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Mean signed fatal-class SHAP interaction")
    ax.set_title(
        "Fatal-class interaction: seasonal timing × lighting condition"
    )
    ax.legend(title="Lighting condition", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    figure_base = output_dir / "Figure_7b_seasonal_timing_by_lighting"
    save_figure(fig, figure_base)


def write_captions(output_dir: Path, min_cell_count: int) -> None:
    captions = f"""Figure 7a. Fatal-class SHAP interaction between road geometry and primary crash cause. Cells report the mean signed grouped SHAP interaction contribution for each observed category combination; sample counts are shown in parentheses. Combinations with fewer than {min_cell_count} sampled records are left blank to reduce interpretation of sparse cells. Positive and negative values indicate upward and downward contributions, respectively, to the fitted model's fatal-class output. The values describe model behavior and should not be interpreted causally.

Figure 7b. Fatal-class SHAP interaction between seasonal timing and lighting condition. Seasonal interaction contributions combine the month-sine × lighting and month-cosine × lighting components and are summarized by reconstructed calendar month. Points show mean signed grouped SHAP interaction contributions; error bars show approximate 95% confidence intervals based on the sampled records. The values describe model behavior rather than causal effects, and sparse month-lighting combinations should be interpreted cautiously.
"""
    (output_dir / "FIGURE_CAPTIONS.txt").write_text(captions, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.primary_output.expanduser().resolve()
    pipeline_script = (
        args.pipeline.expanduser().resolve()
        if args.pipeline is not None
        else (Path(__file__).resolve().parent / "motorcycle_crash_pipeline.py")
    )
    figure_output = args.figure_output.expanduser().resolve()
    verify_required_files(output_dir, pipeline_script)

    manifest = load_json(output_dir / "00_run_manifest.json")
    verify_provenance(
        manifest,
        pipeline_script,
        allow_version_mismatch=args.allow_version_mismatch,
    )
    module = import_pipeline_module(pipeline_script)

    feature_set = load_json(output_dir / "audit" / "primary_feature_set.json")
    categorical_features = list(feature_set["primary_categorical_features"])
    numeric_features = list(feature_set["primary_numeric_features"])
    model_features = [*categorical_features, *numeric_features]

    cohort = pd.read_csv(
        output_dir / "audit" / "motorcycle_cohort_analysis_ready.csv",
        encoding="utf-8-sig",
    )
    test = cohort.loc[
        pd.to_numeric(cohort["analysis_year"], errors="coerce")
        == TEMPORAL_TEST_YEAR
    ].reset_index(drop=True)

    X_test = test[model_features].copy()
    y_test = pd.to_numeric(test["target"], errors="raise").to_numpy(dtype=int)

    fitted_pipeline = joblib.load(
        output_dir
        / "models"
        / "final_stable_feature_uncalibrated_pipeline.joblib"
    )

    X_sample, grouped, group_names, metadata = compute_grouped_interactions(
        module,
        fitted_pipeline,
        X_test,
        y_test,
        categorical_features,
        numeric_features,
    )

    revised_dir = figure_output
    revised_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(
        revised_dir / "interaction_sample_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        revised_dir / "fatal_grouped_shap_interactions.npz",
        grouped_interactions=grouped,
        group_names=np.asarray(group_names, dtype=object),
        temporal_test_row_index=metadata["temporal_test_row_index"].to_numpy(),
        observed_class=metadata["observed_class"].to_numpy(),
    )

    validation = validate_against_reference_ranking(
        output_dir,
        grouped,
        group_names,
    )
    validation.to_csv(
        revised_dir / "interaction_reproduction_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    geometry_idx = group_names.index("Vaze_Hendese")
    cause_idx = group_names.index("Elat_Tame")
    geometry_cause_outputs(
        module,
        X_sample,
        grouped[:, geometry_idx, cause_idx],
        revised_dir,
        min_cell_count=args.min_cell_count,
        max_causes=args.max_causes,
    )

    month_sin_idx = group_names.index("month_sin")
    month_cos_idx = group_names.index("month_cos")
    lighting_idx = group_names.index("Roshanaei")
    combined_month_lighting = (
        grouped[:, month_sin_idx, lighting_idx]
        + grouped[:, month_cos_idx, lighting_idx]
    )
    seasonal_lighting_outputs(
        X_sample,
        combined_month_lighting,
        revised_dir,
        min_lighting_total=args.min_lighting_total,
    )

    write_captions(revised_dir, args.min_cell_count)

    summary = {
        "status": "complete",
        "output_directory": revised_dir.name,
        "sample_rows": int(len(X_sample)),
        "fatal_rows": int((metadata["observed_class"] == FATAL_CLASS).sum()),
        "sample_seed": DEFAULT_SAMPLE_SEED,
        "sample_size_limit": DEFAULT_SAMPLE_SIZE,
        "generated_files": sorted(
            path.name for path in revised_dir.iterdir() if path.is_file()
        ),
    }
    (revised_dir / "FIGURE_REBUILD_SUMMARY.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("SHAP interaction figure rebuild completed.")
    print(f"Output: {revised_dir}")
    print(f"Sample rows: {len(X_sample)}")
    print(
        "Fatal rows: "
        f"{int((metadata['observed_class'] == FATAL_CLASS).sum())}"
    )
    print("\nReproduction check:")
    print(validation.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())