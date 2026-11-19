#!/usr/bin/env python3
"""Additional SHAP sensitivity and stability analyses for the locked model."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.stats import spearmanr

CLASSES = ["PDO", "Injury", "Fatal"]


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def xgb_view(matrix) -> np.ndarray:
    """Dense view preserving XGBoost sparse-matrix missing-value semantics."""
    if not sparse.issparse(matrix):
        return np.asarray(matrix, dtype=float)
    coo = sparse.coo_matrix(matrix)
    dense = np.full(coo.shape, np.nan, dtype=float)
    dense[coo.row, coo.col] = coo.data
    return dense


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float), encoding="utf-8")


def grouped_main_effects(mcp, model, X, categorical, numeric, background=None):
    transformed, _, encoded_groups = mcp.transformed_matrix_and_metadata(
        model, X, categorical, numeric
    )
    estimator = mcp.unwrap_tree_estimator(model)
    dense = xgb_view(transformed)
    if background is None:
        explainer = shap.TreeExplainer(estimator)
    else:
        background_t, _, _ = mcp.transformed_matrix_and_metadata(
            model, background, categorical, numeric
        )
        explainer = shap.TreeExplainer(
            estimator,
            data=xgb_view(background_t),
            feature_perturbation="interventional",
            model_output="raw",
        )
    values = mcp.standardize_shap_values(
        explainer.shap_values(dense, check_additivity=False),
        dense.shape[0], dense.shape[1], 3,
    )
    grouped, groups = mcp.group_shap_values(values, encoded_groups)
    return grouped, list(groups)


def importance_tables(mcp, grouped, groups):
    rows = []
    for g, feature in enumerate(groups):
        for k, class_name in enumerate(CLASSES):
            rows.append({
                "feature": feature,
                "feature_label": mcp.FEATURE_LABELS.get(feature, feature),
                "class": class_name,
                "mean_absolute_shap": float(np.mean(np.abs(grouped[:, g, k]))),
            })
    class_table = pd.DataFrame(rows)
    global_table = (
        class_table.groupby(["feature", "feature_label"], as_index=False)
        .mean(numeric_only=True)
        .sort_values("mean_absolute_shap", ascending=False)
    )
    return class_table, global_table


def main() -> int:
    parser = argparse.ArgumentParser(description="SHAP sensitivity and stability analyses.")
    here = Path(__file__).resolve().parent
    parser.add_argument("--pipeline", type=Path, default=here / "motorcycle_crash_pipeline.py")
    parser.add_argument("--runner", type=Path, default=here / "supplementary_analyses.py")
    parser.add_argument("--primary-output", type=Path, default=Path("outputs") / "main_analysis")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "shap_analysis")
    parser.add_argument("--background-size", type=int, default=500)
    parser.add_argument("--refits", type=int, default=20)
    args = parser.parse_args()

    mcp = load_module("motorcycle_crash_pipeline", args.pipeline.resolve())
    supp = load_module("supplementary_analyses", args.runner.resolve())
    primary = args.primary_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    selection = json.loads(
        (primary / "models" / "final_model_selection.json").read_text(encoding="utf-8")
    )
    categorical = list(selection["primary_categorical_features"])
    numeric = list(selection["primary_numeric_features"])
    features = [*categorical, *numeric]
    model = joblib.load(
        primary / "models" / "final_stable_feature_uncalibrated_pipeline.joblib"
    )
    cohort = pd.read_csv(
        primary / "audit" / "motorcycle_cohort_analysis_ready.csv",
        encoding="utf-8-sig", keep_default_na=False, na_values=[""],
    )
    development = cohort[cohort.analysis_year.isin([96, 97])].reset_index(drop=True)
    test = cohort[cohort.analysis_year == 98].reset_index(drop=True)
    X_dev, y_dev = development[features], development.target.to_numpy(int)
    X_test, y_test = test[features], test.target.to_numpy(int)

    sample_idx = mcp.stratified_sample_indices(
        y_test, max_samples=min(1500, len(y_test)), seed=42 + 90000, include_all_class=2
    )
    X_sample = X_test.iloc[sample_idx]

    path_values, groups = grouped_main_effects(
        mcp, model, X_sample, categorical, numeric
    )
    path_class, path_global = importance_tables(mcp, path_values, groups)
    path_class.to_csv(output / "path_dependent_class_specific.csv", index=False)
    path_global.to_csv(output / "path_dependent_global.csv", index=False)

    interventional = {}
    for label, seed in (("A", 42 + 95000), ("B", 42 + 95001)):
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            len(development), size=min(args.background_size, len(development)), replace=False
        )
        values, g2 = grouped_main_effects(
            mcp, model, X_sample, categorical, numeric, background=X_dev.iloc[idx]
        )
        if g2 != groups:
            raise RuntimeError("Grouped feature order changed.")
        class_table, global_table = importance_tables(mcp, values, groups)
        class_table.to_csv(output / f"interventional_{label}_class_specific.csv", index=False)
        global_table.to_csv(output / f"interventional_{label}_global.csv", index=False)
        interventional[label] = (class_table, global_table)

    comparison = path_global[["feature", "feature_label", "mean_absolute_shap"]].rename(
        columns={"mean_absolute_shap": "path_dependent"}
    )
    for label in ("A", "B"):
        table = interventional[label][1][["feature", "mean_absolute_shap"]].rename(
            columns={"mean_absolute_shap": f"interventional_{label}"}
        )
        comparison = comparison.merge(table, on="feature")
    for column in ("path_dependent", "interventional_A", "interventional_B"):
        comparison[f"{column}_rank"] = comparison[column].rank(
            ascending=False, method="min"
        ).astype(int)
    comparison.to_csv(output / "main_effect_algorithm_comparison.csv", index=False)

    summary = {
        "global_spearman_path_vs_interventional": float(
            spearmanr(comparison.path_dependent, comparison.interventional_A)[0]
        ),
        "global_spearman_background_A_vs_B": float(
            spearmanr(comparison.interventional_A, comparison.interventional_B)[0]
        ),
    }

    config = mcp.config_for_mode("full", 42, 2)
    refit_rows = []
    for refit, bootstrap_idx in enumerate(
        supp.stratified_bootstrap_samples(y_dev, args.refits, 62002), start=1
    ):
        pipe = supp.fixed_xgb_pipeline(
            mcp, categorical, numeric, config, selection["best_params"],
            seed=supp.FINAL_MODEL_SEED + refit,
        )
        pipe.fit(X_dev.iloc[bootstrap_idx], y_dev[bootstrap_idx])
        values, refit_groups = grouped_main_effects(
            mcp, pipe, X_sample, categorical, numeric
        )
        if refit_groups != groups:
            raise RuntimeError("Grouped feature order changed across refits.")
        means = np.mean(np.abs(values), axis=0)
        for class_idx, class_name in enumerate([*CLASSES, "Global"]):
            score = means.mean(axis=1) if class_name == "Global" else means[:, class_idx]
            ranks = pd.Series(score).rank(ascending=False, method="min").to_numpy()
            for g, feature in enumerate(groups):
                refit_rows.append({
                    "refit": refit,
                    "class": class_name,
                    "feature": feature,
                    "feature_label": mcp.FEATURE_LABELS.get(feature, feature),
                    "mean_absolute_shap": float(score[g]),
                    "rank": int(ranks[g]),
                })
    refits = pd.DataFrame(refit_rows)
    stability = (
        refits.groupby(["class", "feature", "feature_label"], as_index=False)
        .agg(
            median_rank=("rank", "median"),
            top5_frequency=("rank", lambda x: float(np.mean(x <= 5))),
            top10_frequency=("rank", lambda x: float(np.mean(x <= 10))),
        )
        .sort_values(["class", "median_rank"])
    )
    stability.to_csv(output / "main_effect_refit_stability.csv", index=False)
    summary["refits"] = int(args.refits)
    write_json(output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
