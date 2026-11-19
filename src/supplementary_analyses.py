#!/usr/bin/env python3
"""
Supplementary robustness and sensitivity analyses for the motorcycle
crash-severity study.

This script imports the primary pipeline and reuses its cohort construction,
preprocessing, model classes, evaluation conventions, feature mapping, and SHAP
utilities. It is designed to run after the main analysis and verifies the saved
reference model before executing additional analyses.

Analyses included:
- multinomial logistic-regression benchmark with paired comparisons;
- fatal-class SHAP interaction stability under record bootstrap and model refits;
- rolling temporal validation (2017 development, 2018 test);
- bootstrap uncertainty for fatal precision, recall, and F1;
- fatal-class calibration diagnostics;
- sensitivity to investigator- and fault-assigned predictors;
- predictor-availability threshold sensitivity; and
- Missing/Unreported and SHAP diagnostics.

The raw data and row-level outputs are not distributed with this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it=None, *args, **kwargs):
        return it if it is not None else range(kwargs.get("total", 0))

RUN_VERSION = "supplementary_analysis_v1.1"

# Numerical settings used for the archived reference analysis
PROBABILITY_TOLERANCE = 1e-6      # XGBoost/saved CSV are float32; also requires zero class changes
FINAL_MODEL_SEED = 42 + 50000     # seed of the reference final model (fit_final_model_search)
BOOTSTRAP_SEED = 42 + 60000       # seed of the reference 2019 bootstrap (bootstrap_temporal_metrics)
LOGIT_MAX_ITER = 5000
LOGIT_C_MIN, LOGIT_C_MAX = 1e-4, 1e4
STRICT_REPRODUCTION = False     # set by --strict-reproduction

DEFAULT_PRIMARY_OUTPUT_NAME = "main_analysis"
DEFAULT_SUPPLEMENTARY_OUTPUT_NAME = "supplementary_analyses"

REFERENCE_COUNTS = {
    "total": 6802,
    "pdo": 3871,
    "injury": 2780,
    "fatal": 151,
    "dev": 3904,
    "dev_fatal": 93,
    "test": 2898,
    "test_fatal": 58,
}

# Reference metrics used only as a transparent reproduction audit.
# Probability-file equality is the stronger gate.
REFERENCE_METRIC_CHECKS = {
    "macro_f1": 0.478,
    "balanced_accuracy": 0.501,
    "fatal_f1": 0.154,
    "log_loss": 0.718,
    "multiclass_brier": 0.447,
}


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_logging(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output / "supplementary_run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def locate_pipeline(path_arg: Path | None) -> Path:
    if path_arg:
        p = path_arg.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Main pipeline not found: {p}")
        return p
    here = Path(__file__).resolve().parent
    preferred = here / "motorcycle_crash_pipeline.py"
    if preferred.exists():
        return preferred
    matches = sorted(here.glob("motorcycle_crash_pipeline*.py"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        "Place motorcycle_crash_pipeline.py in this folder or pass --pipeline. "
        f"Candidates found: {[m.name for m in matches]}"
    )


def import_main_pipeline(path: Path):
    # Critical for unpickling AutoBalancedClassifier/MulticlassLogitCalibrator.
    # The pipeline folder is also placed on sys.path (and PYTHONPATH) so that
    # joblib/loky worker processes used by parallel searches can import the
    # module by name; otherwise parallel XGBoost searches fail to unpickle.
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    existing = os.environ.get("PYTHONPATH", "")
    if folder not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = folder + (os.pathsep + existing if existing else "")
    spec = importlib.util.spec_from_file_location("motorcycle_crash_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import main pipeline from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["motorcycle_crash_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_main_analysis_files(primary_output: Path) -> dict[str, Path]:
    files = {
        "model": primary_output / "models" / "final_stable_feature_uncalibrated_pipeline.joblib",
        "calibrator": primary_output / "models" / "multiclass_logit_calibrator.joblib",
        "selection": primary_output / "models" / "final_model_selection.json",
        "predictions": primary_output / "evaluation" / "temporal_test_predictions.csv",
        "xgb_folds": primary_output / "modeling" / "nested_cv_XGBoost_folds.csv",
        "xgb_oof": primary_output / "modeling" / "nested_oof_XGBoost_predictions.csv",
        "stability": primary_output / "audit" / "feature_stability_selection.csv",
        "cohort": primary_output / "audit" / "motorcycle_cohort_analysis_ready.csv",
        "interaction_ranking": primary_output / "shap" / "fatal_shap_interaction_ranking.csv",
        "manifest": primary_output / "00_run_manifest.json",
    }
    missing = [f"{k}: {v}" for k, v in files.items() if k not in {"interaction_ranking", "manifest"} and not v.exists()]
    if missing:
        raise FileNotFoundError("Required main-analysis outputs are missing:\n" + "\n".join(missing))
    return files


def make_run_fingerprint(args, pipeline: Path, primary_files: Mapping[str, Path]) -> str:
    payload = {
        "version": RUN_VERSION,
        "pipeline_sha256": sha256_file(pipeline),
        "bootstrap_iterations": args.bootstrap_iterations,
        "shap_refit_repeats": args.shap_refit_repeats,
        "n_jobs": args.n_jobs,
        "reference_files": {
            k: sha256_file(v)
            for k, v in primary_files.items()
            if v.exists() and v.is_file() and k in {"selection", "predictions", "xgb_folds", "stability", "cohort"}
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def stage_marker(output: Path, stage: str) -> Path:
    return output / "_state" / f"{stage}.json"


def stage_done(output: Path, stage: str, run_id: str) -> bool:
    p = stage_marker(output, stage)
    if not p.exists():
        return False
    try:
        d = read_json(p)
        return d.get("run_version") == RUN_VERSION and d.get("run_id") == run_id and d.get("status") == "complete"
    except Exception:
        return False


def mark_stage(output: Path, stage: str, run_id: str, extra: Mapping[str, Any] | None = None) -> None:
    write_json(stage_marker(output, stage), {
        "run_version": RUN_VERSION,
        "run_id": run_id,
        "status": "complete",
        "completed_at": pd.Timestamp.now().isoformat(),
        **(dict(extra or {})),
    })


def require_counts(cohort: pd.DataFrame) -> dict[str, int]:
    counts = cohort["target"].value_counts().to_dict()
    result = {
        "total": len(cohort),
        "pdo": int(counts.get(0, 0)),
        "injury": int(counts.get(1, 0)),
        "fatal": int(counts.get(2, 0)),
        "dev": int(cohort["analysis_year"].isin([96, 97]).sum()),
        "dev_fatal": int(((cohort["analysis_year"].isin([96, 97])) & (cohort["target"] == 2)).sum()),
        "test": int((cohort["analysis_year"] == 98).sum()),
        "test_fatal": int(((cohort["analysis_year"] == 98) & (cohort["target"] == 2)).sum()),
    }
    mismatches = {k: (result[k], v) for k, v in REFERENCE_COUNTS.items() if result[k] != v}
    if mismatches:
        raise RuntimeError(
            "Cohort does not match the reference analysis. "
            f"Observed vs expected: {mismatches}. Regenerate the main analysis before running supplementary analyses."
        )
    return result


def load_reference_cohort(files: Mapping[str, Path], primary_cat, primary_num, model):
    """Load the saved analysis-ready cohort and verify it against saved predictions."""
    saved_pred = pd.read_csv(files["predictions"], encoding="utf-8-sig")
    target_probs = saved_pred[["base_p_pdo", "base_p_injury", "base_p_fatal"]].to_numpy(float)
    target_y = saved_pred["y_true"].to_numpy(int)
    features = [*primary_cat, *primary_num]

    # keep_default_na=False preserves the substantive category "None".
    cohort = pd.read_csv(
        files["cohort"],
        encoding="utf-8-sig",
        keep_default_na=False,
        na_values=[""],
    )
    counts = require_counts(cohort)
    test = cohort.loc[cohort["analysis_year"] == 98].reset_index(drop=True)
    if len(test) != len(saved_pred):
        raise RuntimeError(f"Temporal-test row count {len(test)} != saved predictions {len(saved_pred)}")
    y = test["target"].to_numpy(int)
    if not np.array_equal(y, target_y):
        raise RuntimeError("Temporal-test outcome order does not match the saved predictions.")
    check = probability_reproduction_check(model.predict_proba(test[features]), target_probs)
    if not check["passed"]:
        raise RuntimeError(f"Reference-model probability reproduction failed: {check}")
    logging.info("Saved analysis-ready cohort reproduces the reference predictions: %s", check)
    return cohort, "saved_analysis_ready_csv", counts, check

def build_logistic_pipeline(mcp, categorical_features: Sequence[str], numeric_features: Sequence[str], min_frequency: int, seed: int) -> Pipeline:
    from sklearn.preprocessing import StandardScaler
    pre = mcp.make_preprocessor(categorical_features, numeric_features, min_frequency)
    # Standardize the numeric block (after the original median imputation and missing
    # indicators). One-hot columns stay 0/1. Without this, unscaled posted speed limit
    # prevents convergence and makes the L2 penalty act unevenly across predictors.
    name, num_pipe, _cols = pre.transformers[0]
    if name != "num" or not hasattr(num_pipe, "steps"):
        raise RuntimeError("Unexpected original preprocessor structure; cannot add numeric scaling.")
    num_pipe.steps.append(("scale", StandardScaler()))
    return Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(solver="lbfgs", penalty="l2", class_weight="balanced",
                                     max_iter=LOGIT_MAX_ITER, random_state=seed)),
    ])


def run_logistic_benchmark(mcp, X_dev, y_dev, X_test, y_test, cats, nums, config, primary_output: Path, output: Path, n_jobs: int, saved_pred: pd.DataFrame, iterations: int):
    out = output / "01_logistic_benchmark"; out.mkdir(parents=True, exist_ok=True)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)  # same outer folds as the reference XGBoost
    folds = []
    for fold, (tr, va) in enumerate(tqdm(list(outer.split(X_dev, y_dev)), desc="logistic nested CV", unit="fold"), 1):
        seed = 42 + fold * 1009
        pipe = build_logistic_pipeline(mcp, cats, nums, config.onehot_min_frequency, seed)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(
            pipe, {"model__C": loguniform(LOGIT_C_MIN, LOGIT_C_MAX)}, n_iter=20,
            scoring=mcp.scoring_dictionary(), refit="macro_f1", cv=inner,
            random_state=seed, n_jobs=n_jobs, error_score="raise"
        )
        search.fit(X_dev.iloc[tr], y_dev[tr])
        p = search.best_estimator_.predict_proba(X_dev.iloc[va])
        metrics = mcp.evaluate_probabilities(y_dev[va], p)
        folds.append({"model": "MultinomialLogisticRegression", "outer_fold": fold,
                      "best_inner_macro_f1": float(search.best_score_),
                      "best_C": float(search.best_params_["model__C"]),
                      "solver_iterations": logistic_iterations(search.best_estimator_), **metrics})
    lf = pd.DataFrame(folds)
    lf.to_csv(out / "logistic_nested_cv_folds.csv", index=False, encoding="utf-8-sig")
    lsum = mcp.summarize_nested_cv(lf)
    lsum.to_csv(out / "logistic_nested_cv_summary.csv", index=False, encoding="utf-8-sig")

    xgbf = pd.read_csv(primary_output / "modeling" / "nested_cv_XGBoost_folds.csv", encoding="utf-8-sig")
    pairs = []
    for metric in ["macro_f1", "balanced_accuracy", "fatal_f1", "log_loss"]:
        merged = xgbf[["outer_fold", metric]].merge(lf[["outer_fold", metric]], on="outer_fold", suffixes=("_xgb", "_logit"))
        d = merged[f"{metric}_xgb"] - merged[f"{metric}_logit"]
        pairs.append({"metric": metric, "paired_folds": len(d), "mean_xgb_minus_logit": d.mean(),
                      "sd_difference": d.std(ddof=1), "xgb_better_folds": int((d > 0).sum()),
                      "ties": int((d == 0).sum()), "logit_better_folds": int((d < 0).sum()),
                      "min_difference": d.min(), "max_difference": d.max()})
    pd.DataFrame(pairs).to_csv(out / "paired_XGBoost_vs_logistic.csv", index=False, encoding="utf-8-sig")

    final_pipe = build_logistic_pipeline(mcp, cats, nums, config.onehot_min_frequency, 42 + 50000)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + 50000)
    final_search = RandomizedSearchCV(
        final_pipe, {"model__C": loguniform(LOGIT_C_MIN, LOGIT_C_MAX)}, n_iter=40,
        scoring=mcp.scoring_dictionary(), refit="macro_f1", cv=inner,
        random_state=42 + 50000, n_jobs=n_jobs, error_score="raise"
    )
    final_search.fit(X_dev, y_dev)
    best = final_search.best_estimator_
    ptest = best.predict_proba(X_test)
    tmetrics = mcp.evaluate_probabilities(y_test, ptest)
    joblib.dump(best, out / "multinomial_logistic_final_pipeline.joblib")
    pd.DataFrame({"y_true": y_test, "p_pdo": ptest[:, 0], "p_injury": ptest[:, 1], "p_fatal": ptest[:, 2],
                  "predicted_class": ptest.argmax(axis=1)}).to_csv(out / "logistic_temporal_predictions.csv", index=False)

    # Paired 2019 bootstrap: logistic vs the saved XGBoost predictions, same resamples as the paper.
    ref = saved_pred[["base_p_pdo", "base_p_injury", "base_p_fatal"]].to_numpy(float)
    boot, boot_summary = paired_bootstrap(y_test, ref, ptest, iterations, BOOTSTRAP_SEED)
    boot_summary.to_csv(out / "logistic_temporal_bootstrap_vs_XGBoost.csv", index=False, encoding="utf-8-sig")

    iters = [*lf["solver_iterations"].tolist(), logistic_iterations(best)]
    best_c = float(final_search.best_params_["model__C"])
    convergence = {"max_iter": LOGIT_MAX_ITER, "max_solver_iterations_used": int(max(iters)),
                   "all_converged": bool(max(iters) < LOGIT_MAX_ITER),
                   "C_search_range": [LOGIT_C_MIN, LOGIT_C_MAX],
                   "final_best_C": best_c,
                   "best_C_near_search_boundary": bool(best_c < 2 * LOGIT_C_MIN or best_c > LOGIT_C_MAX / 2)}
    write_json(out / "logistic_temporal_metrics.json", {"best_C": best_c, "convergence": convergence, **tmetrics})
    if not convergence["all_converged"]:
        raise RuntimeError(f"Logistic benchmark did not converge: {convergence}")
    return {"nested_summary": lsum.to_dict(orient="records"), "paired": pairs, "temporal": tmetrics,
            "temporal_bootstrap_vs_xgboost": boot_summary.to_dict(orient="records"),
            "best_C": best_c, "convergence": convergence}


def binary_calibration_intercept_slope(y_binary: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    logit = np.log(p/(1-p)).reshape(-1,1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000).fit(logit, y_binary)
    except Exception:
        model = LogisticRegression(C=1e12, penalty="l2", solver="lbfgs", max_iter=10000).fit(logit, y_binary)
    return float(model.intercept_[0]), float(model.coef_[0,0])


def equal_width_ece(y_binary: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0,1,n_bins+1); e=0.0
    for a,b in zip(edges[:-1],edges[1:]):
        mask = (p>=a)&(p<=b) if b==1 else (p>=a)&(p<b)
        if mask.any():
            e += mask.mean() * abs(y_binary[mask].mean()-p[mask].mean())
    return float(e)


def run_fatal_calibration(saved_predictions: pd.DataFrame, output: Path):
    out = output / "04_fatal_calibration"; out.mkdir(parents=True, exist_ok=True)
    y = (saved_predictions["y_true"].to_numpy(int)==2).astype(int)
    rows=[]
    curves={}
    for label,col in [("Uncalibrated","base_p_fatal"),("Development-fitted logistic calibration","calibrated_p_fatal")]:
        p=saved_predictions[col].to_numpy(float)
        intercept,slope=binary_calibration_intercept_slope(y,p)
        rows.append({"method":label,"mean_predicted_fatal_probability":p.mean(),"observed_fatal_prevalence":y.mean(),
                     "fatal_brier":brier_score_loss(y,p),"fatal_ece_equal_width_10bins":equal_width_ece(y,p,10),
                     "calibration_intercept":intercept,"calibration_slope":slope})
        frac,mean=calibration_curve(y,p,n_bins=10,strategy="uniform")
        curves[label]=(mean,frac)
    frame=pd.DataFrame(rows); frame.to_csv(out/"fatal_class_calibration.csv",index=False,encoding="utf-8-sig")
    plt.figure(figsize=(6.5,5.5))
    plt.plot([0,1],[0,1],"--",linewidth=1,label="Perfect calibration")
    for label,(mean,frac) in curves.items(): plt.plot(mean,frac,marker="o",label=label)
    plt.xlabel("Mean predicted fatal probability"); plt.ylabel("Observed fatal frequency")
    plt.title("Temporal-test fatal-class calibration"); plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(out/"fatal_class_calibration.png",dpi=300,bbox_inches="tight"); plt.close()
    return frame.to_dict(orient="records")


def fixed_xgb_pipeline(mcp, cats, nums, config, best_params, seed):
    p = mcp.build_primary_pipeline("XGBoost", cats, nums, config, seed)
    return mcp.set_model_parameters(p, best_params)


def evaluate_fixed_feature_set(mcp, cohort, cats, nums, config, best_params, label, seed):
    """Refit the reference XGBoost specification on a given feature set. The seed is the
    reference final-model seed, so the primary feature set reproduces the reference model."""
    features = [*cats, *nums]
    dev = cohort[cohort.analysis_year.isin([96, 97])].reset_index(drop=True)
    test = cohort[cohort.analysis_year == 98].reset_index(drop=True)
    Xd, yd = dev[features], dev.target.to_numpy(int)
    Xt, yt = test[features], test.target.to_numpy(int)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    pipe = fixed_xgb_pipeline(mcp, cats, nums, config, best_params, seed)
    cv = mcp.fixed_pipeline_cv(pipe, Xd, yd, list(splitter.split(Xd, yd)), label)
    fitted = clone(pipe).fit(Xd, yd)
    pt = fitted.predict_proba(Xt)
    tm = mcp.evaluate_probabilities(yt, pt)
    return cv, tm, fitted, pt


def run_investigator_sensitivity(mcp, cohort, primary_cat, primary_num, config, best_params, output: Path, n_jobs: int, saved_pred: pd.DataFrame, iterations: int):
    out = output / "05_investigator_predictor_sensitivity"; out.mkdir(parents=True, exist_ok=True)
    tiers = {
        "Primary reference feature set": [],
        "exclude primary cause + human factor": ["Elat_Tame", "Amel_Ensani_Moaser"],
        "Strict retrospective-field exclusion": ["Elat_Tame", "Amel_Ensani_Moaser", "Amel_Vasile_Naqlie", "Naqs_Moaser_Rah", "Noe_Vasile_Naqlie_1", "Jens_Rannade_Moqaser_1"],
    }
    saved = saved_pred[["base_p_pdo", "base_p_injury", "base_p_fatal"]].to_numpy(float)
    yt = saved_pred["y_true"].to_numpy(int)
    summaries = []
    ref = None  # primary refit = reference for paired comparisons (same seed, same machine)
    for i, (label, remove) in enumerate(tqdm(list(tiers.items()), desc="feature-set sensitivity", unit="set"), 1):
        cats = [f for f in primary_cat if f not in remove]
        nums = [f for f in primary_num if f not in remove]
        cv, tm, _, pt = evaluate_fixed_feature_set(mcp, cohort, cats, nums, config, best_params, label, FINAL_MODEL_SEED)
        cv.to_csv(out / f"cv_{i}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"y_true": yt, "p_pdo": pt[:, 0], "p_injury": pt[:, 1], "p_fatal": pt[:, 2]}).to_csv(out / f"temporal_predictions_{i}.csv", index=False)
        row = {"analysis": label, "removed": "; ".join(mcp.FEATURE_LABELS.get(f, f) for f in remove),
               "feature_count": len(cats) + len(nums),
               "development_macro_f1_mean": cv.macro_f1.mean(), "development_fatal_f1_mean": cv.fatal_f1.mean(),
               **{f"temporal_{k}": v for k, v in tm.items()}}
        if not remove:
            ref = pt
            check_primary_refit(pt, saved, row, "primary feature set")
        else:
            _, bs = paired_bootstrap(yt, ref, pt, iterations, BOOTSTRAP_SEED)
            bs.to_csv(out / f"bootstrap_vs_primary_{i}.csv", index=False, encoding="utf-8-sig")
            for _, r in bs.iterrows():
                row[f"{r.metric}_ci_low"] = r.ci_low
                row[f"{r.metric}_ci_high"] = r.ci_high
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out / "investigator_predictor_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    return summaries


def run_rolling_validation(mcp, cohort, primary_cat, primary_num, config, output: Path, n_jobs: int, iterations: int):
    out = output / "03_rolling_2017_to_2018"; out.mkdir(parents=True, exist_ok=True)
    train = cohort[cohort.analysis_year == 96].reset_index(drop=True)
    test = cohort[cohort.analysis_year == 97].reset_index(drop=True)
    feats = [*primary_cat, *primary_num]
    Xtr, ytr = train[feats], train.target.to_numpy(int)
    Xte, yte = test[feats], test.target.to_numpy(int)
    # Original final-search function and search space; 2017 records only.
    search = mcp.fit_final_model_search("XGBoost", Xtr, ytr, primary_cat, primary_num, config)
    p = search.best_estimator_.predict_proba(Xte)
    metrics = mcp.evaluate_probabilities(yte, p)
    pred = p.argmax(axis=1)
    boot = pd.DataFrame([hard_metrics(yte[idx], pred[idx]) for idx in stratified_bootstrap_samples(yte, iterations, BOOTSTRAP_SEED)])
    bsum = summarize_bootstrap_frame(boot, skip=())
    bsum.to_csv(out / "rolling_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    write_json(out / "rolling_2017_train_2018_test.json", {
        "feature_schema": "reference 20-predictor schema held fixed; no new availability audit",
        "hyperparameter_selection": "2017 only, original XGBoost search space and final-search procedure",
        "train_rows": len(train), "train_fatal": int((ytr == 2).sum()),
        "test_rows": len(test), "test_fatal": int((yte == 2).sum()),
        "best_params": search.best_params_, "inner_cv_macro_f1": search.best_score_,
        "bootstrap_ci": bsum.to_dict(orient="records"), **metrics})
    pd.DataFrame({"y_true": yte, "p_pdo": p[:, 0], "p_injury": p[:, 1], "p_fatal": p[:, 2],
                  "predicted_class": pred}).to_csv(out / "rolling_predictions.csv", index=False)
    return {**metrics, "bootstrap_ci": bsum.to_dict(orient="records"),
            "train_rows": len(train), "test_rows": len(test)}


def select_by_rule(stability: pd.DataFrame, annual: float, gap: float) -> list[str]:
    return stability.loc[(stability.maximum_annual_missing_percent<=annual)&(stability.absolute_dev_test_missing_gap_pp<=gap),"feature"].tolist()


def ordered_feature_lists(mcp, retained: set[str], cohort_columns: set[str], primary_cat: Sequence[str], primary_num: Sequence[str]):
    """Reference predictors keep the reference order (so the primary rule reproduces the
    reference model); additional predictors follow in the main pipeline's order."""
    cat = [f for f in primary_cat if f in retained]
    cat += [f for f in mcp.RAW_CATEGORICAL_FEATURES if f in retained and f in cohort_columns and f not in cat]
    num = [f for f in primary_num if f in retained]
    num += [f for f in [*mcp.RAW_NUMERIC_FEATURES, *mcp.DERIVED_NUMERIC_FEATURES] if f in retained and f in cohort_columns and f not in num]
    return cat, num


def run_threshold_sensitivity(mcp, cohort, stability: pd.DataFrame, config, best_params, output: Path, n_jobs: int, primary_cat, primary_num, saved_pred: pd.DataFrame, iterations: int):
    out = output / "06_availability_threshold_sensitivity"; out.mkdir(parents=True, exist_ok=True)
    rules = [("Strict", 15, 10), ("Primary", 20, 15), ("Moderate", 30, 20), ("Permissive", 50, 30), ("All candidates", 100, 100)]
    primary_set = set(primary_cat) | set(primary_num)
    saved = saved_pred[["base_p_pdo", "base_p_injury", "base_p_fatal"]].to_numpy(float)
    yt = saved_pred["y_true"].to_numpy(int)
    rows, seen = [], {}
    cols = set(cohort.columns)

    def fit_row(cats, nums, label):
        cv, tm, _, pt = evaluate_fixed_feature_set(mcp, cohort, cats, nums, config, best_params, label, FINAL_MODEL_SEED)
        row = {"features": "; ".join([*cats, *nums]),
               "development_macro_f1_mean": cv.macro_f1.mean(), "development_fatal_f1_mean": cv.fatal_f1.mean(),
               "temporal_macro_f1": tm["macro_f1"], "temporal_balanced_accuracy": tm["balanced_accuracy"],
               "temporal_fatal_f1": tm["fatal_f1"], "temporal_log_loss": tm["log_loss"],
               "temporal_macro_roc_auc_ovr": tm["macro_roc_auc_ovr"], "note": ""}
        return row, pt

    primary_row, ref = fit_row(list(primary_cat), list(primary_num), "Primary")
    check_primary_refit(ref, saved, primary_row, "primary availability rule")
    for label, annual, gap in tqdm(rules, desc="availability thresholds", unit="rule"):
        retained = set(select_by_rule(stability, annual, gap))
        cats, nums = ordered_feature_lists(mcp, retained, cols, primary_cat, primary_num)
        fs = frozenset([*cats, *nums])
        if not fs:
            continue
        base = {"rule": label, "max_annual_missing_percent": annual, "max_dev_test_gap_pp": gap,
                "feature_count": len(fs),
                "added_vs_primary": "; ".join(mcp.FEATURE_LABELS.get(f, f) for f in [*cats, *nums] if f not in primary_set),
                "removed_vs_primary": "; ".join(mcp.FEATURE_LABELS.get(f, f) for f in [*primary_cat, *primary_num] if f not in fs)}
        if fs in seen:
            prior = dict(seen[fs]); prior.update(base)
            prior["note"] = f"identical feature set to rule '{seen[fs]['rule']}'; results copied, not refit"
            rows.append(prior)
            continue
        if fs == primary_set:
            row = {**base, **primary_row}
        else:
            r_, pt = fit_row(cats, nums, label)
            row = {**base, **r_}
            _, bs = paired_bootstrap(yt, ref, pt, iterations, BOOTSTRAP_SEED)
            for _, r in bs.iterrows():
                if r.metric.startswith("difference_"):
                    row[f"{r.metric}_ci_low"] = r.ci_low
                    row[f"{r.metric}_ci_high"] = r.ci_high
        seen[fs] = row
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "availability_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    return rows






def xgb_view(matrix) -> np.ndarray:
    """Return a dense SHAP input matching XGBoost sparse-matrix semantics."""
    if not sparse.issparse(matrix):
        return np.asarray(matrix, dtype=float)
    coo = sparse.coo_matrix(matrix)
    dense = np.full(coo.shape, np.nan, dtype=float)
    dense[coo.row, coo.col] = coo.data
    return dense


def grouped_interaction_tensor(mcp, model, X_sample, cats, nums):
    import shap
    transformed,_,encoded_groups=mcp.transformed_matrix_and_metadata(model,X_sample,cats,nums)
    dense=xgb_view(transformed)
    explainer=shap.TreeExplainer(mcp.unwrap_tree_estimator(model))
    raw=explainer.shap_interaction_values(dense)
    std=mcp.standardize_interaction_values(raw,dense.shape[0],dense.shape[1],3)[:,:,:,2]
    G,groups=mcp.grouping_matrix(encoded_groups)
    grouped=np.einsum("fi,sfj,jg->sig",G,std,G)
    return grouped,groups


def rank_interactions(grouped: np.ndarray, groups: Sequence[str], mcp) -> pd.DataFrame:
    mean_abs=np.mean(np.abs(grouped),axis=0); rows=[]
    for i in range(len(groups)):
        for j in range(i+1,len(groups)):
            rows.append({"feature_1":groups[i],"feature_2":groups[j],
                         "feature_1_label":mcp.FEATURE_LABELS.get(groups[i],groups[i]),
                         "feature_2_label":mcp.FEATURE_LABELS.get(groups[j],groups[j]),
                         "mean_absolute_interaction":float(mean_abs[i,j])})
    return pd.DataFrame(rows).sort_values("mean_absolute_interaction",ascending=False).reset_index(drop=True)


def summarize_rank_runs(rank_runs: list[pd.DataFrame]) -> pd.DataFrame:
    records=[]
    for run_id,rank in enumerate(rank_runs,1):
        for pos,row in enumerate(rank.itertuples(index=False),1):
            key=" || ".join(sorted([row.feature_1_label,row.feature_2_label]))
            records.append({"run":run_id,"pair":key,"rank":pos})
    df=pd.DataFrame(records); total=len(rank_runs)
    s=df.groupby("pair").agg(median_rank=("rank","median"),rank_q1=("rank",lambda x:np.quantile(x,.25)),
                              rank_q3=("rank",lambda x:np.quantile(x,.75)),mean_rank=("rank","mean"),
                              top5_frequency=("rank",lambda x:np.mean(x<=5)),top10_frequency=("rank",lambda x:np.mean(x<=10)),
                              appearances=("rank","count")).reset_index()
    # Pairs absent from a top list are still present because rank_interactions ranks all pairs.
    return s.sort_values(["median_rank","mean_rank"]).reset_index(drop=True)


def run_shap_stability(mcp, model, cohort, primary_cat, primary_num, config, best_params, primary_output: Path, output: Path, boot_iters: int, refit_repeats: int):
    out = output / "02_shap_interaction_stability"; out.mkdir(parents=True, exist_ok=True)
    test = cohort[cohort.analysis_year == 98].reset_index(drop=True); ytest = test.target.to_numpy(int); Xtest = test[[*primary_cat, *primary_num]]
    idx = mcp.stratified_sample_indices(ytest, max_samples=min(600, len(test)), seed=42 + 91000, include_all_class=2)
    Xs = Xtest.iloc[idx].copy(); ys = ytest[idx]
    if int((ys == 2).sum()) != 58:
        raise RuntimeError(f"Interaction sample should contain all 58 fatal cases; found {(ys == 2).sum()}")
    grouped, groups = grouped_interaction_tensor(mcp, model, Xs, primary_cat, primary_num)
    locked_rank = rank_interactions(grouped, groups, mcp)
    locked_rank.to_csv(out / "locked_model_interaction_ranking.csv", index=False, encoding="utf-8-sig")
    # (a) Class-stratified bootstrap of the exact 600-record interpretation sample
    # (every resample keeps 58 fatal records), model held fixed.
    ranks = []
    for bi in tqdm(stratified_bootstrap_samples(ys, boot_iters, 62001), total=boot_iters, desc="SHAP sample bootstrap", unit="boot"):
        ranks.append(rank_interactions(grouped[bi], groups, mcp))
    sample_summary = summarize_rank_runs(ranks); sample_summary.to_csv(out / "sample_bootstrap_rank_stability.csv", index=False, encoding="utf-8-sig")

    # (b) Model-fit stability: class-stratified bootstrap of development rows, refit the
    # reference XGBoost specification, recompute the ranking on the SAME 600 records.
    refit_summary = []
    if refit_repeats > 0:
        dev = cohort[cohort.analysis_year.isin([96, 97])].reset_index(drop=True); yd = dev.target.to_numpy(int); Xd = dev[[*primary_cat, *primary_num]]
        rranks = []
        for r, bi in enumerate(tqdm(stratified_bootstrap_samples(yd, refit_repeats, 62002), total=refit_repeats, desc="SHAP model-refit stability", unit="refit")):
            pipe = fixed_xgb_pipeline(mcp, primary_cat, primary_num, config, best_params, seed=FINAL_MODEL_SEED + 1 + r)
            pipe.fit(Xd.iloc[bi], yd[bi])
            g2, groups2 = grouped_interaction_tensor(mcp, pipe, Xs, primary_cat, primary_num)
            if list(groups2) != list(groups):
                raise RuntimeError("Grouped feature order changed across SHAP refits")
            rranks.append(rank_interactions(g2, groups2, mcp))
        refit_df = summarize_rank_runs(rranks); refit_df.to_csv(out / "model_refit_rank_stability.csv", index=False, encoding="utf-8-sig")
        refit_summary = refit_df.to_dict(orient="records")
    return {"sample_bootstrap": sample_summary.to_dict(orient="records"), "model_refit": refit_summary,
            "locked_top10": locked_rank.head(10).to_dict(orient="records")}


def write_analysis_summary(output: Path, results: Mapping[str, Any]) -> None:
    """Write a concise, public-facing summary of supplementary analysis results."""
    out = output / "summary"
    out.mkdir(parents=True, exist_ok=True)

    def f3(x: Any) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{float(x):.3f}"

    def ci(records: Any, metric: str) -> str:
        record = next((x for x in (records or []) if x.get("metric") == metric), None)
        return f"{f3(record['ci_low'])} to {f3(record['ci_high'])}" if record else "NA"

    lines = [
        "# Supplementary analysis summary",
        "",
        "All values in this file are generated from the current run.",
        "",
    ]

    lg = results.get("logistic_benchmark", {})
    t = lg.get("temporal", {})
    lines += [
        "## Multinomial logistic-regression benchmark",
        f"- Temporal macro F1: {f3(t.get('macro_f1'))}",
        f"- Temporal balanced accuracy: {f3(t.get('balanced_accuracy'))}",
        f"- Temporal fatal-class F1: {f3(t.get('fatal_f1'))}",
        f"- Temporal macro AUC: {f3(t.get('macro_roc_auc_ovr'))}",
        f"- Temporal log loss: {f3(t.get('log_loss'))}",
        "",
    ]

    sh = results.get("shap_interaction_stability", {})
    lines += ["## SHAP interaction stability"]
    for item in sh.get("model_refit", [])[:10]:
        lines.append(
            f"- {item['pair']}: median rank {f3(item.get('median_rank'))}, "
            f"top-10 frequency {f3(item.get('top10_frequency'))}"
        )
    lines.append("")

    roll = results.get("rolling_validation", {})
    lines += [
        "## Rolling temporal validation",
        f"- Macro F1: {f3(roll.get('macro_f1'))} ({ci(roll.get('bootstrap_ci'), 'macro_f1')})",
        f"- Balanced accuracy: {f3(roll.get('balanced_accuracy'))}",
        f"- Fatal-class F1: {f3(roll.get('fatal_f1'))} ({ci(roll.get('bootstrap_ci'), 'fatal_f1')})",
        f"- Macro AUC: {f3(roll.get('macro_roc_auc_ovr'))}",
        "",
    ]

    unc = results.get("fatal_class_uncertainty", [])
    lines += [
        "## Fatal-class uncertainty",
        f"- Fatal precision 95% CI: {ci(unc, 'fatal_precision')}",
        f"- Fatal recall 95% CI: {ci(unc, 'fatal_recall')}",
        f"- Fatal F1 95% CI: {ci(unc, 'fatal_f1')}",
        "",
    ]

    lines += ["## Fatal-class calibration"]
    for item in results.get("fatal_calibration", []):
        lines.append(
            f"- {item['method']}: mean predicted fatal probability {f3(item.get('mean_predicted_fatal_probability'))}; "
            f"Brier {f3(item.get('fatal_brier'))}; ECE {f3(item.get('fatal_ece_equal_width_10bins'))}; "
            f"slope {f3(item.get('calibration_slope'))}"
        )
    lines.append("")

    lines += ["## Investigator-field sensitivity"]
    for item in results.get("investigator_field_sensitivity", []):
        lines.append(
            f"- {item.get('analysis')}: macro F1 {f3(item.get('macro_f1'))}, fatal F1 {f3(item.get('fatal_f1'))}"
        )
    lines.append("")

    lines += ["## Predictor-availability threshold sensitivity"]
    for item in results.get("availability_threshold_sensitivity", []):
        lines.append(
            f"- {item.get('rule')}: {item.get('feature_count')} predictors; "
            f"macro F1 {f3(item.get('temporal_macro_f1'))}; fatal F1 {f3(item.get('temporal_fatal_f1'))}"
        )
    lines.append("")

    miss = results.get("missingness_diagnostics", {})
    lines += [
        "## Missingness diagnostics",
        f"- Missingness-only macro F1: {f3(miss.get('missingness_only', {}).get('macro_f1'))}",
        f"- Missingness-only macro AUC: {f3(miss.get('missingness_only', {}).get('macro_roc_auc_ovr'))}",
        "",
    ]

    (out / "ANALYSIS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

def parse_args():
    global STRICT_REPRODUCTION
    p = argparse.ArgumentParser(description="Supplementary robustness analyses for the motorcycle crash-severity study.")
    p.add_argument("--pipeline", type=Path, default=None)
    p.add_argument("--primary-output", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--bootstrap-iterations", type=int, default=2000)
    p.add_argument("--shap-refit-repeats", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--strict-reproduction", action="store_true",
                   help="Stop if reference-model refits do not reproduce the saved predictions.")
    p.add_argument("--force", action="store_true", help="Delete supplementary output and rerun every stage.")
    args = p.parse_args()
    STRICT_REPRODUCTION = bool(args.strict_reproduction)
    return args


def probability_reproduction_check(p: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """XGBoost returns float32 probabilities and the saved CSV stores float32 values,
    so equality is tested at float32 precision (or <=1e-6) with zero class changes."""
    p = np.asarray(p, dtype=float)
    target = np.asarray(target, dtype=float)
    maxdiff = float(np.max(np.abs(p - target)))
    float32_exact = bool(np.array_equal(p.astype(np.float32), target.astype(np.float32)))
    class_changes = int((p.argmax(axis=1) != target.argmax(axis=1)).sum())
    passed = (float32_exact or maxdiff <= PROBABILITY_TOLERANCE) and class_changes == 0
    return {
        "max_abs_difference": maxdiff,
        "float32_exact": float32_exact,
        "predicted_class_changes": class_changes,
        "tolerance": PROBABILITY_TOLERANCE,
        "passed": bool(passed),
    }

def stratified_bootstrap_samples(y: np.ndarray, iterations: int, seed: int):
    """Identical resampling scheme to the original bootstrap_temporal_metrics():
    class-stratified resampling with replacement, then a shuffle."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    class_indices = {cls: np.where(y == cls)[0] for cls in [0, 1, 2]}
    for _ in range(iterations):
        sampled = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in class_indices.values()])
        rng.shuffle(sampled)
        yield sampled

def hard_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "fatal_precision": float(precision[2]),
        "fatal_recall": float(recall[2]),
        "fatal_f1": float(f1[2]),
    }

def summarize_bootstrap_frame(frame: pd.DataFrame, skip: Iterable[str] = ("iteration",)) -> pd.DataFrame:
    rows = []
    for c in frame.columns:
        if c in set(skip):
            continue
        v = frame[c].to_numpy(dtype=float)
        rows.append({"metric": c, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                     "ci_low": float(np.quantile(v, 0.025)), "ci_high": float(np.quantile(v, 0.975))})
    return pd.DataFrame(rows)

def paired_bootstrap(y: np.ndarray, ref_probs: np.ndarray, new_probs: np.ndarray, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap CIs for a comparison model and for its paired difference from the reference
    (both evaluated on the same resampled 2019 records)."""
    y = np.asarray(y, dtype=int)
    ref_pred = np.asarray(ref_probs).argmax(axis=1)
    new_pred = np.asarray(new_probs).argmax(axis=1)
    rows = []
    for b, idx in enumerate(stratified_bootstrap_samples(y, iterations, seed), 1):
        r = hard_metrics(y[idx], ref_pred[idx])
        n = hard_metrics(y[idx], new_pred[idx])
        row = {"iteration": b}
        for k in ["macro_f1", "balanced_accuracy", "fatal_f1"]:
            row[k] = n[k]
            row[f"difference_{k}_vs_reference"] = n[k] - r[k]
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, summarize_bootstrap_frame(frame)

def run_primary_bootstrap(saved_pred: pd.DataFrame, primary_output: Path, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """: fatal precision/recall/F1 CIs from the SAME 2,000 resamples (same seed and
    scheme) as the reference bootstrap, so the fatal-F1 interval matches the paper."""
    y = saved_pred["y_true"].to_numpy(int)
    pred = saved_pred[["base_p_pdo", "base_p_injury", "base_p_fatal"]].to_numpy(float).argmax(axis=1)
    rows = []
    for b, idx in enumerate(tqdm(stratified_bootstrap_samples(y, iterations, seed), total=iterations,
                                 desc="fatal metric bootstrap", unit="boot"), 1):
        rows.append({"iteration": b, **hard_metrics(y[idx], pred[idx])})
    frame = pd.DataFrame(rows)
    summary = summarize_bootstrap_frame(frame)
    check: dict[str, Any] = {"seed": seed, "iterations": iterations}
    saved_iter = primary_output / "evaluation" / "temporal_test_bootstrap_iterations.csv"
    if saved_iter.exists():
        s = pd.read_csv(saved_iter, encoding="utf-8-sig")
        if len(s) == len(frame):
            d_f1 = float(np.max(np.abs(s["classification_fatal_f1"].to_numpy() - frame["fatal_f1"].to_numpy())))
            d_mf1 = float(np.max(np.abs(s["classification_macro_f1"].to_numpy() - frame["macro_f1"].to_numpy())))
            check.update({"max_diff_fatal_f1_vs_reference_iterations": d_f1,
                          "max_diff_macro_f1_vs_reference_iterations": d_mf1,
                          "matches_reference_bootstrap": bool(d_f1 < 1e-12 and d_mf1 < 1e-12)})
        else:
            check.update({"matches_reference_bootstrap": None,
                          "note": f"iteration count {len(frame)} differs from reference {len(s)}; not comparable"})
    if check.get("matches_reference_bootstrap") is False:
        logging.warning("Fatal-metric bootstrap does NOT reproduce the reference resamples: %s", check)
    return frame, summary, check

def logistic_iterations(estimator) -> int:
    return int(np.max(estimator.named_steps["model"].n_iter_))

def check_primary_refit(pt: np.ndarray, saved: np.ndarray, row: dict, where: str) -> None:
    """The refit of the reference specification should reproduce the saved predictions on the
    machine that produced them. XGBoost can grow different trees on another OS/CPU, so a
    mismatch is reported loudly (and the refit is used as the comparison baseline); with
    --strict-reproduction it stops the run instead."""
    check = probability_reproduction_check(pt, saved)
    row["reproduces_reference_model"] = check["passed"]
    row["refit_vs_saved_predicted_class_changes"] = check["predicted_class_changes"]
    if not check["passed"]:
        msg = (f"{where}: refit of the reference XGBoost specification does NOT reproduce the saved 2019 "
               f"predictions ({check}). Comparisons in this stage use the refit as the baseline; its metrics "
               f"will differ slightly from the reference values. Expected only if this machine/OS differs "
               f"from the one that produced the reference outputs.")
        if STRICT_REPRODUCTION:
            raise RuntimeError(msg)
        logging.warning(msg)

def run_missingness_and_shap_diagnostics(mcp, model, cohort, primary_cat, primary_num, config, output: Path):
    """b plus a SHAP-offset diagnostic.
    (a) missingness-only model; (b) neutralization ablation of Missing/Unreported entries;
    (c) share of encoded SHAP attributable to missing-level columns; (d) grouped SHAP
    offset diagnostic (mean vs between-record spread)."""
    import shap
    out = output / "07_missingness_and_shap_diagnostics"; out.mkdir(parents=True, exist_ok=True)
    feats = [*primary_cat, *primary_num]
    dev = cohort[cohort.analysis_year.isin([96, 97])].reset_index(drop=True)
    test = cohort[cohort.analysis_year == 98].reset_index(drop=True)
    yd, yt = dev.target.to_numpy(int), test.target.to_numpy(int)
    label = lambda f: mcp.FEATURE_LABELS.get(f, f)
    result: dict[str, Any] = {}

    # Missingness profile. Only NaN is missing (the main pipeline decodes code 0/blank
    # to NaN and imputes "Missing/Unreported"); the substantive category "None" is not missing.
    prof = []
    for f in feats:
        md, mt = dev[f].isna().to_numpy(), test[f].isna().to_numpy()
        prof.append({"feature": f, "feature_label": label(f),
                     "dev_missing_percent": 100 * md.mean(), "test_missing_percent": 100 * mt.mean(),
                     "dev_fatal_percent_if_missing": 100 * np.mean(yd[md] == 2) if md.any() else np.nan,
                     "dev_fatal_percent_if_observed": 100 * np.mean(yd[~md] == 2) if (~md).any() else np.nan,
                     "test_fatal_percent_if_missing": 100 * np.mean(yt[mt] == 2) if mt.any() else np.nan,
                     "test_fatal_percent_if_observed": 100 * np.mean(yt[~mt] == 2) if (~mt).any() else np.nan})
    pd.DataFrame(prof).to_csv(out / "missingness_profile.csv", index=False, encoding="utf-8-sig")

    # (a) Missingness-only model: indicators only, class-weighted logistic regression.
    def indicators(df):
        return pd.DataFrame({f"{f}__missing": df[f].isna().astype(int) for f in feats})
    Zd, Zt = indicators(dev), indicators(test)
    keep = [c for c in Zd.columns if Zd[c].nunique() > 1]
    miss_model = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Zd[keep], yd)
    pm = miss_model.predict_proba(Zt[keep])
    miss_metrics = mcp.evaluate_probabilities(yt, pm)
    majority = np.tile(np.bincount(yd, minlength=3) == np.bincount(yd, minlength=3).max(), (len(yt), 1)).astype(float)
    base_metrics = mcp.evaluate_probabilities(yt, majority / majority.sum(axis=1, keepdims=True))
    result["missingness_only_model"] = {k: miss_metrics[k] for k in ["macro_f1", "balanced_accuracy", "fatal_f1", "macro_roc_auc_ovr"]}
    result["majority_class_baseline"] = {k: base_metrics[k] for k in ["macro_f1", "balanced_accuracy", "fatal_f1"]}
    write_json(out / "missingness_only_model.json", {"indicators_used": keep, **result})

    # (b) Neutralization ablation: replace Missing/Unreported with the development mode
    # (categorical) or median (numeric) and re-predict with the REFERENCE model.
    Xt = test[feats].copy()
    p0 = model.predict_proba(Xt)
    m0 = mcp.evaluate_probabilities(yt, p0)
    fill = {}
    for f in primary_cat:
        mode = dev[f].dropna().mode()
        fill[f] = mode.iloc[0] if len(mode) else None
    for f in primary_num:
        fill[f] = float(dev[f].median())
    abl = []
    Xall = Xt.copy()
    for f in feats:
        miss = Xt[f].isna().to_numpy()
        if not miss.any() or fill[f] is None:
            continue
        Xa = Xt.copy(); Xa.loc[miss, f] = fill[f]; Xall.loc[miss, f] = fill[f]
        pa = model.predict_proba(Xa); ma = mcp.evaluate_probabilities(yt, pa)
        abl.append({"feature": f, "feature_label": label(f), "test_missing_rows": int(miss.sum()), "filled_with": fill[f],
                    "mean_abs_change_p_fatal_on_missing_rows": float(np.mean(np.abs(pa[miss, 2] - p0[miss, 2]))),
                    "predicted_class_changes": int((pa.argmax(1) != p0.argmax(1)).sum()),
                    "delta_macro_f1": ma["macro_f1"] - m0["macro_f1"], "delta_fatal_f1": ma["fatal_f1"] - m0["fatal_f1"]})
    pall = model.predict_proba(Xall); mall = mcp.evaluate_probabilities(yt, pall)
    any_miss = Xt.isna().any(axis=1).to_numpy()
    abl.append({"feature": "ALL", "feature_label": "All Missing/Unreported entries neutralized together",
                "test_missing_rows": int(any_miss.sum()), "filled_with": "dev mode/median",
                "mean_abs_change_p_fatal_on_missing_rows": float(np.mean(np.abs(pall[any_miss, 2] - p0[any_miss, 2]))),
                "predicted_class_changes": int((pall.argmax(1) != p0.argmax(1)).sum()),
                "delta_macro_f1": mall["macro_f1"] - m0["macro_f1"], "delta_fatal_f1": mall["fatal_f1"] - m0["fatal_f1"]})
    pd.DataFrame(abl).to_csv(out / "missing_neutralization_ablation.csv", index=False, encoding="utf-8-sig")
    result["neutralization_all"] = abl[-1]

    # (c) + (d) SHAP on the same 1,500-record main-effect sample used in the reference analysis.
    idx = mcp.stratified_sample_indices(yt, max_samples=min(1500, len(test)), seed=42 + 90000, include_all_class=2)
    Xs = Xt.iloc[idx]
    transformed, encoded_names, encoded_groups = mcp.transformed_matrix_and_metadata(model, Xs, primary_cat, primary_num)
    dense = xgb_view(transformed)
    raw = shap.TreeExplainer(mcp.unwrap_tree_estimator(model)).shap_values(dense, check_additivity=False)
    std = mcp.standardize_shap_values(raw, dense.shape[0], dense.shape[1], 3)  # [n, encoded, class]
    abs_mean = np.abs(std).mean(axis=0)  # [encoded, class]
    totals = abs_mean.sum(axis=0)
    enc_rows = []
    for j, name in enumerate(encoded_names):
        low = str(name).lower()
        if "missing/unreported" in low or "missingindicator" in low:
            enc_rows.append({"encoded_column": name, "feature": encoded_groups[j], "feature_label": label(encoded_groups[j]),
                             **{f"share_of_total_abs_shap_{c}": float(abs_mean[j, k] / totals[k]) for k, c in enumerate(["pdo", "injury", "fatal"])}})
    enc = pd.DataFrame(enc_rows)
    enc.to_csv(out / "missing_level_encoded_shap_share.csv", index=False, encoding="utf-8-sig")
    result["missing_level_shap_share_total"] = {c: float(enc[f"share_of_total_abs_shap_{c}"].sum()) if len(enc) else 0.0
                                                for c in ["pdo", "injury", "fatal"]}
    without_level = [label(f) for f in primary_cat if not any(r["feature"] == f and "missing/unreported" in str(r["encoded_column"]).lower() for r in enc_rows)]
    result["categorical_features_without_own_missing_column"] = without_level

    grouped, gnames = mcp.group_shap_values(std, encoded_groups)  # [n, group, class]
    diag = []
    for k, c in enumerate(["PDO", "Injury", "Fatal"]):
        for g, gname in enumerate(gnames):
            s = grouped[:, g, k]
            diag.append({"class": c, "feature": gname, "feature_label": label(gname),
                         "mean_abs_shap": float(np.mean(np.abs(s))), "mean_signed_shap": float(np.mean(s)),
                         "sd_across_records": float(np.std(s)),
                         "centered_mean_abs_shap": float(np.mean(np.abs(s - s.mean()))),
                         "share_records_same_sign": float(max(np.mean(s > 0), np.mean(s < 0)))})
    dg = pd.DataFrame(diag)
    dg["rank_by_mean_abs"] = dg.groupby("class")["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    dg["rank_by_centered"] = dg.groupby("class")["centered_mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    dg.sort_values(["class", "rank_by_mean_abs"]).to_csv(out / "grouped_shap_offset_diagnostic.csv", index=False, encoding="utf-8-sig")    fat = dg[dg["class"] == "Fatal"]
    result["fatal_shap_offset"] = {
        "median_ratio_mean_abs_to_sd": float(np.median(fat.mean_abs_shap / fat.sd_across_records.replace(0, np.nan))),
        "share_features_same_sign_all_records": float(np.mean(fat.share_records_same_sign >= 0.999)),
        "top5_by_mean_abs": fat.nsmallest(5, "rank_by_mean_abs").feature_label.tolist(),
        "top5_by_centered": fat.nsmallest(5, "rank_by_centered").feature_label.tolist(),
    }
    write_json(out / "summary.json", result)
    return result


def resolve_primary_output(arg_primary: Path | None) -> Path:
    """Resolve the main-analysis output directory."""
    path = arg_primary.expanduser() if arg_primary is not None else Path("outputs") / DEFAULT_PRIMARY_OUTPUT_NAME
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Main-analysis output not found: {path}. Run motorcycle_crash_pipeline.py first or pass --primary-output."
    )

def package_versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "shap", "imbalanced-learn", "joblib"]:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    pipeline = locate_pipeline(args.pipeline); mcp = import_main_pipeline(pipeline)
    primary_output = resolve_primary_output(args.primary_output)
    args.primary_output = primary_output
    output = (args.output or (Path("outputs") / DEFAULT_SUPPLEMENTARY_OUTPUT_NAME)).expanduser()
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True); setup_logging(output)
    files = load_main_analysis_files(primary_output); run_id = make_run_fingerprint(args, pipeline, files)
    write_json(output / "RUN_MANIFEST.json", {
        "run_version": RUN_VERSION,
        "run_id": run_id,
        "pipeline_file": pipeline.name,
        "pipeline_sha256": sha256_file(pipeline),
        "primary_output_name": primary_output.name,
        "settings": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "shap_refit_repeats": args.shap_refit_repeats,
            "n_jobs": args.n_jobs,
            "strict_reproduction": args.strict_reproduction,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
    })
    config = mcp.config_for_mode("full", 42, args.n_jobs)
    results: dict[str, Any] = {}

    # 00 Preflight: load the reference artifacts and prove the cohort + model reproduce
    # the SAVED 2019 probabilities before any new analysis runs.
    selection = read_json(files["selection"])
    primary_cat = list(selection["primary_categorical_features"])
    primary_num = list(selection["primary_numeric_features"])
    best_params = selection["best_params"]
    model = joblib.load(files["model"])
    joblib.load(files["calibrator"])  # confirms the custom calibrator class deserializes
    cohort, cohort_source, counts, check = load_reference_cohort(files, primary_cat, primary_num, model)
    saved_pred = pd.read_csv(files["predictions"], encoding="utf-8-sig")
    stage = "00_preflight"
    if not stage_done(output, stage, run_id):
        logging.info("Stage 00: reference-model preflight")
        y = saved_pred.y_true.to_numpy(int); p = saved_pred[["base_p_pdo", "base_p_injury", "base_p_fatal"]].to_numpy(float)
        met = mcp.evaluate_probabilities(y, p)
        write_json(output / "00_preflight" / "preflight.json", {
            "cohort_source": cohort_source, "counts": counts, "probability_reproduction": check,
            "reference_metric_audit": {k: met[k] for k in ["macro_f1", "balanced_accuracy", "fatal_f1", "log_loss", "multiclass_brier"]},
            "primary_categorical_features": primary_cat, "primary_numeric_features": primary_num, "best_params": best_params})
        mark_stage(output, stage, run_id)
    else:
        logging.info("Stage 00 already complete for this run fingerprint.")

    dev = cohort[cohort.analysis_year.isin([96, 97])].reset_index(drop=True)
    test = cohort[cohort.analysis_year == 98].reset_index(drop=True)
    feats = [*primary_cat, *primary_num]
    Xdev, ydev = dev[feats], dev.target.to_numpy(int)
    Xtest, ytest = test[feats], test.target.to_numpy(int)
    B = args.bootstrap_iterations

    stage = "01_logistic_benchmark"
    if not stage_done(output, stage, run_id):
        results["logistic_benchmark"] = run_logistic_benchmark(mcp, Xdev, ydev, Xtest, ytest, primary_cat, primary_num, config, primary_output, output, args.n_jobs, saved_pred, B)
        write_json(output / "01_logistic_benchmark" / "summary.json", results["logistic_benchmark"]); mark_stage(output, stage, run_id)
    else:
        results["logistic_benchmark"] = read_json(output / "01_logistic_benchmark" / "summary.json")

    stage = "02_shap_stability"
    if not stage_done(output, stage, run_id):
        results["shap_interaction_stability"] = run_shap_stability(mcp, model, cohort, primary_cat, primary_num, config, best_params, primary_output, output, B, args.shap_refit_repeats)
        write_json(output / "02_shap_interaction_stability" / "summary.json", results["shap_interaction_stability"]); mark_stage(output, stage, run_id)
    else:
        results["shap_interaction_stability"] = read_json(output / "02_shap_interaction_stability" / "summary.json")

    stage = "03_rolling_validation"
    if not stage_done(output, stage, run_id):
        results["rolling_validation"] = run_rolling_validation(mcp, cohort, primary_cat, primary_num, config, output, args.n_jobs, B); mark_stage(output, stage, run_id)
    else:
        results["rolling_validation"] = read_json(output / "03_rolling_2017_to_2018" / "rolling_2017_train_2018_test.json")

    stage = "04_fatal_CIs_and_calibration"
    d = output / "04_fatal_calibration"
    if not stage_done(output, stage, run_id):
        d.mkdir(parents=True, exist_ok=True)
        boots, summary, bcheck = run_primary_bootstrap(saved_pred, primary_output, B, BOOTSTRAP_SEED)
        boots.to_csv(d / "primary_classification_bootstrap_iterations.csv", index=False)
        summary.to_csv(d / "primary_classification_bootstrap_summary.csv", index=False)
        results["fatal_class_uncertainty"] = summary.to_dict(orient="records"); results["bootstrap_reproduction_check"] = bcheck
        results["fatal_calibration"] = run_fatal_calibration(saved_pred, output)
        write_json(d / "combined_summary.json", {"fatal_CIs": results["fatal_class_uncertainty"], "bootstrap_check": bcheck, "fatal_calibration": results["fatal_calibration"]})
        mark_stage(output, stage, run_id)
    else:
        x = read_json(d / "combined_summary.json")
        results["fatal_class_uncertainty"] = x["fatal_CIs"]; results["bootstrap_reproduction_check"] = x["bootstrap_check"]; results["fatal_calibration"] = x["fatal_calibration"]

    stage = "05_investigator_sensitivity"
    if not stage_done(output, stage, run_id):
        results["investigator_field_sensitivity"] = run_investigator_sensitivity(mcp, cohort, primary_cat, primary_num, config, best_params, output, args.n_jobs, saved_pred, B)
        write_json(output / "05_investigator_predictor_sensitivity" / "summary.json", results["investigator_field_sensitivity"]); mark_stage(output, stage, run_id)
    else:
        results["investigator_field_sensitivity"] = read_json(output / "05_investigator_predictor_sensitivity" / "summary.json")

    stability = pd.read_csv(files["stability"], encoding="utf-8-sig")
    stage = "06_threshold_sensitivity"
    if not stage_done(output, stage, run_id):
        results["availability_threshold_sensitivity"] = run_threshold_sensitivity(mcp, cohort, stability, config, best_params, output, args.n_jobs, primary_cat, primary_num, saved_pred, B)
        write_json(output / "06_availability_threshold_sensitivity" / "summary.json", results["availability_threshold_sensitivity"]); mark_stage(output, stage, run_id)
    else:
        results["availability_threshold_sensitivity"] = read_json(output / "06_availability_threshold_sensitivity" / "summary.json")

    stage = "07_missingness_and_shap_diagnostics"
    if not stage_done(output, stage, run_id):
        results["missingness_diagnostics"] = run_missingness_and_shap_diagnostics(mcp, model, cohort, primary_cat, primary_num, config, output); mark_stage(output, stage, run_id)
    else:
        results["missingness_diagnostics"] = read_json(output / "07_missingness_and_shap_diagnostics" / "summary.json")

    write_analysis_summary(output, results)
    write_json(output / "FINAL_SUPPLEMENTARY_ANALYSIS_SUMMARY.json", results)
    (output / "RUN_COMPLETE.txt").write_text(f"{RUN_VERSION}\nAll supplementary analysis stages completed successfully.\nRun fingerprint: {run_id}\n", encoding="utf-8")
    logging.info("All supplementary analyses completed: %s", output)
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Completed stages are resumable by rerunning the same launcher.",file=sys.stderr); raise
    except Exception as exc:
        traceback.print_exc(); sys.exit(1)