#!/usr/bin/env python3
"""
Reproducible analysis pipeline for the study:
"Motorcycle Crash Severity Prediction under Temporal Validation with
Leakage-Controlled Machine Learning and SHAP Interaction Analysis."

The workflow implements:
- cohort construction from the coded crash workbook;
- cross-year predictor-availability auditing;
- leakage-controlled repeated nested cross-validation;
- Random Forest, XGBoost, and LightGBM comparison;
- development-only probability calibration and fatal-crash screening;
- temporal evaluation on the held-out final year;
- SMOTENC, spatial-group, and full-feature sensitivity analyses; and
- grouped SHAP main-effect and interaction analyses.

The raw crash data are not distributed with this repository. See README.md for
expected input structure and data-use restrictions.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

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
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy import sparse
from scipy.stats import randint, loguniform, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Ensure joblib artifacts created when this file is executed as a script remain
# loadable later by importing ``motorcycle_crash_pipeline``.
if __name__ == "__main__":
    sys.modules.setdefault("motorcycle_crash_pipeline", sys.modules[__name__])


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DATASET = Path("data") / "UTM 96-98.xlsx"
DATA_SHEET = "TUTM 96-98"
GUIDE_SHEET = "راهنما"

TARGET_COLUMN = "Noe_Tasadof"
MOTORCYCLE_FILTER_COLUMN = "Noe_Vasile_Naqlie_2"
MOTORCYCLE_FILTER_VALUE = "1"
YEAR_COLUMN = "Sal_Barasi"
DEVELOPMENT_YEARS = (96, 97)
TEMPORAL_TEST_YEAR = 98

TARGET_CODE_TO_NAME = {1: "PDO", 2: "Injury", 3: "Fatal"}
TARGET_CODE_TO_INTERNAL = {1: 0, 2: 1, 3: 2}
INTERNAL_TO_NAME = {0: "PDO", 1: "Injury", 2: "Fatal"}
CLASS_ORDER = [0, 1, 2]
CLASS_NAMES = [INTERNAL_TO_NAME[i] for i in CLASS_ORDER]
FATAL_CLASS = 2

# Reference cohort checks for the dataset used in the study.
EXPECTED_COHORT_SIZE = 6802
EXPECTED_SEVERITY_COUNTS = {"PDO": 3871, "Injury": 2780, "Fatal": 151}

PROTOCOL_VERSION = "paper_release_v1.0"
MAX_ANNUAL_MISSING_PERCENT = 20.0
MAX_DEV_TEST_MISSING_GAP_PERCENT = 15.0
PRACTICAL_MODEL_TIE_MACRO_F1 = 0.01

# The primary predictor set deliberately excludes exact locations, record IDs,
# outcome-derived fields, casualty counts, and post-crash response information.
RAW_CATEGORICAL_FEATURES = [
    "Noe_Rah",
    "Naqs_Moaser_Rah",
    "Samt_Rah",
    "Roshanaei",
    "Roye_Rah",
    "Noe_Shane",
    "Mane_Did",
    "Tamirat",
    "Karbari_Mahal",
    "Moqeiate_Tasadof",
    "Vaze_Hava",
    "Vaze_Hendese",
    "Khatkeshi",
    "Amel_Vasile_Naqlie",
    "Amel_Ensani_Moaser",
    "Elat_Tame",
    "Noe_Vasile_Naqlie_1",
    "Manovr_Vasile_1",
    "Moayene_Fani_1",
    "Jens_Rannade_Moqaser_1",
    "بومی_غیربومی",
    "رخ_به_رخ",
    "Mahdode_Abadi",
    "Mahdode_Madares",
]

RAW_NUMERIC_FEATURES = [
    "Sorat_Hadeaksar",
    "Arz_Rah",
    "Sen_Rannade_Moqaser_1",
]

DERIVED_NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

TIME_SOURCE_COLUMNS = ["Saat", "Mah"]
SPATIAL_COLUMNS = ["X", "Y"]

IDENTIFIER_COLUMNS = {
    "OBJECTID",
    "Cod_Uniq_Faqare",
    "Cod_Uniq",
    "Serial",
    "Serial_Pelak_1",
    "Serial_Pelak_2",
    "Shomare_Pelak_1",
    "Shomare_Pelak_2",
    "Koroki",
}

EXACT_LOCATION_OR_HIGH_CARDINALITY_COLUMNS = {
    "Name_Hoze",
    "Cod_Hoze",
    "Name_Mehvar",
    "Band_Raft_Bargasht",
    "Raft_Bargasht",
    "Name_Markaz",
    "Cod_Rah",
    "Jahat",
    "Name_Rah",
    "Cod_Qate",
    "Name_Qate",
    "Cod_Noqte",
    "Name_Noqte",
    "Fasele",
    "Fasele_Az_Mabda",
    "Onvan_File",
    "X",
    "Y",
}

OUTCOME_LEAKAGE_COLUMNS = {
    "Noe_Tasadof1",
    "Tedad_Footi",
    "Tedad_jarhi",
    "Tedad_Footi_Rannade",
    "Tedad_Footi_Sarneshin",
    "Tedad_Footi_Aber",
    "Tedad_Masdom_Rannade",
    "Tedad_Masdom_Sarneshin",
    "Tedad_Masdom_Aber",
    "Footi",
    "Jarhi",
    "Khesarati",
}

POST_CRASH_OR_RESPONSE_COLUMNS = {
    "Khesarat_Varede",
    "Zaman_Residegi",
    "Satt_Residegi",
    "Daqiqe_Residegi",
}

SUBSET_OR_SPLIT_COLUMNS = {
    TARGET_COLUMN,
    MOTORCYCLE_FILTER_COLUMN,
    YEAR_COLUMN,
    "Sal",
    "Tarikh",
    "Tarikh_Sabt",
    "F6_mah_sal",
    "موتورسیکلت",
    "عابر",
}

FEATURE_LABELS = {
    "Noe_Rah": "Road type",
    "Naqs_Moaser_Rah": "Effective road defect",
    "Samt_Rah": "Road direction configuration",
    "Roshanaei": "Lighting condition",
    "Roye_Rah": "Road-surface condition",
    "Noe_Shane": "Shoulder type",
    "Mane_Did": "Visibility obstruction",
    "Tamirat": "Roadwork status",
    "Karbari_Mahal": "Land-use context",
    "Moqeiate_Tasadof": "Crash location",
    "Vaze_Hava": "Weather condition",
    "Vaze_Hendese": "Road geometry",
    "Khatkeshi": "Road marking",
    "Sorat_Hadeaksar": "Posted speed limit",
    "Arz_Rah": "Road width",
    "Amel_Vasile_Naqlie": "Vehicle-related factor",
    "Amel_Ensani_Moaser": "Human factor",
    "Elat_Tame": "Primary crash cause",
    "Noe_Vasile_Naqlie_1": "At-fault vehicle type",
    "Manovr_Vasile_1": "At-fault vehicle maneuver",
    "Moayene_Fani_1": "Technical inspection status",
    "Sen_Rannade_Moqaser_1": "At-fault driver age",
    "Jens_Rannade_Moqaser_1": "At-fault driver sex",
    "بومی_غیربومی": "Local/nonlocal status",
    "رخ_به_رخ": "Head-on indicator",
    "Mahdode_Abadi": "Settlement-area indicator",
    "Mahdode_Madares": "School-zone indicator",
    "hour_sin": "Crash hour (sine)",
    "hour_cos": "Crash hour (cosine)",
    "month_sin": "Crash month (sine)",
    "month_cos": "Crash month (cosine)",
}

# English decoding for coded variables used in the primary analysis.
ENGLISH_CODE_MAPS: dict[str, dict[str, str]] = {
    "Naqs_Moaser_Rah": {
        "1": "None",
        "2": "Vertical-sign defect",
        "3": "Horizontal-sign defect",
        "4": "Insufficient road width",
        "5": "Bump/obstacle",
        "6": "No shoulder/parking area",
        "7": "Pavement-shoulder level difference",
        "8": "Missing roadside barrier",
        "9": "Nonstandard roadside barrier",
        "10": "Road settlement",
        "11": "Pavement-surface defect",
        "12": "Sharp curve",
        "13": "Nonstandard longitudinal/cross slope",
        "14": "Road-lighting defect",
        "15": "Road-marking defect",
        "16": "Other",
        "17": "Other/derived code 17",
    },
    "Samt_Rah": {
        "1": "One-way",
        "2": "Two-way undivided",
        "3": "Two-way physically divided",
    },
    "Roshanaei": {
        "1": "Daylight",
        "2": "Dawn",
        "3": "Dusk",
        "4": "Night with adequate lighting",
        "5": "Night without adequate lighting",
    },
    "Roye_Rah": {
        "1": "Dry/normal",
        "2": "Wet",
        "3": "Snow/ice",
        "4": "Gravel/earth",
        "5": "Muddy",
        "6": "Oily/dirty",
        "7": "Flooded",
        "8": "Tar-treated",
        "9": "Other",
    },
    "Noe_Shane": {
        "1": "No shoulder",
        "2": "Unpaved shoulder",
        "3": "Paved shoulder",
    },
    "Mane_Did": {
        "1": "None",
        "2": "Tree/shrub",
        "3": "Building/kiosk",
        "4": "Earth mound/obstacle",
        "5": "Stopped vehicle",
        "6": "Moving vehicle",
        "7": "Sun glare",
        "8": "Slope",
        "9": "Vertical curve",
        "10": "Fog/smoke",
        "11": "Blizzard",
        "12": "Sandstorm",
        "13": "Sign",
        "14": "Oncoming headlight",
        "15": "Frozen windshield",
        "16": "Other",
    },
    "Tamirat": {
        "1": "Not under repair",
        "2": "Under repair without adequate warning",
        "3": "Under repair with adequate warning",
    },
    "Karbari_Mahal": {
        "1": "Residential",
        "2": "Administrative/commercial",
        "3": "Industrial",
        "4": "Agricultural",
        "5": "Educational",
        "6": "Recreational",
        "7": "Nonresidential",
        "8": "Other",
    },
    "Moqeiate_Tasadof": {
        "1": "Carriageway",
        "2": "Shoulder",
        "3": "Median",
        "4": "Roadside within right-of-way",
        "5": "Outside road right-of-way",
        "6": "Other",
    },
    "Vaze_Hava": {
        "1": "Clear",
        "2": "Foggy",
        "3": "Snowy",
        "4": "Rainy",
        "5": "Stormy",
        "6": "Cloudy",
        "7": "Dusty",
    },
    "Vaze_Hendese": {
        "1": "Straight-level",
        "2": "Straight-uphill",
        "3": "Straight-downhill",
        "4": "Curve-level",
        "5": "Curve-uphill",
        "6": "Curve-downhill",
        "7": "Tunnel",
        "8": "Bridge",
    },
    "Khatkeshi": {
        "1": "No marking",
        "2": "Broken line",
        "3": "Solid line",
        "4": "Double line",
    },
    "Amel_Vasile_Naqlie": {
        "1": "None",
        "2": "Lighting-system failure",
        "3": "Brake-system failure",
        "4": "Steering-system failure",
        "5": "Worn tire",
        "6": "Missing tire chains when required",
        "7": "Missing wiper when required",
        "8": "Suspension-system failure",
        "9": "Other code 9",
        "10": "Other vehicle defect",
    },
    "Amel_Ensani_Moaser": {
        "1": "None",
        "2": "Fatigue/drowsiness",
        "3": "Physical impairment",
        "4": "Age-related weakness",
        "5": "Drug use",
        "6": "Alcohol use",
        "7": "Improperly secured cargo",
        "8": "Unnecessary haste",
        "9": "Failure to recognize others' right-of-way",
        "10": "Unfamiliarity with road",
        "11": "Other human factor",
    },
    "Elat_Tame": {
        "1": "Unsafe following distance",
        "2": "Unsafe lateral distance",
        "3": "Failure to yield",
        "4": "Inattention ahead",
        "5": "Loss of vehicle control",
        "6": "Excess speed",
        "7": "Improper passing/left encroachment",
        "8": "Left deviation",
        "9": "Right deviation",
        "10": "Improper turn",
        "11": "Crossing prohibited area",
        "12": "Wrong-way movement",
        "13": "Reversing",
        "14": "Road/vehicle technical defect",
        "15": "Vehicle technical defect",
        "16": "Sudden lane/path change",
        "17": "Pedestrian error",
        "18": "Road-safety-law violation",
        "19": "Cargo-regulation violation",
        "20": "Improper towing",
        "21": "Red-light violation",
        "22": "Illegal U-turn",
        "23": "Regulatory violation code 211/212",
        "24": "Sudden door opening",
        "25": "Other cause",
        "26": "Road defect",
    },
    "Noe_Vasile_Naqlie_1": {
        "1": "Passenger car",
        "2": "Minibus",
        "3": "Bus",
        "4": "Pickup",
        "5": "Truck",
        "6": "Light truck",
        "7": "Ambulance",
        "8": "Fire vehicle",
        "9": "Police vehicle",
        "10": "Motorcycle",
        "11": "Trailer",
        "12": "Bicycle",
        "13": "Agricultural machinery",
        "14": "Roadwork machinery",
        "15": "Hazardous-material tanker",
        "16": "Other vehicle",
    },
    "Manovr_Vasile_1": {
        "1": "Moving forward",
        "2": "Turning left",
        "3": "Turning right",
        "4": "U-turn",
        "5": "Overtaking",
        "6": "Reversing",
        "7": "Sudden start",
        "8": "Sudden stop",
        "9": "Stopped off roadway",
        "10": "Stopped on roadway",
        "11": "Weaving",
        "12": "Other maneuver",
    },
    "Moayene_Fani_1": {
        "1": "Valid inspection",
        "2": "No valid inspection",
        "3": "Inspection not required",
    },
    "Jens_Rannade_Moqaser_1": {
        "1": "Male",
        "2": "Female",
    },
    "بومی_غیربومی": {
        "1": "Local",
        "2": "Nonlocal",
    },
    "رخ_به_رخ": {
        "1": "Head-on",
    },
    "Mahdode_Abadi": {
        "1": "Within settlement area",
    },
    "Mahdode_Madares": {
        "1": "Within school zone",
        "0": "Outside school zone",
    },
}

UNKNOWN_TOKENS = {
    "",
    " ",
    "-",
    "--",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "پرنشده",
    "پر نشده",
    "نامشخص",
    "unknown",
}


@dataclass(frozen=True)
class RunConfig:
    mode: str
    seed: int
    outer_splits: int
    outer_repeats: int
    inner_splits: int
    search_iterations: int
    final_search_iterations: int
    bootstrap_iterations: int
    shap_max_samples: int
    interaction_max_samples: int
    onehot_min_frequency: int
    spatial_grid_m: int
    run_smotenc_sensitivity: bool
    run_spatial_sensitivity: bool
    run_full_feature_sensitivity: bool
    run_shap: bool
    run_interactions: bool
    n_jobs: int


def config_for_mode(mode: str, seed: int, n_jobs: int) -> RunConfig:
    if mode == "smoke":
        return RunConfig(
            mode=mode,
            seed=seed,
            outer_splits=2,
            outer_repeats=1,
            inner_splits=2,
            search_iterations=1,
            final_search_iterations=1,
            bootstrap_iterations=50,
            shap_max_samples=120,
            interaction_max_samples=50,
            onehot_min_frequency=10,
            spatial_grid_m=5000,
            run_smotenc_sensitivity=False,
            run_spatial_sensitivity=False,
            run_full_feature_sensitivity=False,
            run_shap=True,
            run_interactions=False,
            n_jobs=n_jobs,
        )
    return RunConfig(
        mode=mode,
        seed=seed,
        outer_splits=5,
        outer_repeats=2,
        inner_splits=3,
        search_iterations=20,
        final_search_iterations=40,
        bootstrap_iterations=2000,
        shap_max_samples=1500,
        interaction_max_samples=600,
        onehot_min_frequency=10,
        spatial_grid_m=5000,
        run_smotenc_sensitivity=True,
        run_spatial_sensitivity=True,
        run_full_feature_sensitivity=True,
        run_shap=True,
        run_interactions=True,
        n_jobs=n_jobs,
    )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_code(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text.casefold() in UNKNOWN_TOKENS:
        return None
    # Excel often converts integer codes to strings such as "3.0".
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        text = text.split(".")[0]
    return text


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = ["__".join(str(part) for part in col if str(part) != "") for col in frame.columns]
    return frame


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def metric_display_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def stratified_sample_indices(
    y: np.ndarray,
    max_samples: int,
    seed: int,
    include_all_class: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    all_indices = np.arange(len(y))
    selected: list[int] = []
    if include_all_class is not None:
        selected.extend(all_indices[y == include_all_class].tolist())
    remaining_budget = max(0, max_samples - len(selected))
    other = all_indices if include_all_class is None else all_indices[y != include_all_class]
    if remaining_budget >= len(other):
        selected.extend(other.tolist())
    elif remaining_budget > 0:
        classes, counts = np.unique(y[other], return_counts=True)
        allocations: dict[int, int] = {}
        for cls, count in zip(classes, counts):
            allocations[int(cls)] = max(1, int(round(remaining_budget * count / counts.sum())))
        while sum(allocations.values()) > remaining_budget:
            largest = max(allocations, key=allocations.get)
            allocations[largest] -= 1
        while sum(allocations.values()) < remaining_budget:
            largest_remaining = max(
                allocations,
                key=lambda cls: np.sum(y[other] == cls) - allocations[cls],
            )
            allocations[largest_remaining] += 1
        for cls, amount in allocations.items():
            cls_idx = other[y[other] == cls]
            amount = min(amount, len(cls_idx))
            selected.extend(rng.choice(cls_idx, size=amount, replace=False).tolist())
    return np.array(sorted(set(selected)), dtype=int)


# -----------------------------------------------------------------------------
# Dataset loading, cohort reconstruction, and governance
# -----------------------------------------------------------------------------


def load_workbook(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Reading workbook: %s", dataset_path)
    data = pd.read_excel(dataset_path, sheet_name=DATA_SHEET, engine="openpyxl")
    guide = pd.read_excel(dataset_path, sheet_name=GUIDE_SHEET, header=None, engine="openpyxl")
    data.columns = [str(col).strip() for col in data.columns]
    logging.info("Loaded %d rows and %d columns.", len(data), len(data.columns))
    return data, guide


def workbook_codebook_to_long(guide: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    pair_count = guide.shape[1] // 2
    for pair_idx in range(pair_count):
        code_col = pair_idx * 2
        name_col = code_col + 1
        category_name = guide.iloc[0, code_col] if code_col < guide.shape[1] else None
        for row_idx in range(2, len(guide)):
            code = normalize_code(guide.iloc[row_idx, code_col])
            label = guide.iloc[row_idx, name_col] if name_col < guide.shape[1] else None
            if code is None and (label is None or pd.isna(label)):
                continue
            records.append(
                {
                    "guide_pair": pair_idx + 1,
                    "guide_category": category_name,
                    "code": code,
                    "persian_label": None if pd.isna(label) else str(label).strip(),
                }
            )
    return pd.DataFrame(records)


def build_feature_governance(columns: Sequence[str]) -> pd.DataFrame:
    predictor_cat = set(RAW_CATEGORICAL_FEATURES)
    predictor_num = set(RAW_NUMERIC_FEATURES)
    records: list[dict[str, str]] = []
    for column in columns:
        if column == TARGET_COLUMN:
            role = "target"
            rationale = "Three-class severity outcome; never used as a predictor."
        elif column == MOTORCYCLE_FILTER_COLUMN:
            role = "cohort_filter"
            rationale = "Code 1 reconstructs the 6,802-record motorcycle cohort."
        elif column == YEAR_COLUMN:
            role = "temporal_split"
            rationale = "1396-1397 development; 1398 untouched temporal test."
        elif column in predictor_cat:
            role = "primary_categorical_predictor"
            rationale = "Prespecified crash, roadway, environment, vehicle, or human factor."
        elif column in predictor_num:
            role = "primary_numeric_predictor"
            rationale = "Prespecified continuous predictor with plausibility checks."
        elif column in TIME_SOURCE_COLUMNS:
            role = "derived_time_source"
            rationale = "Used only to create deterministic cyclical time features."
        elif column in IDENTIFIER_COLUMNS:
            role = "excluded_identifier"
            rationale = "Record/person/plate identifier; memorization risk and no policy meaning."
        elif column in EXACT_LOCATION_OR_HIGH_CARDINALITY_COLUMNS:
            role = "excluded_exact_location_or_high_cardinality"
            rationale = "Excluded from primary prediction to reduce spatial/route memorization."
        elif column in OUTCOME_LEAKAGE_COLUMNS:
            role = "excluded_outcome_leakage"
            rationale = "Directly encodes severity or casualty counts."
        elif column in POST_CRASH_OR_RESPONSE_COLUMNS:
            role = "excluded_post_crash"
            rationale = "Occurs after the crash or can proxy realized consequences."
        elif column in SUBSET_OR_SPLIT_COLUMNS:
            role = "excluded_design_variable"
            rationale = "Used for cohort definition, splitting, or auditing only."
        elif column in SPATIAL_COLUMNS:
            role = "spatial_audit_only"
            rationale = "Used to form 5-km spatial groups; not included in primary model."
        else:
            role = "excluded_not_prespecified"
            rationale = "Not included in the locked primary predictor set."
        records.append({"column": column, "role": role, "rationale": rationale})
    return pd.DataFrame(records)


def decode_categorical_feature(series: pd.Series, feature: str) -> pd.Series:
    mapping = ENGLISH_CODE_MAPS.get(feature, {})

    def decode_one(value: Any) -> Any:
        code = normalize_code(value)
        if code is None or code == "0":
            return np.nan
        if mapping:
            return mapping.get(code, f"Unmapped code {code}")
        return code

    return series.map(decode_one).astype("object")


def prepare_motorcycle_cohort(
    raw: pd.DataFrame,
    output_dir: Path,
    config: RunConfig,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    required = {
        TARGET_COLUMN,
        MOTORCYCLE_FILTER_COLUMN,
        YEAR_COLUMN,
        *RAW_CATEGORICAL_FEATURES,
        *RAW_NUMERIC_FEATURES,
        *TIME_SOURCE_COLUMNS,
        *SPATIAL_COLUMNS,
    }
    missing_required = sorted(required - set(raw.columns))
    if missing_required:
        raise KeyError(f"Required columns are missing: {missing_required}")

    duplicate_full_rows = int(raw.duplicated().sum())
    if duplicate_full_rows:
        logging.warning("Dropping %d exact duplicate full rows.", duplicate_full_rows)
        raw = raw.drop_duplicates().copy()
    else:
        raw = raw.copy()

    filter_codes = raw[MOTORCYCLE_FILTER_COLUMN].map(normalize_code)
    cohort = raw.loc[filter_codes == MOTORCYCLE_FILTER_VALUE].copy()
    if cohort.empty:
        raise RuntimeError(
            f"No motorcycle records found using {MOTORCYCLE_FILTER_COLUMN} == {MOTORCYCLE_FILTER_VALUE}."
        )

    target_codes = safe_numeric(cohort[TARGET_COLUMN]).astype("Int64")
    invalid_targets = ~target_codes.isin(TARGET_CODE_TO_INTERNAL)
    invalid_target_count = int(invalid_targets.sum())
    if invalid_target_count:
        logging.warning("Removing %d records with invalid/missing target codes.", invalid_target_count)
        cohort = cohort.loc[~invalid_targets].copy()
        target_codes = target_codes.loc[~invalid_targets]

    cohort["target_code"] = target_codes.astype(int)
    cohort["target"] = cohort["target_code"].map(TARGET_CODE_TO_INTERNAL).astype(int)
    cohort["severity_label"] = cohort["target_code"].map(TARGET_CODE_TO_NAME)
    cohort["analysis_year"] = safe_numeric(cohort[YEAR_COLUMN]).astype("Int64")

    # Numeric plausibility rules; invalid/unknown values become missing and are
    # imputed inside training folds.
    cohort["Sorat_Hadeaksar"] = safe_numeric(cohort["Sorat_Hadeaksar"])
    cohort.loc[~cohort["Sorat_Hadeaksar"].between(10, 130), "Sorat_Hadeaksar"] = np.nan

    cohort["Arz_Rah"] = safe_numeric(cohort["Arz_Rah"])
    cohort.loc[~cohort["Arz_Rah"].between(1, 60), "Arz_Rah"] = np.nan

    cohort["Sen_Rannade_Moqaser_1"] = safe_numeric(cohort["Sen_Rannade_Moqaser_1"])
    cohort.loc[
        ~cohort["Sen_Rannade_Moqaser_1"].between(10, 100),
        "Sen_Rannade_Moqaser_1",
    ] = np.nan

    # Deterministic cyclical time features. Values recorded as 24 are treated as
    # midnight (0). No learned statistic is used, so this transformation cannot
    # leak information across folds.
    hour = safe_numeric(cohort["Saat"])
    hour = hour.where(hour.between(0, 24))
    hour = hour.mod(24)
    month = safe_numeric(cohort["Mah"])
    month = month.where(month.between(1, 12))
    cohort["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    cohort["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    cohort["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    cohort["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    for feature in RAW_CATEGORICAL_FEATURES:
        cohort[feature] = decode_categorical_feature(cohort[feature], feature)

    # Spatial grouping is audit/sensitivity only, never a primary predictor.
    x_coord = safe_numeric(cohort["X"])
    y_coord = safe_numeric(cohort["Y"])
    valid_xy = x_coord.notna() & y_coord.notna()
    cohort["spatial_group_5km"] = "MissingSpatial"
    cohort.loc[valid_xy, "spatial_group_5km"] = (
        np.floor(x_coord.loc[valid_xy] / config.spatial_grid_m).astype(int).astype(str)
        + "_"
        + np.floor(y_coord.loc[valid_xy] / config.spatial_grid_m).astype(int).astype(str)
    )

    categorical_features = [f for f in RAW_CATEGORICAL_FEATURES if f in cohort.columns]
    numeric_features = [*RAW_NUMERIC_FEATURES, *DERIVED_NUMERIC_FEATURES]

    # Drop features that are entirely missing or constant after cleaning. This
    # is deterministic and is documented before any model fitting.
    dropped_features: list[dict[str, Any]] = []
    retained_cat: list[str] = []
    retained_num: list[str] = []
    for feature in categorical_features:
        nonmissing = cohort[feature].dropna()
        unique_count = int(nonmissing.nunique())
        if unique_count <= 1:
            dropped_features.append(
                {
                    "feature": feature,
                    "reason": "constant_or_all_missing",
                    "nonmissing": int(nonmissing.size),
                    "unique_nonmissing": unique_count,
                }
            )
        else:
            retained_cat.append(feature)
    for feature in numeric_features:
        nonmissing = cohort[feature].dropna()
        unique_count = int(nonmissing.nunique())
        if unique_count <= 1:
            dropped_features.append(
                {
                    "feature": feature,
                    "reason": "constant_or_all_missing",
                    "nonmissing": int(nonmissing.size),
                    "unique_nonmissing": unique_count,
                }
            )
        else:
            retained_num.append(feature)

    pd.DataFrame(dropped_features).to_csv(
        output_dir / "audit" / "dropped_constant_features.csv", index=False, encoding="utf-8-sig"
    )

    class_counts = (
        cohort["severity_label"]
        .value_counts(dropna=False)
        .reindex(CLASS_NAMES)
        .rename_axis("severity")
        .reset_index(name="count")
    )
    class_counts["percent"] = class_counts["count"] / len(cohort) * 100
    class_counts.to_csv(
        output_dir / "audit" / "severity_distribution.csv", index=False, encoding="utf-8-sig"
    )

    class_by_year = pd.crosstab(
        cohort["analysis_year"], cohort["severity_label"], margins=True
    ).reindex(columns=[*CLASS_NAMES, "All"])
    class_by_year.to_csv(
        output_dir / "audit" / "severity_by_year.csv", encoding="utf-8-sig"
    )

    missingness_records = []
    for feature in [*retained_cat, *retained_num]:
        missing = int(cohort[feature].isna().sum())
        missingness_records.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "type": "categorical" if feature in retained_cat else "numeric",
                "missing_count": missing,
                "missing_percent": missing / len(cohort) * 100,
                "unique_nonmissing": int(cohort[feature].nunique(dropna=True)),
            }
        )
    pd.DataFrame(missingness_records).sort_values(
        "missing_percent", ascending=False
    ).to_csv(
        output_dir / "audit" / "feature_missingness.csv", index=False, encoding="utf-8-sig"
    )

    categorical_rows: list[dict[str, Any]] = []
    for feature in retained_cat:
        counts = cohort[feature].fillna("Missing/Unreported").value_counts(dropna=False)
        for level, count in counts.items():
            categorical_rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "level": level,
                    "count": int(count),
                    "percent": float(count / len(cohort) * 100),
                }
            )
    pd.DataFrame(categorical_rows).to_csv(
        output_dir / "audit" / "categorical_descriptive_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_summary = cohort[retained_num].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    numeric_summary.insert(0, "feature_label", [FEATURE_LABELS.get(i, i) for i in numeric_summary.index])
    numeric_summary.to_csv(
        output_dir / "audit" / "numeric_descriptive_summary.csv", encoding="utf-8-sig"
    )

    actual_counts = {
        row["severity"]: int(row["count"])
        for _, row in class_counts.iterrows()
    }

    audit_summary = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "exact_full_row_duplicates_removed": duplicate_full_rows,
        "motorcycle_filter": f"{MOTORCYCLE_FILTER_COLUMN} == {MOTORCYCLE_FILTER_VALUE}",
        "motorcycle_cohort_rows_before_invalid_target_removal": int(len(cohort) + invalid_target_count),
        "invalid_target_rows_removed": invalid_target_count,        "final_motorcycle_cohort_rows": int(len(cohort)),
        "expected_cohort_rows": EXPECTED_COHORT_SIZE,
        "cohort_size_matches_expected": bool(len(cohort) == EXPECTED_COHORT_SIZE),
        "actual_severity_counts": actual_counts,
        "expected_severity_counts": EXPECTED_SEVERITY_COUNTS,
        "severity_counts_match_expected": actual_counts == EXPECTED_SEVERITY_COUNTS,
        "development_years": list(DEVELOPMENT_YEARS),
        "temporal_test_year": TEMPORAL_TEST_YEAR,
        "development_rows": int(cohort["analysis_year"].isin(DEVELOPMENT_YEARS).sum()),
        "temporal_test_rows": int((cohort["analysis_year"] == TEMPORAL_TEST_YEAR).sum()),
        "spatial_groups_5km": int(cohort["spatial_group_5km"].nunique()),
        "retained_categorical_features": retained_cat,
        "retained_numeric_features": retained_num,
        "dropped_features": dropped_features,
    }
    write_json(output_dir / "audit" / "audit_summary.json", audit_summary)

    # Export an analysis-ready cohort containing only governance and model fields.
    export_columns = [
        "target_code",
        "target",
        "severity_label",
        "analysis_year",
        "spatial_group_5km",
        *retained_cat,
        *retained_num,
    ]
    cohort[export_columns].to_csv(
        output_dir / "audit" / "motorcycle_cohort_analysis_ready.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_class_distribution(class_counts, output_dir / "figures" / "severity_distribution.png")
    plot_class_by_year(cohort, output_dir / "figures" / "severity_by_year.png")
    plot_missingness(
        pd.DataFrame(missingness_records),
        output_dir / "figures" / "feature_missingness.png",
    )

    return cohort, retained_cat, retained_num, audit_summary


def audit_and_select_stable_features(
    cohort: pd.DataFrame,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    output_dir: Path,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Apply the locked, outcome-independent feature-availability rule.

    The rule is evaluated before outcome modeling and uses only whether a field
    was recorded, never the target value or model performance. A predictor is
    retained for the primary analysis when:

    * missingness is <= MAX_ANNUAL_MISSING_PERCENT in every study year; and
    * the absolute development-to-temporal-test missingness difference is
      <= MAX_DEV_TEST_MISSING_GAP_PERCENT percentage points.

    All retained and excluded features are documented. Excluded fields remain
    eligible for the separately labeled full-feature sensitivity analysis.
    """
    years = [*DEVELOPMENT_YEARS, TEMPORAL_TEST_YEAR]
    records: list[dict[str, Any]] = []
    candidate_features = [*categorical_features, *numeric_features]

    for feature in candidate_features:
        feature_type = "categorical" if feature in categorical_features else "numeric"
        year_missing: dict[int, float] = {}
        for year in years:
            year_mask = cohort["analysis_year"] == year
            denominator = int(year_mask.sum())
            year_missing[year] = (
                float(cohort.loc[year_mask, feature].isna().mean() * 100)
                if denominator
                else np.nan
            )

        dev_mask = cohort["analysis_year"].isin(DEVELOPMENT_YEARS)
        test_mask = cohort["analysis_year"] == TEMPORAL_TEST_YEAR
        dev_missing = float(cohort.loc[dev_mask, feature].isna().mean() * 100)
        test_missing = float(cohort.loc[test_mask, feature].isna().mean() * 100)
        valid_year_values = [value for value in year_missing.values() if np.isfinite(value)]
        max_annual = max(valid_year_values) if valid_year_values else 100.0
        gap = abs(dev_missing - test_missing)

        annual_ok = max_annual <= MAX_ANNUAL_MISSING_PERCENT
        shift_ok = gap <= MAX_DEV_TEST_MISSING_GAP_PERCENT
        primary_eligible = bool(annual_ok and shift_ok)

        reasons: list[str] = []
        if not annual_ok:
            reasons.append(
                f"maximum annual missingness {max_annual:.1f}% exceeds "
                f"{MAX_ANNUAL_MISSING_PERCENT:.1f}%"
            )
        if not shift_ok:
            reasons.append(
                f"development-test missingness gap {gap:.1f} percentage points exceeds "
                f"{MAX_DEV_TEST_MISSING_GAP_PERCENT:.1f}"
            )
        if not reasons:
            reasons.append("meets locked cross-year availability rule")

        row: dict[str, Any] = {
            "feature": feature,
            "feature_label": FEATURE_LABELS.get(feature, feature),
            "type": feature_type,
            "development_missing_percent": dev_missing,
            "temporal_test_missing_percent": test_missing,
            "absolute_dev_test_missing_gap_pp": gap,
            "maximum_annual_missing_percent": max_annual,
            "primary_eligible": primary_eligible,
            "decision": "retain_primary" if primary_eligible else "full_feature_sensitivity_only",
            "rationale": "; ".join(reasons),
        }
        for year, value in year_missing.items():
            row[f"year_{year}_missing_percent"] = value
        records.append(row)

    stability = pd.DataFrame(records).sort_values(
        ["primary_eligible", "maximum_annual_missing_percent", "feature"],
        ascending=[False, True, True],
    )
    stability.to_csv(
        output_dir / "audit" / "feature_stability_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    year_columns = [
        "feature",
        "feature_label",
        "type",
        *[f"year_{year}_missing_percent" for year in years],
        "development_missing_percent",
        "temporal_test_missing_percent",
        "absolute_dev_test_missing_gap_pp",
        "maximum_annual_missing_percent",
    ]
    stability[year_columns].to_csv(
        output_dir / "audit" / "feature_missingness_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    retained = set(stability.loc[stability["primary_eligible"], "feature"])
    primary_cat = [feature for feature in categorical_features if feature in retained]
    primary_num = [feature for feature in numeric_features if feature in retained]
    excluded = stability.loc[~stability["primary_eligible"]].copy()
    excluded.to_csv(
        output_dir / "audit" / "excluded_unstable_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_json(
        output_dir / "audit" / "primary_feature_set.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "selection_rule": {
                "maximum_annual_missing_percent": MAX_ANNUAL_MISSING_PERCENT,
                "maximum_development_test_missingness_gap_percentage_points": MAX_DEV_TEST_MISSING_GAP_PERCENT,
                "uses_outcome_information": False,
            },
            "primary_categorical_features": primary_cat,
            "primary_numeric_features": primary_num,
            "primary_feature_count": len(primary_cat) + len(primary_num),
            "full_feature_candidate_count": len(candidate_features),
            "excluded_unstable_features": excluded[
                ["feature", "feature_label", "rationale"]
            ].to_dict(orient="records"),
        },
    )

    if len(primary_cat) + len(primary_num) < 5:
        raise RuntimeError(
            "The stable-feature rule retained fewer than five predictors; review "
            "audit/feature_stability_selection.csv before continuing."
        )

    logging.info(
        "Stable-feature primary set: %d categorical + %d numeric; excluded %d unstable fields.",
        len(primary_cat),
        len(primary_num),
        int((~stability["primary_eligible"]).sum()),
    )
    return primary_cat, primary_num, stability


def plot_feature_stability(stability: pd.DataFrame, path: Path) -> None:
    """Plot cross-year missingness for the audited candidate predictors."""
    if stability.empty:
        return
    year_columns = [
        f"year_{year}_missing_percent" for year in [*DEVELOPMENT_YEARS, TEMPORAL_TEST_YEAR]
    ]
    data = stability.sort_values("maximum_annual_missing_percent", ascending=True)
    y = np.arange(len(data))
    plt.figure(figsize=(10, max(6, 0.32 * len(data))))
    offsets = np.linspace(-0.22, 0.22, len(year_columns))
    for offset, column, year in zip(
        offsets, year_columns, [*DEVELOPMENT_YEARS, TEMPORAL_TEST_YEAR]
    ):
        plt.scatter(data[column], y + offset, label=f"Year {year}", s=26)
    plt.axvline(
        MAX_ANNUAL_MISSING_PERCENT,
        linestyle="--",
        linewidth=1,
        label=f"Primary threshold ({MAX_ANNUAL_MISSING_PERCENT:.0f}%)",
    )
    plt.yticks(y, data["feature_label"])
    plt.xlabel("Missing or unreported (%)")
    plt.title("Cross-year predictor availability audit")
    plt.legend()
    plt.grid(axis="x", alpha=0.2)
    save_figure(path)


def plot_class_distribution(class_counts: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(class_counts["severity"], class_counts["count"])
    plt.ylabel("Number of crashes")
    plt.xlabel("Severity")
    plt.title("Motorcycle crash-severity distribution")
    for bar, count, percent in zip(bars, class_counts["count"], class_counts["percent"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}\n({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(path)


def plot_class_by_year(cohort: pd.DataFrame, path: Path) -> None:
    table = pd.crosstab(cohort["analysis_year"], cohort["severity_label"]).reindex(
        columns=CLASS_NAMES, fill_value=0
    )
    ax = table.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Persian calendar year")
    ax.set_ylabel("Number of crashes")
    ax.set_title("Crash severity by analysis year")
    ax.legend(title="Severity")
    plt.xticks(rotation=0)
    save_figure(path)


def plot_missingness(missingness: pd.DataFrame, path: Path) -> None:
    if missingness.empty:
        return
    data = missingness.sort_values("missing_percent", ascending=True)
    plt.figure(figsize=(9, max(5, 0.28 * len(data))))
    plt.barh(data["feature_label"], data["missing_percent"])
    plt.xlabel("Missing or unreported (%)")
    plt.title("Predictor missingness after data cleaning")
    save_figure(path)


# -----------------------------------------------------------------------------
# Pipelines and model search
# -----------------------------------------------------------------------------


class AutoBalancedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a classifier and compute balanced sample weights inside each fit.

    Because every fit call receives only its training-fold y values, class
    weights cannot leak prevalence information from validation or test data.
    """

    def __init__(self, estimator: BaseEstimator, use_balanced_weights: bool = True):
        self.estimator = estimator
        self.use_balanced_weights = use_balanced_weights

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "AutoBalancedClassifier":
        self.estimator_ = clone(self.estimator)
        params = dict(fit_params)
        if self.use_balanced_weights:
            weights = compute_sample_weight(class_weight="balanced", y=np.asarray(y))
            if "sample_weight" in params and params["sample_weight"] is not None:
                weights = weights * np.asarray(params.pop("sample_weight"))
            params["sample_weight"] = weights
        self.estimator_.fit(X, y, **params)
        self.classes_ = np.asarray(self.estimator_.classes_)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def decision_function(self, X: Any) -> np.ndarray:
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError("Wrapped estimator has no decision_function.")
        return self.estimator_.decision_function(X)


AutoBalancedClassifier.__module__ = "motorcycle_crash_pipeline"


def make_preprocessor(
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    min_frequency: int,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing/Unreported")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def base_estimators(seed: int) -> dict[str, BaseEstimator]:
    return {
        "RandomForest": RandomForestClassifier(
            random_state=seed,
            n_jobs=1,
            bootstrap=True,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            subsample_freq=1,
        ),
    }


def model_search_spaces(mode: str = "full") -> dict[str, dict[str, Any]]:
    if mode == "smoke":
        return {
            "RandomForest": {
                "model__estimator__n_estimators": [100],
                "model__estimator__max_depth": [12],
                "model__estimator__min_samples_split": [5],
                "model__estimator__min_samples_leaf": [2],
                "model__estimator__max_features": ["sqrt"],
                "model__estimator__criterion": ["gini"],
            },
            "XGBoost": {
                "model__estimator__n_estimators": [100],
                "model__estimator__max_depth": [4],
                "model__estimator__learning_rate": [0.05],
                "model__estimator__subsample": [0.8],
                "model__estimator__colsample_bytree": [0.8],
                "model__estimator__min_child_weight": [1.0],
                "model__estimator__gamma": [0.0],
                "model__estimator__reg_alpha": [0.0],
                "model__estimator__reg_lambda": [1.0],
            },
            "LightGBM": {
                "model__estimator__n_estimators": [100],
                "model__estimator__num_leaves": [31],
                "model__estimator__max_depth": [-1],
                "model__estimator__learning_rate": [0.05],
                "model__estimator__min_child_samples": [20],
                "model__estimator__subsample": [0.8],
                "model__estimator__colsample_bytree": [0.8],
                "model__estimator__reg_alpha": [0.0],
                "model__estimator__reg_lambda": [1.0],
            },
        }
    return {
        "RandomForest": {
            "model__estimator__n_estimators": randint(300, 1201),
            "model__estimator__max_depth": [None, 6, 10, 14, 18, 24, 32],
            "model__estimator__min_samples_split": randint(2, 21),
            "model__estimator__min_samples_leaf": randint(1, 13),
            "model__estimator__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
            "model__estimator__criterion": ["gini", "entropy", "log_loss"],
        },
        "XGBoost": {
            "model__estimator__n_estimators": randint(250, 1201),
            "model__estimator__max_depth": randint(2, 9),
            "model__estimator__learning_rate": loguniform(0.01, 0.20),
            "model__estimator__subsample": uniform(0.60, 0.40),
            "model__estimator__colsample_bytree": uniform(0.50, 0.50),
            "model__estimator__min_child_weight": loguniform(0.5, 30),
            "model__estimator__gamma": uniform(0.0, 5.0),
            "model__estimator__reg_alpha": loguniform(1e-5, 10.0),
            "model__estimator__reg_lambda": loguniform(0.1, 100.0),
        },
        "LightGBM": {
            "model__estimator__n_estimators": randint(250, 1201),
            "model__estimator__num_leaves": randint(15, 129),
            "model__estimator__max_depth": [-1, 4, 6, 8, 10, 12, 16],
            "model__estimator__learning_rate": loguniform(0.01, 0.20),
            "model__estimator__min_child_samples": randint(10, 101),
            "model__estimator__subsample": uniform(0.60, 0.40),
            "model__estimator__colsample_bytree": uniform(0.50, 0.50),
            "model__estimator__reg_alpha": loguniform(1e-5, 10.0),
            "model__estimator__reg_lambda": loguniform(0.1, 100.0),
        },
    }


def build_primary_pipeline(
    model_name: str,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    config: RunConfig,
    seed: int,
) -> Pipeline:
    estimator = base_estimators(seed)[model_name]
    return Pipeline(
        steps=[
            (
                "preprocess",
                make_preprocessor(
                    categorical_features,
                    numeric_features,
                    config.onehot_min_frequency,
                ),
            ),
            (
                "model",
                AutoBalancedClassifier(estimator=estimator, use_balanced_weights=True),
            ),
        ]
    )


def build_smotenc_pipeline(
    model_name: str,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    config: RunConfig,
    seed: int,
) -> ImbPipeline:
    # Stage 1: fold-specific imputation and ordinal coding for SMOTENC only.
    pre_smote = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=False), list(numeric_features)),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant", fill_value="Missing/Unreported"
                            ),
                        ),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                list(categorical_features),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    n_num = len(numeric_features)
    cat_indices = list(range(n_num, n_num + len(categorical_features)))
    post_smote = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(range(n_num))),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                cat_indices,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    estimator = base_estimators(seed)[model_name]
    return ImbPipeline(
        steps=[
            ("pre_smote", pre_smote),
            (
                "smote",
                SMOTENC(
                    categorical_features=cat_indices,
                    sampling_strategy="not majority",
                    random_state=seed,
                    k_neighbors=3,
                ),
            ),
            ("post_smote", post_smote),
            (
                "model",
                AutoBalancedClassifier(estimator=estimator, use_balanced_weights=False),
            ),
        ]
    )


def scoring_dictionary() -> dict[str, Any]:
    return {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "balanced_accuracy": "balanced_accuracy",
        "neg_log_loss": "neg_log_loss",
        "accuracy": "accuracy",
    }


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_onehot = np.eye(len(CLASS_ORDER))[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    y_true = np.asarray(y_true)
    for class_idx, class_name in zip(CLASS_ORDER, CLASS_NAMES):
        observed = (y_true == class_idx).astype(int)
        probs = probabilities[:, class_idx]
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for left, right in zip(edges[:-1], edges[1:]):
            if right == 1.0:
                mask = (probs >= left) & (probs <= right)
            else:
                mask = (probs >= left) & (probs < right)
            if not np.any(mask):
                continue
            ece += np.mean(mask) * abs(np.mean(observed[mask]) - np.mean(probs[mask]))
        per_class[class_name] = float(ece)
    return float(np.mean(list(per_class.values()))), per_class


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    prefix = f"{prefix}_" if prefix else ""
    metrics: dict[str, float] = {
        f"{prefix}accuracy": accuracy_score(y_true, predictions),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        f"{prefix}macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        f"{prefix}weighted_f1": f1_score(
            y_true, predictions, average="weighted", zero_division=0
        ),
        f"{prefix}mcc": matthews_corrcoef(y_true, predictions),
        f"{prefix}quadratic_weighted_kappa": cohen_kappa_score(
            y_true, predictions, weights="quadratic"
        ),
        f"{prefix}ordinal_mae": float(np.mean(np.abs(predictions - y_true))),
        f"{prefix}log_loss": log_loss(y_true, probabilities, labels=CLASS_ORDER),
        f"{prefix}multiclass_brier": multiclass_brier(y_true, probabilities),
    }
    try:
        metrics[f"{prefix}macro_roc_auc_ovr"] = roc_auc_score(
            y_true, probabilities, multi_class="ovr", average="macro"
        )
        metrics[f"{prefix}weighted_roc_auc_ovr"] = roc_auc_score(
            y_true, probabilities, multi_class="ovr", average="weighted"
        )
    except ValueError:
        metrics[f"{prefix}macro_roc_auc_ovr"] = np.nan
        metrics[f"{prefix}weighted_roc_auc_ovr"] = np.nan
    y_bin = label_binarize(y_true, classes=CLASS_ORDER)
    try:
        metrics[f"{prefix}macro_average_precision"] = average_precision_score(
            y_bin, probabilities, average="macro"
        )
    except ValueError:
        metrics[f"{prefix}macro_average_precision"] = np.nan
    ece, per_class_ece = expected_calibration_error(y_true, probabilities)
    metrics[f"{prefix}macro_ece"] = ece
    for name, value in per_class_ece.items():
        metrics[f"{prefix}ece_{name.lower()}"] = value
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=CLASS_ORDER, zero_division=0
    )
    for idx, name in enumerate(CLASS_NAMES):
        metrics[f"{prefix}{name.lower()}_precision"] = precision[idx]
        metrics[f"{prefix}{name.lower()}_recall"] = recall[idx]
        metrics[f"{prefix}{name.lower()}_f1"] = f1[idx]
        metrics[f"{prefix}{name.lower()}_support"] = support[idx]
    return {k: float(v) for k, v in metrics.items()}


def nested_cv_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    config: RunConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Starting repeated nested CV for %s.", model_name)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
        random_state=config.seed,
    )
    spaces = model_search_spaces(config.mode)[model_name]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_seed = config.seed + fold_number * 1009
        pipeline = build_primary_pipeline(
            model_name, categorical_features, numeric_features, config, fold_seed
        )
        inner_cv = StratifiedKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=fold_seed,
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=spaces,
            n_iter=config.search_iterations,
            scoring=scoring_dictionary(),
            refit="macro_f1",
            cv=inner_cv,
            random_state=fold_seed,
            n_jobs=config.n_jobs,
            verbose=0,
            return_train_score=False,
            error_score="raise",
        )
        start = time.time()
        search.fit(X.iloc[train_idx], y[train_idx])
        elapsed = time.time() - start
        probabilities = search.best_estimator_.predict_proba(X.iloc[valid_idx])
        metrics = evaluate_probabilities(y[valid_idx], probabilities)
        row = {
            "model": model_name,
            "outer_fold": fold_number,
            "train_size": len(train_idx),
            "validation_size": len(valid_idx),
            "best_inner_macro_f1": float(search.best_score_),
            "elapsed_seconds": elapsed,
            "best_params_json": json.dumps(
                json_safe(search.best_params_), ensure_ascii=False, sort_keys=True
            ),
            **metrics,
        }
        fold_rows.append(row)
        for local_pos, original_idx in enumerate(valid_idx):
            prediction_rows.append(
                {
                    "model": model_name,
                    "outer_fold": fold_number,
                    "development_row_index": int(original_idx),
                    "y_true": int(y[original_idx]),
                    "p_pdo": float(probabilities[local_pos, 0]),
                    "p_injury": float(probabilities[local_pos, 1]),
                    "p_fatal": float(probabilities[local_pos, 2]),
                }
            )
        logging.info(
            "%s fold %d/%d: outer macro F1=%.4f, fatal F1=%.4f, %.1f sec",
            model_name,
            fold_number,
            config.outer_splits * config.outer_repeats,
            metrics["macro_f1"],
            metrics["fatal_f1"],
            elapsed,
        )

    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    fold_frame.to_csv(
        output_dir / "modeling" / f"nested_cv_{model_name}_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_frame.to_csv(
        output_dir / "modeling" / f"nested_oof_{model_name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return fold_frame, prediction_frame


def summarize_nested_cv(all_folds: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "accuracy",
        "fatal_precision",
        "fatal_recall",
        "fatal_f1",
        "log_loss",
        "multiclass_brier",
        "macro_roc_auc_ovr",
        "macro_average_precision",
        "macro_ece",
        "quadratic_weighted_kappa",
        "ordinal_mae",
    ]
    rows: list[dict[str, Any]] = []
    for model, group in all_folds.groupby("model"):
        row: dict[str, Any] = {"model": model, "outer_folds": len(group)}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["macro_f1_mean", "fatal_f1_mean", "log_loss_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def paired_model_comparison(
    all_folds: pd.DataFrame,
    cv_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Summarize fold-aligned model differences without overclaiming superiority."""
    metrics = ["macro_f1", "fatal_f1", "balanced_accuracy", "log_loss"]
    records: list[dict[str, Any]] = []
    models = sorted(all_folds["model"].unique())
    for metric in metrics:
        pivot = all_folds.pivot(index="outer_fold", columns="model", values=metric)
        for idx, model_a in enumerate(models):
            for model_b in models[idx + 1 :]:
                paired = pivot[[model_a, model_b]].dropna()
                difference = paired[model_a] - paired[model_b]
                records.append(
                    {
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "paired_folds": len(difference),
                        "mean_a_minus_b": float(difference.mean()),
                        "sd_a_minus_b": float(difference.std(ddof=1)),
                        "minimum_a_minus_b": float(difference.min()),
                        "maximum_a_minus_b": float(difference.max()),
                        "model_a_better_folds": int((difference > 0).sum()),
                        "ties": int((difference == 0).sum()),
                        "model_b_better_folds": int((difference < 0).sum()),
                    }
                )
    paired_frame = pd.DataFrame(records)
    paired_frame.to_csv(
        output_dir / "modeling" / "paired_outer_fold_model_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ranked = cv_summary.sort_values(
        ["macro_f1_mean", "fatal_f1_mean", "log_loss_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected = str(ranked.iloc[0]["model"])
    runner_up = str(ranked.iloc[1]["model"]) if len(ranked) > 1 else None
    macro_gap = (
        float(ranked.iloc[0]["macro_f1_mean"] - ranked.iloc[1]["macro_f1_mean"])
        if len(ranked) > 1
        else np.nan
    )
    practical_tie = bool(
        len(ranked) > 1 and abs(macro_gap) < PRACTICAL_MODEL_TIE_MACRO_F1
    )
    summary = {
        "selected_model": selected,
        "runner_up_model": runner_up,
        "top_minus_runner_up_macro_f1": macro_gap,
        "practical_tie_threshold": PRACTICAL_MODEL_TIE_MACRO_F1,
        "models_practically_tied": practical_tie,
        "interpretation": (
            "Selected by the prespecified highest mean outer-fold macro F1; "
            "the leading models are practically tied and superiority should not be claimed."
            if practical_tie
            else "Selected by the prespecified highest mean outer-fold macro F1."
        ),
    }
    write_json(output_dir / "modeling" / "model_selection_interpretation.json", summary)
    return summary


def aggregate_repeated_oof(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    aggregate = (
        predictions.groupby("development_row_index", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            p_pdo=("p_pdo", "mean"),
            p_injury=("p_injury", "mean"),
            p_fatal=("p_fatal", "mean"),
            predictions_per_row=("outer_fold", "count"),
        )
        .sort_values("development_row_index")
    )
    y = aggregate["y_true"].to_numpy(dtype=int)
    probabilities = aggregate[["p_pdo", "p_injury", "p_fatal"]].to_numpy(dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return y, probabilities


def fit_final_model_search(
    model_name: str,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    config: RunConfig,
) -> RandomizedSearchCV:
    logging.info("Final development-only hyperparameter search for %s.", model_name)
    pipeline = build_primary_pipeline(
        model_name, categorical_features, numeric_features, config, config.seed + 50000
    )
    cv = StratifiedKFold(
        n_splits=max(3, config.inner_splits),
        shuffle=True,
        random_state=config.seed + 50000,
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=model_search_spaces(config.mode)[model_name],
        n_iter=config.final_search_iterations,
        scoring=scoring_dictionary(),
        refit="macro_f1",
        cv=cv,
        random_state=config.seed + 50000,
        n_jobs=config.n_jobs,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_dev, y_dev)
    return search


# -----------------------------------------------------------------------------
# Calibration, thresholding, bootstrap uncertainty
# -----------------------------------------------------------------------------


class MulticlassLogitCalibrator:
    """Multinomial logistic recalibration fitted to out-of-fold probabilities."""

    def __init__(self, epsilon: float = 1e-6, random_state: int = 42):
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "MulticlassLogitCalibrator":
        probabilities = np.clip(np.asarray(probabilities), self.epsilon, 1 - self.epsilon)
        features = np.log(probabilities)
        self.model_ = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=self.random_state,
        )
        self.model_.fit(features, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(np.asarray(probabilities), self.epsilon, 1 - self.epsilon)
        calibrated = self.model_.predict_proba(np.log(probabilities))
        return calibrated / calibrated.sum(axis=1, keepdims=True)


MulticlassLogitCalibrator.__module__ = "motorcycle_crash_pipeline"


class OneVsRestIsotonicCalibrator:
    """One-vs-rest isotonic calibration with probability renormalization."""

    def __init__(self, out_of_bounds: str = "clip"):
        self.out_of_bounds = out_of_bounds

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "OneVsRestIsotonicCalibrator":
        self.models_: list[IsotonicRegression] = []
        for class_idx in CLASS_ORDER:
            model = IsotonicRegression(out_of_bounds=self.out_of_bounds)
            model.fit(probabilities[:, class_idx], (y == class_idx).astype(int))
            self.models_.append(model)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = np.column_stack(
            [model.predict(probabilities[:, idx]) for idx, model in enumerate(self.models_)]
        )
        calibrated = np.clip(calibrated, 1e-9, None)
        return calibrated / calibrated.sum(axis=1, keepdims=True)


OneVsRestIsotonicCalibrator.__module__ = "motorcycle_crash_pipeline"


def select_fatal_threshold(
    y_true: np.ndarray,
    fatal_probabilities: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, pd.DataFrame]:
    y_binary = (np.asarray(y_true) == FATAL_CLASS).astype(int)
    thresholds = np.linspace(0.001, 0.999, 999)
    rows = []
    for threshold in thresholds:
        pred = (fatal_probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_binary, pred, average="binary", zero_division=0
        )
        fbeta = fbeta_score(y_binary, pred, beta=beta, zero_division=0)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": fbeta,
                "flagged": int(pred.sum()),
            }
        )
    table = pd.DataFrame(rows)
    # Tie-break toward higher recall, then lower threshold.
    best = table.sort_values(
        ["f2", "recall", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    return float(best["threshold"]), table


def fatal_screening_metrics(
    y_true: np.ndarray,
    fatal_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_binary = (np.asarray(y_true) == FATAL_CLASS).astype(int)
    pred = (fatal_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_binary, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_binary, pred, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "fatal_threshold": threshold,
        "fatal_precision": float(precision),
        "fatal_recall": float(recall),
        "fatal_specificity": float(specificity),
        "fatal_f1": float(f1),
        "fatal_f2": float(fbeta_score(y_binary, pred, beta=2, zero_division=0)),
        "fatal_tp": int(tp),
        "fatal_fp": int(fp),
        "fatal_tn": int(tn),
        "fatal_fn": int(fn),
        "fatal_flagged": int(pred.sum()),
    }

def bootstrap_temporal_metrics(
    y_true: np.ndarray,
    classification_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    fatal_threshold: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified bootstrap for distinct classification and calibration targets.

    Hard multiclass classification metrics use the locked uncalibrated model.
    Probability-quality and screening metrics use development-fitted calibrated
    probabilities. Keeping these roles separate prevents calibration-induced
    argmax changes from obscuring the classifier's discrimination performance.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    classification_probabilities = np.asarray(
        classification_probabilities, dtype=float
    )
    calibrated_probabilities = np.asarray(calibrated_probabilities, dtype=float)
    class_indices = {cls: np.where(y_true == cls)[0] for cls in CLASS_ORDER}
    rows: list[dict[str, float]] = []
    for iteration in range(iterations):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices.values()
            ]
        )
        rng.shuffle(sampled)
        classification = evaluate_probabilities(
            y_true[sampled], classification_probabilities[sampled]
        )
        calibration = evaluate_probabilities(
            y_true[sampled], calibrated_probabilities[sampled]
        )
        screening = fatal_screening_metrics(
            y_true[sampled],
            calibrated_probabilities[sampled, FATAL_CLASS],
            fatal_threshold,
        )
        rows.append(
            {
                "iteration": iteration + 1,
                "classification_macro_f1": classification["macro_f1"],
                "classification_balanced_accuracy": classification[
                    "balanced_accuracy"
                ],
                "classification_fatal_f1": classification["fatal_f1"],
                "classification_accuracy": classification["accuracy"],
                "calibrated_log_loss": calibration["log_loss"],
                "calibrated_multiclass_brier": calibration[
                    "multiclass_brier"
                ],
                "calibrated_macro_ece": calibration["macro_ece"],
                "fatal_screen_precision": screening["fatal_precision"],
                "fatal_screen_recall": screening["fatal_recall"],
                "fatal_screen_f2": screening["fatal_f2"],
            }
        )
    return pd.DataFrame(rows)

def summarize_bootstrap(bootstrap: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in bootstrap.columns:
        if column == "iteration":
            continue
        values = bootstrap[column].dropna().to_numpy()
        records.append(
            {
                "metric": column,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_sd": float(np.std(values, ddof=1)),
                "ci_2.5_percent": float(np.quantile(values, 0.025)),
                "ci_97.5_percent": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Evaluation plots
# -----------------------------------------------------------------------------


def save_confusion_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
    title: str,
    normalize: Optional[str] = None,
) -> None:
    predictions = probabilities.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=CLASS_ORDER,
        display_labels=CLASS_NAMES,
        normalize=normalize,
        values_format=".2f" if normalize else "d",
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curves(
    y_true: np.ndarray,
    probability_sets: Mapping[str, np.ndarray],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for class_idx, (class_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        observed = (y_true == class_idx).astype(int)
        for label, probabilities in probability_sets.items():
            frac_pos, mean_pred = calibration_curve(
                observed,
                probabilities[:, class_idx],
                n_bins=10,
                strategy="quantile",
            )
            ax.plot(mean_pred, frac_pos, marker="o", label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(class_name)
        ax.set_xlabel("Mean predicted probability")
        if class_idx == 0:
            ax.set_ylabel("Observed proportion")
        ax.grid(alpha=0.25)
    axes[-1].legend(loc="best")
    fig.suptitle("Temporal-test calibration by severity class")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    roc_path: Path,
    pr_path: Path,
) -> None:
    y_bin = label_binarize(y_true, classes=CLASS_ORDER)
    plt.figure(figsize=(7, 5.5))
    for class_idx, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, class_idx], probabilities[:, class_idx])
        auc = roc_auc_score(y_bin[:, class_idx], probabilities[:, class_idx])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Temporal-test one-vs-rest ROC curves")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(roc_path)

    plt.figure(figsize=(7, 5.5))
    for class_idx, class_name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(
            y_bin[:, class_idx], probabilities[:, class_idx]
        )
        ap = average_precision_score(y_bin[:, class_idx], probabilities[:, class_idx])
        plt.plot(recall, precision, label=f"{class_name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Temporal-test one-vs-rest precision-recall curves")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(pr_path)


def plot_model_cv_comparison(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("macro_f1_mean", ascending=False)
    plt.figure(figsize=(8, 5))
    x = np.arange(len(ordered))
    plt.errorbar(
        x,
        ordered["macro_f1_mean"],
        yerr=ordered["macro_f1_sd"],
        fmt="o",
        capsize=5,
    )
    plt.xticks(x, ordered["model"])
    plt.ylabel("Outer-fold macro F1")
    plt.title("Repeated nested cross-validation model comparison")
    plt.grid(axis="y", alpha=0.25)
    save_figure(path)


# -----------------------------------------------------------------------------
# SMOTENC and spatial robustness sensitivity analyses
# -----------------------------------------------------------------------------


def set_model_parameters(pipeline: BaseEstimator, params: Mapping[str, Any]) -> BaseEstimator:
    valid = pipeline.get_params(deep=True)
    filtered = {k: v for k, v in params.items() if k in valid}
    return pipeline.set_params(**filtered)


def fixed_pipeline_cv(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    splitter: Iterable[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> pd.DataFrame:
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter, start=1):
        fitted = clone(pipeline)
        fitted.fit(X.iloc[train_idx], y[train_idx])
        probabilities = fitted.predict_proba(X.iloc[valid_idx])
        rows.append({"analysis": label, "fold": fold, **evaluate_probabilities(y[valid_idx], probabilities)})
    return pd.DataFrame(rows)


def run_smotenc_sensitivity(
    selected_model: str,
    best_params: Mapping[str, Any],
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    config: RunConfig,
    output_dir: Path,
) -> pd.DataFrame:
    logging.info("Running training-fold-only SMOTENC sensitivity analysis.")
    splitter = list(
        RepeatedStratifiedKFold(
            n_splits=config.outer_splits,
            n_repeats=config.outer_repeats,
            random_state=config.seed + 70000,
        ).split(X_dev, y_dev)
    )
    weighted = build_primary_pipeline(
        selected_model,
        categorical_features,
        numeric_features,
        config,
        config.seed + 70001,
    )
    weighted = set_model_parameters(weighted, best_params)
    smotenc = build_smotenc_pipeline(
        selected_model,
        categorical_features,
        numeric_features,
        config,
        config.seed + 70002,
    )
    smotenc = set_model_parameters(smotenc, best_params)
    weighted_results = fixed_pipeline_cv(
        weighted, X_dev, y_dev, splitter, "Class-weighted primary"
    )
    smotenc_results = fixed_pipeline_cv(
        smotenc, X_dev, y_dev, splitter, "Fold-only SMOTENC sensitivity"
    )
    results = pd.concat([weighted_results, smotenc_results], ignore_index=True)
    results.to_csv(
        output_dir / "sensitivity" / "smotenc_sensitivity_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        results.groupby("analysis")
        .agg(
            folds=("fold", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            fatal_f1_mean=("fatal_f1", "mean"),
            fatal_f1_sd=("fatal_f1", "std"),
            log_loss_mean=("log_loss", "mean"),
            brier_mean=("multiclass_brier", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        output_dir / "sensitivity" / "smotenc_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def run_spatial_sensitivity(
    selected_pipeline: BaseEstimator,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    spatial_groups: np.ndarray,
    config: RunConfig,
    output_dir: Path,
) -> pd.DataFrame:
    logging.info("Running 5-km stratified spatial-group sensitivity analysis.")
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.seed + 80000,
    )
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X_dev, y_dev, groups=spatial_groups), start=1
    ):
        fitted = clone(selected_pipeline)
        fitted.fit(X_dev.iloc[train_idx], y_dev[train_idx])
        probs = fitted.predict_proba(X_dev.iloc[valid_idx])
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "validation_rows": len(valid_idx),
                "train_groups": len(np.unique(spatial_groups[train_idx])),
                "validation_groups": len(np.unique(spatial_groups[valid_idx])),
                **evaluate_probabilities(y_dev[valid_idx], probs),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / "sensitivity" / "spatial_group_sensitivity_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return results


def run_full_feature_sensitivity(
    selected_model: str,
    X_dev_full: pd.DataFrame,
    y_dev: np.ndarray,
    X_test_full: pd.DataFrame,
    y_test: np.ndarray,
    full_categorical_features: Sequence[str],
    full_numeric_features: Sequence[str],
    config: RunConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Retune the selected family with all candidate predictors as sensitivity.

    This model is explicitly secondary because several included predictors have
    strong year-specific availability shifts. It is never used for primary
    model selection, calibration, threshold selection, SHAP, or interaction
    interpretation.
    """
    logging.info(
        "Running full-feature sensitivity with %d predictors.",
        len(full_categorical_features) + len(full_numeric_features),
    )
    search = fit_final_model_search(
        selected_model,
        X_dev_full,
        y_dev,
        full_categorical_features,
        full_numeric_features,
        config,
    )
    pipeline = search.best_estimator_
    probabilities = pipeline.predict_proba(X_test_full)
    metrics = evaluate_probabilities(y_test, probabilities)
    joblib.dump(
        pipeline,
        output_dir / "models" / "full_feature_sensitivity_pipeline.joblib",
    )
    write_json(
        output_dir / "sensitivity" / "full_feature_model_selection.json",
        {
            "selected_model_family_fixed_from_primary_analysis": selected_model,
            "final_inner_cv_macro_f1": float(search.best_score_),
            "best_params": search.best_params_,
            "categorical_features": list(full_categorical_features),
            "numeric_features": list(full_numeric_features),
        },
    )
    write_json(
        output_dir / "sensitivity" / "full_feature_temporal_test_metrics.json",
        metrics,
    )
    pd.DataFrame(
        {
            "y_true": y_test,
            "severity_true": [INTERNAL_TO_NAME[i] for i in y_test],
            "p_pdo": probabilities[:, 0],
            "p_injury": probabilities[:, 1],
            "p_fatal": probabilities[:, 2],
            "predicted_class": probabilities.argmax(axis=1),
            "predicted_severity": [
                INTERNAL_TO_NAME[i] for i in probabilities.argmax(axis=1)
            ],
        }
    ).to_csv(
        output_dir / "sensitivity" / "full_feature_temporal_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "feature_count": len(full_categorical_features) + len(full_numeric_features),
        "inner_cv_macro_f1": float(search.best_score_),
        "test_metrics": metrics,
        "best_params": search.best_params_,
    }


# -----------------------------------------------------------------------------
# SHAP and interaction analysis
# -----------------------------------------------------------------------------


def transformed_matrix_and_metadata(
    fitted_pipeline: BaseEstimator,
    X: pd.DataFrame,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
) -> tuple[Any, list[str], list[str]]:
    """Return transformed data, encoded names, and original-feature group names."""
    if "preprocess" in fitted_pipeline.named_steps:
        preprocessor = fitted_pipeline.named_steps["preprocess"]
        matrix = preprocessor.transform(X)
        encoded_names = list(preprocessor.get_feature_names_out())
        group_names = []
        for name in encoded_names:
            if name.startswith("num__"):
                raw_name = name[len("num__") :]
                # Missing-indicator names are generated by SimpleImputer.
                if raw_name.startswith("missingindicator_"):
                    raw_name = raw_name[len("missingindicator_") :]
                group_names.append(raw_name)
            elif name.startswith("cat__"):
                remainder = name[len("cat__") :]
                matches = [f for f in categorical_features if remainder == f or remainder.startswith(f + "_")]
                group_names.append(max(matches, key=len) if matches else remainder)
            else:
                group_names.append(name)
        return matrix, encoded_names, group_names

    # SMOTENC sensitivity pipelines are not used as the primary SHAP model, but
    # this branch keeps the function complete.
    pre = fitted_pipeline.named_steps["pre_smote"].transform(X)
    post = fitted_pipeline.named_steps["post_smote"]
    matrix = post.transform(pre)
    encoded_names = list(post.get_feature_names_out())
    group_names = []
    for name in encoded_names:
        if name.startswith("num__"):
            idx = int(name.split("x")[-1]) if re.search(r"x\d+$", name) else None
            group_names.append(numeric_features[idx] if idx is not None and idx < len(numeric_features) else name)
        else:
            remainder = name[len("cat__") :] if name.startswith("cat__") else name
            match = re.match(r"x(\d+)_", remainder)
            if match:
                transformed_idx = int(match.group(1))
                cat_idx = transformed_idx - len(numeric_features)
                group_names.append(categorical_features[cat_idx] if 0 <= cat_idx < len(categorical_features) else name)
            else:
                group_names.append(name)
    return matrix, encoded_names, group_names


def unwrap_tree_estimator(fitted_pipeline: BaseEstimator) -> BaseEstimator:
    wrapper = fitted_pipeline.named_steps["model"]
    return wrapper.estimator_


def standardize_shap_values(
    shap_values: Any,
    n_samples: int,
    n_features: int,
    n_classes: int = 3,
) -> np.ndarray:
    """Return SHAP values with shape (samples, features, classes)."""
    if isinstance(shap_values, list):
        array = np.stack([np.asarray(v) for v in shap_values], axis=-1)
    else:
        array = np.asarray(shap_values)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape == (n_samples, n_features, n_classes):
        return array
    if array.shape == (n_classes, n_samples, n_features):
        return np.transpose(array, (1, 2, 0))
    if array.shape == (n_samples, n_classes, n_features):
        return np.transpose(array, (0, 2, 1))
    raise ValueError(f"Unrecognized SHAP value shape: {array.shape}")


def standardize_interaction_values(
    values: Any,
    n_samples: int,
    n_features: int,
    n_classes: int = 3,
) -> np.ndarray:
    """Return SHAP interactions with shape (samples, features, features, classes)."""
    if isinstance(values, list):
        array = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        array = np.asarray(values)
    if array.shape == (n_samples, n_features, n_features, n_classes):
        return array
    if array.shape == (n_classes, n_samples, n_features, n_features):
        return np.transpose(array, (1, 2, 3, 0))
    if array.shape == (n_samples, n_classes, n_features, n_features):
        return np.transpose(array, (0, 2, 3, 1))
    raise ValueError(f"Unrecognized SHAP interaction shape: {array.shape}")


def group_shap_values(
    shap_array: np.ndarray,
    encoded_group_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    groups = list(dict.fromkeys(encoded_group_names))
    group_index = {group: idx for idx, group in enumerate(groups)}
    grouped = np.zeros((shap_array.shape[0], len(groups), shap_array.shape[2]), dtype=float)
    for encoded_idx, group in enumerate(encoded_group_names):
        grouped[:, group_index[group], :] += shap_array[:, encoded_idx, :]
    return grouped, groups


def grouping_matrix(encoded_group_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    groups = list(dict.fromkeys(encoded_group_names))
    group_index = {group: idx for idx, group in enumerate(groups)}
    matrix = np.zeros((len(encoded_group_names), len(groups)), dtype=float)
    for encoded_idx, group in enumerate(encoded_group_names):
        matrix[encoded_idx, group_index[group]] = 1.0
    return matrix, groups


def run_shap_analysis(
    fitted_pipeline: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    config: RunConfig,
    output_dir: Path,
) -> dict[str, Any]:
    logging.info("Starting TreeSHAP analysis on the locked final model.")
    sample_indices = stratified_sample_indices(
        y_test,
        max_samples=min(config.shap_max_samples, len(y_test)),
        seed=config.seed + 90000,
        include_all_class=FATAL_CLASS,
    )
    X_sample = X_test.iloc[sample_indices].copy()
    y_sample = y_test[sample_indices]
    transformed, encoded_names, encoded_groups = transformed_matrix_and_metadata(
        fitted_pipeline, X_sample, categorical_features, numeric_features
    )
    if sparse.issparse(transformed):
        transformed_for_shap = transformed.toarray()
    else:
        transformed_for_shap = np.asarray(transformed)
    estimator = unwrap_tree_estimator(fitted_pipeline)
    explainer = shap.TreeExplainer(estimator)
    raw_values = explainer.shap_values(transformed_for_shap, check_additivity=False)
    values = standardize_shap_values(
        raw_values,
        n_samples=transformed_for_shap.shape[0],
        n_features=transformed_for_shap.shape[1],
        n_classes=len(CLASS_ORDER),
    )
    grouped_values, group_names = group_shap_values(values, encoded_groups)

    rows: list[dict[str, Any]] = []
    for group_idx, group in enumerate(group_names):
        for class_idx, class_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "feature": group,
                    "feature_label": FEATURE_LABELS.get(group, group),
                    "class": class_name,
                    "mean_absolute_shap": float(np.mean(np.abs(grouped_values[:, group_idx, class_idx]))),
                    "mean_signed_shap": float(np.mean(grouped_values[:, group_idx, class_idx])),
                    "sample_rows": len(sample_indices),
                }
            )
    importance = pd.DataFrame(rows).sort_values(
        ["class", "mean_absolute_shap"], ascending=[True, False]
    )
    importance.to_csv(
        output_dir / "shap" / "grouped_class_specific_shap_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # A global table averages the class-specific absolute effects.
    global_importance = (
        importance.groupby(["feature", "feature_label"], as_index=False)
        .agg(mean_absolute_shap=("mean_absolute_shap", "mean"))
        .sort_values("mean_absolute_shap", ascending=False)
    )
    global_importance.to_csv(
        output_dir / "shap" / "grouped_global_shap_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_grouped_shap_importance(
        importance,
        output_dir / "figures" / "shap_class_specific_importance.png",
    )
    plot_global_shap_importance(
        global_importance,
        output_dir / "figures" / "shap_global_importance.png",
    )

    interaction_summary: dict[str, Any] = {"computed": False}
    if config.run_interactions:
        interaction_summary = run_shap_interactions(
            explainer=explainer,
            fitted_pipeline=fitted_pipeline,
            X_test=X_test,
            y_test=y_test,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            config=config,
            output_dir=output_dir,
        )

    summary = {
        "shap_sample_rows": len(sample_indices),
        "shap_fatal_rows": int((y_sample == FATAL_CLASS).sum()),
        "encoded_feature_count": len(encoded_names),
        "grouped_original_feature_count": len(group_names),
        "top_global_features": global_importance.head(10).to_dict(orient="records"),
        "interactions": interaction_summary,
    }
    write_json(output_dir / "shap" / "shap_summary.json", summary)
    return summary


def plot_grouped_shap_importance(importance: pd.DataFrame, path: Path) -> None:
    top_features = (
        importance.groupby("feature_label")["mean_absolute_shap"]
        .mean()
        .nlargest(15)
        .index
    )
    plot_data = importance[importance["feature_label"].isin(top_features)].copy()
    pivot = plot_data.pivot(
        index="feature_label", columns="class", values="mean_absolute_shap"
    ).fillna(0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).drop(columns="total")
    ax = pivot.plot(kind="barh", stacked=True, figsize=(10, 7))
    ax.set_xlabel("Mean absolute grouped SHAP value")
    ax.set_ylabel("")
    ax.set_title("Class-specific grouped SHAP importance")
    ax.legend(title="Severity")
    save_figure(path)


def plot_global_shap_importance(global_importance: pd.DataFrame, path: Path) -> None:
    data = global_importance.head(15).sort_values("mean_absolute_shap")
    plt.figure(figsize=(9, 6.5))
    plt.barh(data["feature_label"], data["mean_absolute_shap"])
    plt.xlabel("Mean absolute grouped SHAP value")
    plt.title("Global grouped SHAP importance")
    save_figure(path)


def run_shap_interactions(
    explainer: shap.TreeExplainer,
    fitted_pipeline: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    config: RunConfig,
    output_dir: Path,
) -> dict[str, Any]:
    logging.info("Computing fatal-class SHAP interaction values.")
    indices = stratified_sample_indices(
        y_test,
        max_samples=min(config.interaction_max_samples, len(y_test)),
        seed=config.seed + 91000,
        include_all_class=FATAL_CLASS,
    )
    X_sample = X_test.iloc[indices].copy()
    y_sample = y_test[indices]
    transformed, _, encoded_groups = transformed_matrix_and_metadata(
        fitted_pipeline, X_sample, categorical_features, numeric_features
    )
    dense = transformed.toarray() if sparse.issparse(transformed) else np.asarray(transformed)
    raw = explainer.shap_interaction_values(dense)
    standardized = standardize_interaction_values(
        raw,
        n_samples=dense.shape[0],
        n_features=dense.shape[1],
        n_classes=len(CLASS_ORDER),
    )
    fatal_interactions = standardized[:, :, :, FATAL_CLASS]
    group_matrix, group_names = grouping_matrix(encoded_groups)

    # G.T @ encoded_interactions @ G for every sample.
    grouped = np.einsum("fi,sfj,jg->sig", group_matrix, fatal_interactions, group_matrix)
    mean_abs = np.mean(np.abs(grouped), axis=0)
    # SHAP interaction off-diagonal values are symmetric and represent half the
    # pair interaction in standard SHAP conventions. Ranking is unaffected.
    records = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            records.append(
                {
                    "feature_1": group_names[i],
                    "feature_1_label": FEATURE_LABELS.get(group_names[i], group_names[i]),
                    "feature_2": group_names[j],
                    "feature_2_label": FEATURE_LABELS.get(group_names[j], group_names[j]),
                    "mean_absolute_fatal_interaction": float(mean_abs[i, j]),
                    "mean_signed_fatal_interaction": float(np.mean(grouped[:, i, j])),
                    "sample_rows": len(indices),
                    "fatal_rows": int((y_sample == FATAL_CLASS).sum()),
                }
            )
    ranking = pd.DataFrame(records).sort_values(
        "mean_absolute_fatal_interaction", ascending=False
    )
    ranking.to_csv(
        output_dir / "shap" / "fatal_shap_interaction_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_interaction_heatmap(
        mean_abs,
        group_names,
        output_dir / "figures" / "fatal_shap_interaction_heatmap.png",
    )

    top_pairs = ranking.head(6)
    pair_plot_files = []
    for rank, row in enumerate(top_pairs.itertuples(index=False), start=1):
        file_path = output_dir / "figures" / f"fatal_interaction_{rank:02d}_{safe_filename(row.feature_1)}__{safe_filename(row.feature_2)}.png"
        plot_raw_pair_interaction(
            X_sample,
            grouped[:, group_names.index(row.feature_1), group_names.index(row.feature_2)],
            row.feature_1,
            row.feature_2,
            file_path,
        )
        pair_plot_files.append(str(file_path))

    return {
        "computed": True,
        "sample_rows": len(indices),
        "fatal_rows": int((y_sample == FATAL_CLASS).sum()),
        "top_interactions": ranking.head(10).to_dict(orient="records"),
        "pair_plot_files": pair_plot_files,
    }


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "feature"


def plot_interaction_heatmap(mean_abs: np.ndarray, groups: Sequence[str], path: Path) -> None:
    offdiag_strength = mean_abs.copy()
    np.fill_diagonal(offdiag_strength, 0.0)
    top_idx = np.argsort(offdiag_strength.sum(axis=1))[-15:]
    data = offdiag_strength[np.ix_(top_idx, top_idx)]
    labels = [FEATURE_LABELS.get(groups[i], groups[i]) for i in top_idx]
    plt.figure(figsize=(11, 9))
    image = plt.imshow(data, aspect="auto")
    plt.colorbar(image, label="Mean absolute fatal-class SHAP interaction")
    plt.xticks(np.arange(len(labels)), labels, rotation=75, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("Fatal-class grouped SHAP interaction heatmap")
    save_figure(path)


def plot_raw_pair_interaction(
    X_sample: pd.DataFrame,
    interaction_values: np.ndarray,
    feature_1: str,
    feature_2: str,
    path: Path,
) -> None:
    label_1 = FEATURE_LABELS.get(feature_1, feature_1)
    label_2 = FEATURE_LABELS.get(feature_2, feature_2)
    first = X_sample[feature_1]
    second = X_sample[feature_2]
    first_numeric = pd.api.types.is_numeric_dtype(first)
    second_numeric = pd.api.types.is_numeric_dtype(second)

    if not first_numeric and not second_numeric:
        table = pd.DataFrame(
            {
                "first": first.fillna("Missing/Unreported").astype(str),
                "second": second.fillna("Missing/Unreported").astype(str),
                "interaction": interaction_values,
            }
        )
        top_first = table["first"].value_counts().head(10).index
        top_second = table["second"].value_counts().head(10).index
        table = table[table["first"].isin(top_first) & table["second"].isin(top_second)]
        pivot = table.pivot_table(
            index="first", columns="second", values="interaction", aggfunc="mean"
        )
        plt.figure(figsize=(10, max(5, 0.5 * len(pivot))))
        image = plt.imshow(pivot.fillna(0).to_numpy(), aspect="auto")
        plt.colorbar(image, label="Mean fatal-class interaction contribution")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=75, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel(label_2)
        plt.ylabel(label_1)
        plt.title(f"Fatal-class interaction: {label_1} × {label_2}")
        save_figure(path)
        return

    # For mixed or numeric pairs, display the interaction contribution against
    # the first feature and use the second feature as a categorical or numeric color.
    plt.figure(figsize=(8, 5.5))
    if first_numeric:
        x_values = pd.to_numeric(first, errors="coerce")
        x_label = label_1
        color_source = second
        color_label = label_2
    else:
        x_values = pd.to_numeric(second, errors="coerce")
        x_label = label_2
        color_source = first
        color_label = label_1

    if pd.api.types.is_numeric_dtype(color_source):
        color_values = pd.to_numeric(color_source, errors="coerce")
        scatter = plt.scatter(x_values, interaction_values, c=color_values, alpha=0.65)
        plt.colorbar(scatter, label=color_label)
    else:
        categories = color_source.fillna("Missing/Unreported").astype(str)
        top = categories.value_counts().head(8).index
        for category in top:
            mask = categories == category
            plt.scatter(x_values[mask], interaction_values[mask], alpha=0.6, label=category)
        plt.legend(title=color_label, fontsize=8)
    plt.axhline(0, linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel("Fatal-class grouped SHAP interaction contribution")
    plt.title(f"Fatal-class interaction: {label_1} × {label_2}")
    save_figure(path)


# -----------------------------------------------------------------------------
# Publication-ready output tables
# -----------------------------------------------------------------------------


def write_paper_ready_tables(
    output_dir: Path,
    split_summary: pd.DataFrame,
    stability: pd.DataFrame,
    cv_summary: pd.DataFrame,
    test_metrics: Mapping[str, Any],
    bootstrap_summary: pd.DataFrame,
    screening: Mapping[str, Any],
    smotenc_summary: Optional[pd.DataFrame],
    spatial_results: Optional[pd.DataFrame],
    full_feature_summary: Optional[Mapping[str, Any]],
) -> None:
    """Create compact CSV tables corresponding to the main reported analyses."""
    paper_dir = output_dir / "paper"
    ensure_dir(paper_dir)

    split_summary.to_csv(
        paper_dir / "Table_1_cohort_and_temporal_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stability_columns = [
        "feature_label",
        "type",
        "year_96_missing_percent",
        "year_97_missing_percent",
        "year_98_missing_percent",
        "absolute_dev_test_missing_gap_pp",
        "decision",
        "rationale",
    ]
    stability[stability_columns].to_csv(
        paper_dir / "Table_2_predictor_availability_and_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cv_columns = [
        "model",
        "outer_folds",
        "macro_f1_mean",
        "macro_f1_sd",
        "balanced_accuracy_mean",
        "balanced_accuracy_sd",
        "fatal_f1_mean",
        "fatal_f1_sd",
        "log_loss_mean",
        "log_loss_sd",
        "macro_roc_auc_ovr_mean",
        "macro_roc_auc_ovr_sd",
    ]
    cv_summary[cv_columns].to_csv(
        paper_dir / "Table_3_nested_cv_model_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ci_lookup = bootstrap_summary.set_index("metric").to_dict(orient="index")

    def metric_row(
        section: str,
        metric: str,
        estimate: float,
        bootstrap_metric: Optional[str] = None,
    ) -> dict[str, Any]:
        ci = ci_lookup.get(bootstrap_metric or metric, {})
        return {
            "section": section,
            "metric": metric,
            "estimate": estimate,
            "ci_2.5_percent": ci.get("ci_2.5_percent", np.nan),
            "ci_97.5_percent": ci.get("ci_97.5_percent", np.nan),
        }

    temporal_rows = [
        metric_row(
            "Locked uncalibrated hard classification",
            "Macro F1",
            test_metrics["uncalibrated_macro_f1"],
            "classification_macro_f1",
        ),
        metric_row(
            "Locked uncalibrated hard classification",
            "Balanced accuracy",
            test_metrics["uncalibrated_balanced_accuracy"],
            "classification_balanced_accuracy",
        ),
        metric_row(
            "Locked uncalibrated hard classification",
            "Accuracy",
            test_metrics["uncalibrated_accuracy"],
            "classification_accuracy",
        ),
        metric_row(
            "Locked uncalibrated hard classification",
            "Fatal-class F1",
            test_metrics["uncalibrated_fatal_f1"],
            "classification_fatal_f1",
        ),
        metric_row(
            "Development-fitted calibrated probabilities",
            "Log loss",
            test_metrics["calibrated_log_loss"],
            "calibrated_log_loss",
        ),
        metric_row(
            "Development-fitted calibrated probabilities",
            "Multiclass Brier score",
            test_metrics["calibrated_multiclass_brier"],
            "calibrated_multiclass_brier",
        ),
        metric_row(
            "Development-fitted calibrated probabilities",
            "Macro ECE",
            test_metrics["calibrated_macro_ece"],
            "calibrated_macro_ece",
        ),
        metric_row(
            "Development-selected fatal screening",
            "Fatal precision",
            screening["fatal_precision"],
            "fatal_screen_precision",
        ),
        metric_row(
            "Development-selected fatal screening",
            "Fatal recall",
            screening["fatal_recall"],
            "fatal_screen_recall",
        ),
        metric_row(
            "Development-selected fatal screening",
            "Fatal F2",
            screening["fatal_f2"],
            "fatal_screen_f2",
        ),
    ]
    pd.DataFrame(temporal_rows).to_csv(
        paper_dir / "Table_4_temporal_test_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity_rows: list[dict[str, Any]] = []
    if smotenc_summary is not None:
        for _, row in smotenc_summary.iterrows():
            sensitivity_rows.append(
                {
                    "analysis": row["analysis"],
                    "evaluation_scope": "Repeated development CV",
                    "macro_f1": row["macro_f1_mean"],
                    "fatal_f1": row["fatal_f1_mean"],
                    "notes": "Mean across fixed outer folds",
                }            )
    if spatial_results is not None and not spatial_results.empty:
        sensitivity_rows.append(
            {
                "analysis": "5-km stratified spatial-group sensitivity",
                "evaluation_scope": "Development grouped CV",
                "macro_f1": float(spatial_results["macro_f1"].mean()),
                "fatal_f1": float(spatial_results["fatal_f1"].mean()),
                "notes": f"{len(spatial_results)} grouped folds",
            }
        )
    if full_feature_summary is not None:
        full_metrics = full_feature_summary["test_metrics"]
        sensitivity_rows.append(
            {
                "analysis": "Full-feature model including unstable fields",
                "evaluation_scope": "Untouched temporal test",
                "macro_f1": full_metrics["macro_f1"],
                "fatal_f1": full_metrics["fatal_f1"],
                "notes": "Secondary sensitivity; not used for SHAP or primary claims",
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        paper_dir / "Table_5_sensitivity_analyses.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_shap = output_dir / "shap" / "grouped_global_shap_importance.csv"
    class_shap = output_dir / "shap" / "grouped_class_specific_shap_importance.csv"
    interactions = output_dir / "shap" / "fatal_shap_interaction_ranking.csv"
    if global_shap.exists():
        shutil.copy2(global_shap, paper_dir / "Table_6a_global_SHAP_importance.csv")
    if class_shap.exists():
        shutil.copy2(class_shap, paper_dir / "Table_6b_class_specific_SHAP_importance.csv")
    if interactions.exists():
        interaction_frame = pd.read_csv(interactions).head(20)
        interaction_frame.to_csv(
            paper_dir / "Table_7_top_fatal_SHAP_interactions.csv",
            index=False,
            encoding="utf-8-sig",
        )


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled motorcycle crash-severity analysis."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to the Excel workbook. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/main_analysis.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
        help="smoke verifies the workflow quickly; full runs all analyses.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for RandomizedSearchCV; model-internal jobs remain 1.",
    )
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-interactions", action="store_true")
    parser.add_argument("--skip-smotenc", action="store_true")
    parser.add_argument("--skip-spatial", action="store_true")
    parser.add_argument("--skip-full-feature-sensitivity", action="store_true")
    return parser.parse_args()


def initialize_logging(output_dir: Path) -> None:
    log_path = output_dir / "run.log"
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def environment_manifest(
    dataset_path: Path,
    script_path: Path,
    output_dir: Path,
    config: RunConfig,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_started": datetime.now().isoformat(),
        "dataset_file": dataset_path.name,
        "dataset_sha256": sha256_file(dataset_path),
        "script_file": script_path.name,
        "script_sha256": sha256_file(script_path),
        "output_directory": output_dir.name,
        "config": asdict(config),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "shap": shap.__version__,
            "joblib": joblib.__version__,
        },
    }


def main() -> int:
    args = parse_args()
    dataset_path = args.data.expanduser()
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    output_dir = args.output or Path("outputs") / "main_analysis"
    output_dir = ensure_dir(output_dir)
    for subdir in [
        "audit",
        "modeling",
        "evaluation",
        "figures",
        "models",
        "shap",
        "sensitivity",
        "paper",
    ]:
        ensure_dir(output_dir / subdir)
    initialize_logging(output_dir)

    config = config_for_mode(args.mode, args.seed, args.n_jobs)
    config = RunConfig(
        **{
            **asdict(config),
            "run_shap": config.run_shap and not args.skip_shap,
            "run_interactions": config.run_interactions and not args.skip_interactions and not args.skip_shap,
            "run_smotenc_sensitivity": config.run_smotenc_sensitivity and not args.skip_smotenc,
            "run_spatial_sensitivity": config.run_spatial_sensitivity and not args.skip_spatial,
            "run_full_feature_sensitivity": config.run_full_feature_sensitivity and not args.skip_full_feature_sensitivity,
        }
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    script_path = Path(__file__)
    manifest = environment_manifest(dataset_path, script_path, output_dir, config)
    write_json(output_dir / "00_run_manifest.json", manifest)

    try:
        raw, guide = load_workbook(dataset_path)
        workbook_codebook_to_long(guide).to_csv(
            output_dir / "audit" / "workbook_codebook_long.csv",
            index=False,
            encoding="utf-8-sig",
        )
        build_feature_governance(raw.columns).to_csv(
            output_dir / "audit" / "feature_governance.csv",
            index=False,
            encoding="utf-8-sig",
        )

        cohort, categorical_features, numeric_features, audit_summary = prepare_motorcycle_cohort(
            raw, output_dir, config
        )
        logging.info(
            "Motorcycle cohort: %d rows; actual severity counts=%s",
            len(cohort),
            audit_summary["actual_severity_counts"],
        )
        if len(cohort) != EXPECTED_COHORT_SIZE:
            logging.warning(
                "Cohort size differs from expected %d. Continue only after reviewing audit_summary.json.",
                EXPECTED_COHORT_SIZE,
            )

        full_categorical_features = list(categorical_features)
        full_numeric_features = list(numeric_features)
        primary_categorical_features, primary_numeric_features, stability = (
            audit_and_select_stable_features(
                cohort,
                full_categorical_features,
                full_numeric_features,
                output_dir,
            )
        )
        plot_feature_stability(
            stability,
            output_dir / "figures" / "feature_availability_by_year.png",
        )
        audit_summary = {
            **audit_summary,
            "protocol_version": PROTOCOL_VERSION,
            "stable_feature_rule": {
                "maximum_annual_missing_percent": MAX_ANNUAL_MISSING_PERCENT,
                "maximum_development_test_missingness_gap_percentage_points": MAX_DEV_TEST_MISSING_GAP_PERCENT,
            },
            "primary_categorical_features": primary_categorical_features,
            "primary_numeric_features": primary_numeric_features,
            "primary_feature_count": len(primary_categorical_features)
            + len(primary_numeric_features),
            "full_feature_candidate_count": len(full_categorical_features)
            + len(full_numeric_features),
        }
        write_json(output_dir / "audit" / "audit_summary.json", audit_summary)

        dev_mask = cohort["analysis_year"].isin(DEVELOPMENT_YEARS)
        test_mask = cohort["analysis_year"] == TEMPORAL_TEST_YEAR
        excluded_years = ~(dev_mask | test_mask)
        if excluded_years.any():
            logging.warning(
                "Excluding %d records outside years %s and %s.",
                int(excluded_years.sum()),
                DEVELOPMENT_YEARS,
                TEMPORAL_TEST_YEAR,
            )

        primary_model_features = [
            *primary_categorical_features,
            *primary_numeric_features,
        ]
        full_model_features = [*full_categorical_features, *full_numeric_features]
        X_dev = cohort.loc[dev_mask, primary_model_features].reset_index(drop=True)
        X_test = cohort.loc[test_mask, primary_model_features].reset_index(drop=True)
        X_dev_full = cohort.loc[dev_mask, full_model_features].reset_index(drop=True)
        X_test_full = cohort.loc[test_mask, full_model_features].reset_index(drop=True)
        y_dev = cohort.loc[dev_mask, "target"].to_numpy(dtype=int)
        y_test = cohort.loc[test_mask, "target"].to_numpy(dtype=int)
        groups_dev = cohort.loc[dev_mask, "spatial_group_5km"].to_numpy()

        split_summary = pd.DataFrame(
            [
                {
                    "partition": "Development (1396-1397)",
                    "rows": len(X_dev),
                    **{
                        f"{name.lower()}_count": int(np.sum(y_dev == idx))
                        for idx, name in enumerate(CLASS_NAMES)
                    },
                },
                {
                    "partition": "Temporal test (1398)",
                    "rows": len(X_test),
                    **{
                        f"{name.lower()}_count": int(np.sum(y_test == idx))
                        for idx, name in enumerate(CLASS_NAMES)
                    },
                },
            ]
        )
        split_summary.to_csv(
            output_dir / "audit" / "development_temporal_split.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_fold_frames = []
        all_prediction_frames: dict[str, pd.DataFrame] = {}
        for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            folds, predictions = nested_cv_model(
                model_name,
                X_dev,
                y_dev,
                primary_categorical_features,
                primary_numeric_features,
                config,
                output_dir,
            )
            all_fold_frames.append(folds)
            all_prediction_frames[model_name] = predictions

        all_folds = pd.concat(all_fold_frames, ignore_index=True)
        all_folds.to_csv(
            output_dir / "modeling" / "nested_cv_all_models_folds.csv",
            index=False,
            encoding="utf-8-sig",
        )
        cv_summary = summarize_nested_cv(all_folds)
        cv_summary.to_csv(
            output_dir / "modeling" / "nested_cv_model_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_model_cv_comparison(
            cv_summary, output_dir / "figures" / "nested_cv_model_comparison.png"
        )
        model_selection_summary = paired_model_comparison(
            all_folds, cv_summary, output_dir
        )
        selected_model = str(model_selection_summary["selected_model"])
        logging.info(
            "Development-only selected model family: %s. %s",
            selected_model,
            model_selection_summary["interpretation"],
        )

        selected_oof_y, selected_oof_probs = aggregate_repeated_oof(
            all_prediction_frames[selected_model]
        )
        if not np.array_equal(selected_oof_y, y_dev):
            raise RuntimeError(
                "Aggregated nested OOF rows do not align with development outcomes."
            )

        logistic_calibrator = MulticlassLogitCalibrator(
            random_state=config.seed
        ).fit(selected_oof_probs, y_dev)
        isotonic_calibrator = OneVsRestIsotonicCalibrator().fit(
            selected_oof_probs, y_dev
        )
        oof_logistic = logistic_calibrator.predict_proba(selected_oof_probs)
        oof_isotonic = isotonic_calibrator.predict_proba(selected_oof_probs)
        oof_calibration_metrics = {
            **evaluate_probabilities(
                y_dev, selected_oof_probs, prefix="uncalibrated_oof"
            ),
            **evaluate_probabilities(
                y_dev, oof_logistic, prefix="logistic_calibrated_oof"
            ),
            **evaluate_probabilities(
                y_dev, oof_isotonic, prefix="isotonic_calibrated_oof"
            ),
        }
        write_json(
            output_dir / "evaluation" / "development_oof_calibration_metrics.json",
            oof_calibration_metrics,
        )

        fatal_threshold, threshold_table = select_fatal_threshold(
            y_dev, oof_logistic[:, FATAL_CLASS], beta=2.0
        )
        threshold_table.to_csv(
            output_dir / "evaluation" / "development_fatal_threshold_grid.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logging.info(
            "Development-selected fatal screening threshold: %.3f", fatal_threshold
        )

        final_search = fit_final_model_search(
            selected_model,
            X_dev,
            y_dev,
            primary_categorical_features,
            primary_numeric_features,
            config,
        )
        final_pipeline = final_search.best_estimator_
        best_params = final_search.best_params_
        joblib.dump(
            final_pipeline,
            output_dir / "models" / "final_stable_feature_uncalibrated_pipeline.joblib",
        )
        joblib.dump(
            logistic_calibrator,
            output_dir / "models" / "multiclass_logit_calibrator.joblib",
        )
        joblib.dump(
            isotonic_calibrator,
            output_dir / "models" / "ovr_isotonic_calibrator.joblib",
        )
        write_json(
            output_dir / "models" / "final_model_selection.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "selected_model": selected_model,
                "model_selection_interpretation": model_selection_summary,
                "final_inner_cv_macro_f1": float(final_search.best_score_),
                "best_params": best_params,
                "fatal_threshold": fatal_threshold,
                "primary_categorical_features": primary_categorical_features,
                "primary_numeric_features": primary_numeric_features,
            },
        )

        # First and only primary evaluation on the untouched temporal test.
        base_test_probs = final_pipeline.predict_proba(X_test)
        logistic_test_probs = logistic_calibrator.predict_proba(base_test_probs)
        isotonic_test_probs = isotonic_calibrator.predict_proba(base_test_probs)
        test_metrics = {
            **evaluate_probabilities(
                y_test, base_test_probs, prefix="uncalibrated"
            ),
            **evaluate_probabilities(
                y_test, logistic_test_probs, prefix="calibrated"
            ),
            **evaluate_probabilities(
                y_test,
                isotonic_test_probs,
                prefix="isotonic_sensitivity",
            ),
        }
        screening = fatal_screening_metrics(
            y_test,
            logistic_test_probs[:, FATAL_CLASS],
            fatal_threshold,
        )
        write_json(
            output_dir / "evaluation" / "temporal_test_metrics.json",
            test_metrics,
        )
        write_json(
            output_dir / "evaluation" / "fatal_screening_metrics.json",
            screening,
        )

        base_predictions = base_test_probs.argmax(axis=1)
        calibrated_predictions = logistic_test_probs.argmax(axis=1)
        predictions_frame = pd.DataFrame(
            {
                "y_true": y_test,
                "severity_true": [INTERNAL_TO_NAME[i] for i in y_test],
                "base_p_pdo": base_test_probs[:, 0],
                "base_p_injury": base_test_probs[:, 1],
                "base_p_fatal": base_test_probs[:, 2],
                "base_predicted_class": base_predictions,
                "base_predicted_severity": [
                    INTERNAL_TO_NAME[i] for i in base_predictions
                ],
                "calibrated_p_pdo": logistic_test_probs[:, 0],
                "calibrated_p_injury": logistic_test_probs[:, 1],
                "calibrated_p_fatal": logistic_test_probs[:, 2],
                "calibrated_argmax_class": calibrated_predictions,
                "calibrated_argmax_severity": [
                    INTERNAL_TO_NAME[i] for i in calibrated_predictions
                ],
                "fatal_screen_flag": (
                    logistic_test_probs[:, FATAL_CLASS] >= fatal_threshold
                ).astype(int),
            }
        )
        predictions_frame.to_csv(
            output_dir / "evaluation" / "temporal_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        bootstrap = bootstrap_temporal_metrics(
            y_test,
            classification_probabilities=base_test_probs,
            calibrated_probabilities=logistic_test_probs,
            fatal_threshold=fatal_threshold,
            iterations=config.bootstrap_iterations,
            seed=config.seed + 60000,
        )
        bootstrap.to_csv(
            output_dir / "evaluation" / "temporal_test_bootstrap_iterations.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bootstrap_summary = summarize_bootstrap(bootstrap)
        bootstrap_summary.to_csv(
            output_dir / "evaluation" / "temporal_test_bootstrap_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        save_confusion_plot(
            y_test,
            base_test_probs,
            output_dir
            / "figures"
            / "temporal_test_confusion_matrix_counts.png",
            "Temporal-test confusion matrix: locked stable-feature classifier",
        )
        save_confusion_plot(
            y_test,
            base_test_probs,
            output_dir
            / "figures"
            / "temporal_test_confusion_matrix_normalized.png",
            "Temporal-test confusion matrix: row-normalized",
            normalize="true",
        )
        plot_calibration_curves(
            y_test,
            {
                "Uncalibrated": base_test_probs,
                "Logistic calibration": logistic_test_probs,
                "Isotonic sensitivity": isotonic_test_probs,
            },
            output_dir / "figures" / "temporal_test_calibration.png",
        )
        plot_roc_pr_curves(
            y_test,
            base_test_probs,
            output_dir / "figures" / "temporal_test_roc_curves.png",
            output_dir
            / "figures"
            / "temporal_test_precision_recall_curves.png",
        )

        smotenc_summary = None
        if config.run_smotenc_sensitivity:
            smotenc_summary = run_smotenc_sensitivity(
                selected_model,
                best_params,
                X_dev,
                y_dev,
                primary_categorical_features,
                primary_numeric_features,
                config,
                output_dir,
            )

        spatial_results = None
        if config.run_spatial_sensitivity:
            selected_fixed = set_model_parameters(
                build_primary_pipeline(
                    selected_model,
                    primary_categorical_features,
                    primary_numeric_features,
                    config,
                    config.seed + 81000,
                ),
                best_params,
            )
            spatial_results = run_spatial_sensitivity(
                selected_fixed,
                X_dev,
                y_dev,
                groups_dev,
                config,
                output_dir,
            )

        full_feature_summary = None
        if config.run_full_feature_sensitivity:
            full_feature_summary = run_full_feature_sensitivity(
                selected_model,
                X_dev_full,
                y_dev,
                X_test_full,
                y_test,
                full_categorical_features,
                full_numeric_features,
                config,
                output_dir,
            )
            comparison = pd.DataFrame(
                [
                    {
                        "analysis": "Stable-feature primary",
                        "feature_count": len(primary_model_features),
                        "macro_f1": test_metrics["uncalibrated_macro_f1"],
                        "fatal_f1": test_metrics["uncalibrated_fatal_f1"],
                        "balanced_accuracy": test_metrics[
                            "uncalibrated_balanced_accuracy"
                        ],
                        "log_loss": test_metrics["uncalibrated_log_loss"],
                    },
                    {
                        "analysis": "Full-feature sensitivity",
                        "feature_count": full_feature_summary["feature_count"],
                        "macro_f1": full_feature_summary["test_metrics"][
                            "macro_f1"
                        ],
                        "fatal_f1": full_feature_summary["test_metrics"][
                            "fatal_f1"
                        ],
                        "balanced_accuracy": full_feature_summary[
                            "test_metrics"
                        ]["balanced_accuracy"],
                        "log_loss": full_feature_summary["test_metrics"][
                            "log_loss"
                        ],
                    },
                ]
            )
            comparison.to_csv(
                output_dir
                / "sensitivity"
                / "stable_vs_full_feature_temporal_comparison.csv",
                index=False,
                encoding="utf-8-sig",
            )

        shap_summary = None
        if config.run_shap:
            shap_summary = run_shap_analysis(
                final_pipeline,
                X_test,
                y_test,
                primary_categorical_features,
                primary_numeric_features,
                config,
                output_dir,
            )

        write_paper_ready_tables(
            output_dir,
            split_summary,
            stability,
            cv_summary,
            test_metrics,
            bootstrap_summary,
            screening,
            smotenc_summary,
            spatial_results,
            full_feature_summary,
        )

        final_summary = {
            "status": "complete",
            "protocol_version": PROTOCOL_VERSION,
            "selected_model": selected_model,
            "model_selection_interpretation": model_selection_summary,
            "cohort_rows": len(cohort),
            "development_rows": len(X_dev),
            "temporal_test_rows": len(X_test),
            "primary_feature_count": len(primary_model_features),
            "primary_categorical_features": primary_categorical_features,
            "primary_numeric_features": primary_numeric_features,
            "excluded_unstable_features": stability.loc[
                ~stability["primary_eligible"],
                ["feature", "feature_label", "rationale"],
            ].to_dict(orient="records"),
            "test_metrics": test_metrics,
            "fatal_screening": screening,
            "full_feature_sensitivity": full_feature_summary,
            "smotenc_sensitivity": None
            if smotenc_summary is None
            else smotenc_summary.to_dict(orient="records"),
            "spatial_sensitivity": None
            if spatial_results is None
            else spatial_results.to_dict(orient="records"),
            "shap_summary": shap_summary,
            "run_finished": datetime.now().isoformat(),
        }
        write_json(output_dir / "FINAL_SUMMARY.json", final_summary)
        (output_dir / "RUN_COMPLETE.txt").write_text(
            "Final stable-feature analysis completed successfully.\n"
            f"Protocol: {PROTOCOL_VERSION}\n"
            f"Selected model: {selected_model}\n"
            f"Primary predictors: {len(primary_model_features)}\n"
            f"Temporal-test hard-classification macro F1: {test_metrics['uncalibrated_macro_f1']:.4f}\n"
            f"Temporal-test hard-classification fatal F1: {test_metrics['uncalibrated_fatal_f1']:.4f}\n"
            f"Calibrated temporal-test log loss: {test_metrics['calibrated_log_loss']:.4f}\n"
            f"Fatal-screening threshold: {fatal_threshold:.4f}\n",
            encoding="utf-8",
        )
        logging.info("Analysis completed successfully. Output: %s", output_dir)
        return 0

    except Exception as exc:
        logging.exception("Analysis failed: %s", exc)
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": datetime.now().isoformat(),
        }
        write_json(output_dir / "RUN_FAILED.json", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())