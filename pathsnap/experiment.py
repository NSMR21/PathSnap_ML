"""Shared helpers for the experiment scripts (03 model comparison, 05 tuning, 06 HMM).

Keeping data loading and scoring in one place guarantees that the untuned
models, the tuned model and the HMM are all measured in exactly the same way.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config
from .features import FEATURE_COLUMNS
from .metrics import candidate_metrics, route_level_summary, top1_accuracy, top1_choice


def load_partition(part):
    cand = pd.read_pickle(config.FEATURE_DIR / f"{part}_candidates.pkl")
    fixes = pd.read_pickle(config.FEATURE_DIR / f"{part}_fixes.pkl")
    routes = {k: np.asarray(v) for k, v in json.loads((config.FEATURE_DIR / f"{part}_routes.json").read_text()).items()}
    return cand, fixes, routes


def training_rows(cand: pd.DataFrame, stride: int = 2) -> pd.DataFrame:
    """Labelled train rows, every `stride`-th fix (see 03 for the reasoning)."""
    return cand[cand["label_ok"] & (cand["obs_idx"] % stride == 0)].reset_index(drop=True)


def xy(rows: pd.DataFrame):
    return rows[FEATURE_COLUMNS].to_numpy(np.float32), rows["is_match"].to_numpy().astype(int)


def group_folds(rows: pd.DataFrame, n_splits: int = 5):
    """The same trip-grouped folds everywhere (GroupKFold is deterministic)."""
    X, y = xy(rows)
    return list(GroupKFold(n_splits=n_splits).split(X, y, rows["trip_id"].to_numpy()))


def grouped_top1(rows: pd.DataFrame, score) -> float:
    """Top-1 accuracy over the fixes present in `rows` (fix without the true arc counts as wrong)."""
    tmp = pd.DataFrame({"trip": rows["trip_id"].to_numpy(), "obs": rows["obs_idx"].to_numpy(),
                        "s": np.asarray(score), "m": rows["is_match"].to_numpy()})
    idx = tmp.groupby(["trip", "obs"], sort=False)["s"].idxmax()
    return float(tmp.loc[idx.to_numpy(), "m"].mean())


def per_trip_top1(fixes, best):
    m = fixes.loc[fixes["label_ok"], ["trip_id", "obs_idx", "label_arc"]].merge(
        best[["trip_id", "obs_idx", "pred_arc"]], on=["trip_id", "obs_idx"])
    m["ok"] = m["pred_arc"] == m["label_arc"]
    return m.groupby("trip_id")["ok"].mean()


def evaluate_student_test(est, te_cand, te_fix, te_routes) -> dict:
    """All student_test metrics for a fitted candidate classifier (classifier-only decoding)."""
    X = te_cand[FEATURE_COLUMNS].to_numpy(np.float32)
    ok = te_cand["label_ok"].to_numpy()
    y = te_cand["is_match"].to_numpy().astype(int)
    t0 = time.perf_counter()
    p = est.predict_proba(X)[:, 1]
    infer_s = time.perf_counter() - t0
    cm = candidate_metrics(y[ok], p[ok])
    best = top1_choice(te_cand, p)
    t1 = top1_accuracy(te_fix, best)
    route_sum, _ = route_level_summary(best, te_routes)
    return {
        "test_accuracy": cm["accuracy"], "test_precision_macro": cm["precision_macro"],
        "test_recall_macro": cm["recall_macro"], "test_f1_macro": cm["f1_macro"], "test_log_loss": cm["log_loss"],
        "test_top1": t1["top1_acc"], "test_candidate_miss": t1["candidate_miss_rate"],
        **{f"test_{k}": v for k, v in route_sum.items()},
        "infer_ms_per_1k_fixes": infer_s * 1000 / (len(te_fix) / 1000),
        "confusion": cm["confusion"], "proba": p,
    }
