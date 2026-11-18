"""Evaluation helpers.

Two levels are reported:

* candidate level  - binary "is this the right arc" classification
                     (accuracy / precision / recall / macro-F1 / log loss)
* fix / route level - pick the top-scoring candidate of each fix and compare
                     with the aligned reference arc (top-1 accuracy), plus
                     how connected / complete the resulting route is.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def candidate_metrics(y_true, proba, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_pos": precision_score(y_true, y_pred, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def top1_choice(cand: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Highest scoring candidate per (trip, fix). Ties go to the closer arc (rows are distance-sorted)."""
    tmp = cand[["trip_id", "obs_idx", "arc_id", "arc_src", "arc_dst"]].copy()
    tmp["score"] = score
    idx = tmp.groupby(["trip_id", "obs_idx"], sort=True)["score"].idxmax()
    best = tmp.loc[idx.to_numpy()].rename(columns={"arc_id": "pred_arc"})
    return best.reset_index(drop=True)


def top1_accuracy(fixes: pd.DataFrame, best: pd.DataFrame) -> dict:
    """Top-1 accuracy over fixes with a reliable label.

    A fix whose true arc is not among its candidates counts as wrong, so this
    number already includes the candidate-generation losses.
    """
    m = fixes.loc[fixes["label_ok"], ["trip_id", "obs_idx", "label_arc", "label_in_cands"]]
    m = m.merge(best[["trip_id", "obs_idx", "pred_arc"]], on=["trip_id", "obs_idx"], how="left")
    correct = (m["pred_arc"] == m["label_arc"]).to_numpy()
    covered = m["label_in_cands"].to_numpy()
    return {
        "top1_acc": float(correct.mean()),
        "top1_acc_when_covered": float(correct[covered].mean()) if covered.any() else float("nan"),
        "candidate_miss_rate": float(1.0 - covered.mean()),
        "n_scored_fixes": int(len(m)),
    }


def _collapse(seq):
    seq = np.asarray(seq)
    if seq.size == 0:
        return seq
    keep = np.r_[True, seq[1:] != seq[:-1]]
    return seq[keep]


def route_continuity(best_one_trip: pd.DataFrame) -> float:
    """Share of arc changes (after merging repeats) where the next arc starts where the previous ends."""
    b = best_one_trip.sort_values("obs_idx")
    arcs = b["pred_arc"].to_numpy()
    keep = np.r_[True, arcs[1:] != arcs[:-1]]
    src = b["arc_src"].to_numpy()[keep]
    dst = b["arc_dst"].to_numpy()[keep]
    if src.size < 2:
        return 1.0
    return float(np.mean(dst[:-1] == src[1:]))


def route_overlap(pred_arcs, ref_route) -> dict:
    """Set overlap between the arcs the prediction visits and the reference route arcs."""
    p = set(_collapse(pred_arcs).tolist())
    r = set(np.asarray(ref_route).tolist())
    inter = len(p & r)
    prec = inter / len(p) if p else 0.0
    rec = inter / len(r) if r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def route_level_summary(best: pd.DataFrame, routes: dict) -> dict:
    """Average continuity / route-F1 over trips. `routes` maps trip_id -> reference arc array."""
    rows = []
    for tid, g in best.groupby("trip_id"):
        g = g.sort_values("obs_idx")
        row = {"trip_id": tid, "continuity": route_continuity(g)}
        if tid in routes:
            row.update(route_overlap(g["pred_arc"].to_numpy(), routes[tid]))
        rows.append(row)
    per_trip = pd.DataFrame(rows)
    out = {"route_continuity": float(per_trip["continuity"].mean())}
    if "route_f1" in per_trip:
        out["route_f1"] = float(per_trip["route_f1"].mean())
    return out, per_trip
