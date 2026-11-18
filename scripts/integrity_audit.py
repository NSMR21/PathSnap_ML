"""Academic-integrity self-audit (re-runnable before every submission).

Checks the things the PathSnap policy says the organisers will look at that
we CAN verify ourselves:

  1. partitions_disjoint        no trip appears in two partitions (by id or by file content)
  2. features_route_free        features are identical with and without the reference route
                                -> the ground truth cannot leak into the model input
  3. features_reproducible      rebuilding a trip gives exactly the saved feature rows
  4. reported_metrics_match     saved models re-predict student_test and reproduce every
                                number in outputs/results/model_comparison.csv
  5. refit_deterministic        refitting a model from scratch gives the same predictions
  6. no_evaluator_data          code never references an evaluator-test partition
  7. no_secrets_or_personal     no keys / passwords / e-mail addresses in code or docs
  8. hardcoded_trip_ids         every literal trip id in the code is listed for manual review
                                (allowed only for choosing which trip to plot)

What it can NOT check: similarity with other teams' submissions (we have no
access to them) and AI-content detection of the written report.

Run:  python scripts/integrity_audit.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.features import FEATURE_COLUMNS, build_trip_tables
from pathsnap.metrics import candidate_metrics, top1_accuracy, top1_choice
from pathsnap.models import MODEL_ZOO
from pathsnap.trip_io import list_trips, load_trip, trip_files

ROOT = config.ROOT
CODE_FILES = sorted([*ROOT.glob("pathsnap/*.py"), *ROOT.glob("scripts/*.py")])
DOC_FILES = sorted([*ROOT.glob("*.md"), *ROOT.glob("*.txt"), ROOT / ".gitignore"])
TRAIN_SAMPLE = ["00000005", "00000013", "00000033", "00000054", "00000092"]

results = []


def record(name, ok, detail):
    status = "PASS" if ok is True else ("FAIL" if ok is False else str(ok))
    results.append({"check": name, "status": status, "detail": detail})
    print(f"[{status:>7}] {name}: {detail if isinstance(detail, str) else json.dumps(detail)[:300]}")


def file_hash(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ 1
def check_partitions():
    ids, hashes = {}, {}
    for part in config.PARTITIONS:
        for tid in list_trips(part):
            ids.setdefault(tid, []).append(part)
            hashes.setdefault(file_hash(trip_files(part, tid)["track"]), []).append(f"{part}/{tid}")
    dup_ids = {k: v for k, v in ids.items() if len(v) > 1}
    dup_tracks = {k: v for k, v in hashes.items() if len(v) > 1}
    record("partitions_disjoint", not dup_ids and not dup_tracks,
           f"{len(ids)} trips, shared ids={list(dup_ids)}, identical track files={list(dup_tracks.values())}")


# ------------------------------------------------------------------ 2 + 3
def check_features():
    saved = {p: pd.read_pickle(config.FEATURE_DIR / f"{p}_candidates.pkl") for p in config.PARTITIONS}
    todo = [("student_test", t) for t in list_trips("student_test")] + [("train", t) for t in TRAIN_SAMPLE]
    leak, repro = [], []
    for part, tid in todo:
        trip = load_trip(part, tid)
        with_route = build_trip_tables(trip, use_route=True).candidates
        no_route = build_trip_tables(trip, use_route=False).candidates
        same_rows = np.array_equal(with_route[["obs_idx", "arc_id"]].to_numpy(), no_route[["obs_idx", "arc_id"]].to_numpy())
        same_feat = same_rows and np.array_equal(with_route[FEATURE_COLUMNS].to_numpy(), no_route[FEATURE_COLUMNS].to_numpy())
        if not same_feat:
            leak.append(f"{part}/{tid}")
        s = saved[part]
        s = s[s["trip_id"] == tid]
        rebuilt = with_route[FEATURE_COLUMNS].astype(np.float32).to_numpy()
        if not (len(s) == len(rebuilt) and np.array_equal(s[FEATURE_COLUMNS].to_numpy(), rebuilt)
                and np.array_equal(s["is_match"].to_numpy(), with_route["is_match"].to_numpy())):
            repro.append(f"{part}/{tid}")
    record("features_route_free", not leak, f"{len(todo)} trips checked, differing: {leak}")
    record("features_reproducible", not repro, f"{len(todo)} trips rebuilt, mismatching saved tables: {repro}")


# ------------------------------------------------------------------ 4 + 5
def check_models():
    comp = pd.read_csv(config.RESULTS_DIR / "model_comparison.csv").set_index("model")
    te_cand = pd.read_pickle(config.FEATURE_DIR / "student_test_candidates.pkl")
    te_fix = pd.read_pickle(config.FEATURE_DIR / "student_test_fixes.pkl")
    X = te_cand[FEATURE_COLUMNS].to_numpy(np.float32)
    ok = te_cand["label_ok"].to_numpy()
    y = te_cand["is_match"].to_numpy().astype(int)

    bad, skipped, checked, preds = [], [], [], {}
    for key in [k for k in comp.index if k in MODEL_ZOO]:
        path = config.MODEL_DIR / f"{key}_untuned.joblib"
        if not path.exists():
            skipped.append(key)
            continue
        p = joblib.load(path).predict_proba(X)[:, 1]
        preds[key] = p
        m = candidate_metrics(y[ok], p[ok])
        t1 = top1_accuracy(te_fix, top1_choice(te_cand, p))
        got = {"test_f1_macro": m["f1_macro"], "test_log_loss": m["log_loss"], "test_top1": t1["top1_acc"]}
        diff = {k: abs(v - comp.loc[key, k]) for k, v in got.items()}
        checked.append(key)
        if max(diff.values()) > 1e-9:
            bad.append({key: diff})
    detail = f"re-predicted {checked}; mismatches: {bad}"
    if skipped:
        detail += f"; skipped (model file not present): {skipped}"
    record("reported_metrics_match", not bad, detail)

    # refit two cheap models from scratch and compare with the saved ones
    tr = pd.read_pickle(config.FEATURE_DIR / "train_candidates.pkl")
    setup = json.loads((config.RESULTS_DIR / "experiment_setup.json").read_text())
    tr = tr[tr["label_ok"] & (tr["obs_idx"] % setup["train_stride"] == 0)]
    Xtr, ytr = tr[FEATURE_COLUMNS].to_numpy(np.float32), tr["is_match"].to_numpy().astype(int)
    diffs = {}
    for key in ("logreg", "hgb"):
        if key in preds:
            p = MODEL_ZOO[key].build().fit(Xtr, ytr).predict_proba(X)[:, 1]
            diffs[key] = float(np.max(np.abs(p - preds[key])))
    record("refit_deterministic", bool(diffs) and max(diffs.values()) < 1e-9,
           f"max |p_refit - p_saved| per model: {diffs}")


# ------------------------------------------------------------------ 6 / 7 / 8
def check_static():
    texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in CODE_FILES + DOC_FILES if p.exists()}
    me = Path(__file__).resolve()

    evaluator_refs = [f"{p.relative_to(ROOT)}:{i}" for p, t in texts.items() if p.resolve() != me
                      for i, line in enumerate(t.splitlines(), 1)
                      if re.search(r"evaluator[_ ]?test|eval_test", line, re.I) and not line.lstrip().startswith(("#", "*", "-"))]
    parts_cfg = list(config.PARTITIONS)
    record("no_evaluator_data", parts_cfg == ["train", "student_test"] and not evaluator_refs,
           f"partitions in config: {parts_cfg}; code lines referencing evaluator data: {evaluator_refs}")

    # an actual credential assignment (key = "....") or a private key block - prose that merely
    # mentions the word "secret" is not a finding
    secret_pat = re.compile(r"((api[_-]?key|secret|password|passwd|token)\w*\s*[:=]\s*['\"][^'\"\s]{6,}['\"]"
                            r"|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})", re.I)
    email_pat = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    hits = [f"{p.relative_to(ROOT)}:{i}" for p, t in texts.items() if p.resolve() != me
            for i, line in enumerate(t.splitlines(), 1) if secret_pat.search(line) or email_pat.search(line)]
    record("no_secrets_or_personal", not hits, f"suspicious lines: {hits}")

    # allowed: picking which trips the EDA plots/maps show, and CLI usage examples in docstrings
    ids, unexplained = [], []
    for p, t in texts.items():
        if p.resolve() == me or p.suffix != ".py":
            continue
        for i, line in enumerate(t.splitlines(), 1):
            for m in re.findall(r"\b0000\d{4}\b", line):
                where = f"{p.relative_to(ROOT)}:{i} {m}"
                ids.append(where)
                if not (p.name == "01_explore_dataset.py" or "--trip" in line):
                    unexplained.append(where)
    record("hardcoded_trip_ids", True if not unexplained else "REVIEW",
           {"outside_plot_selection_or_usage_example": unexplained, "occurrences": ids})


def main():
    check_partitions()
    check_static()
    check_features()
    check_models()
    out = config.RESULTS_DIR / "integrity_audit.json"
    out.write_text(json.dumps(results, indent=2))
    n_fail = sum(r["status"] == "FAIL" for r in results)
    print(f"\n{len(results)} checks, {n_fail} failed -> {out.relative_to(ROOT)}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
