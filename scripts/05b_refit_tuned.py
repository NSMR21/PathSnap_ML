"""Step 5b - refit the tuned model and re-time it next to the untuned one.

The final refit inside 05 was logged at ~6900 s because the laptop slept during
that step, so that number is not a real training cost. This script refits both
configurations back-to-back on the same rows, re-scores student_test, and
rewrites outputs/tuning/tuned_vs_untuned.csv and search_cost.json["refit_s"].

Run:  python scripts/05b_refit_tuned.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from pathsnap import config
from pathsnap.experiment import evaluate_student_test, load_partition, training_rows, xy
from pathsnap.models import MODEL_ZOO, hgb_from_params

TUNE = config.OUT_DIR / "tuning"


def main():
    best = json.loads((TUNE / "best_params.json").read_text())
    rep = json.loads((TUNE / "best_params_5fold_cv.json").read_text())
    setup = json.loads((config.RESULTS_DIR / "experiment_setup.json").read_text())
    tr_cand, _, _ = load_partition("train")
    te_cand, te_fix, te_routes = load_partition("student_test")
    X, y = xy(training_rows(tr_cand, setup["train_stride"]))

    t0 = time.perf_counter(); untuned = MODEL_ZOO["hgb"].build().fit(X, y); fit_u = time.perf_counter() - t0
    t0 = time.perf_counter(); tuned = hgb_from_params(best["params"]).fit(X, y); fit_t = time.perf_counter() - t0
    joblib.dump(tuned, config.MODEL_DIR / "hgb_tuned.joblib", compress=3)

    r_u = evaluate_student_test(untuned, te_cand, te_fix, te_routes)
    r_t = evaluate_student_test(tuned, te_cand, te_fix, te_routes)
    comp = pd.read_csv(config.RESULTS_DIR / "model_comparison.csv").set_index("model").loc["hgb"]
    drop = ("proba", "confusion")
    table = pd.DataFrame([
        {"model": "HGB untuned", "cv_top1": comp["cv_top1_mean"], "cv_f1_macro": comp["cv_f1_macro_mean"],
         "cv_log_loss": comp["cv_log_loss_mean"], "fit_s_full": fit_u, "n_iterations": int(untuned.n_iter_),
         **{k: v for k, v in r_u.items() if k not in drop}},
        {"model": "HGB tuned", "cv_top1": rep["top1_mean"], "cv_f1_macro": rep["f1_macro_mean"],
         "cv_log_loss": rep["log_loss_mean"], "fit_s_full": fit_t, "n_iterations": int(tuned.n_iter_),
         **{k: v for k, v in r_t.items() if k not in drop}},
    ])
    table.to_csv(TUNE / "tuned_vs_untuned.csv", index=False)
    cost = json.loads((TUNE / "search_cost.json").read_text())
    cost["refit_s"] = fit_t
    cost["refit_note"] = "re-timed by scripts/05b_refit_tuned.py (original refit was interrupted by laptop sleep)"
    (TUNE / "search_cost.json").write_text(json.dumps(cost, indent=2))
    print(table[["model", "cv_top1", "cv_f1_macro", "cv_log_loss", "test_top1", "test_f1_macro", "test_log_loss",
                 "test_route_continuity", "infer_ms_per_1k_fixes", "fit_s_full", "n_iterations"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
