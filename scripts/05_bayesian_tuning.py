"""Step 5 - Bayesian hyper-parameter optimisation of the chosen model.

Model:      Histogram Gradient Boosting (selection argument in outputs/results:
            tied with Random Forest on CV top-1, best student-test top-1 and log
            loss, ~170x smaller model file, ~9x faster to fit).
Search:     Optuna TPE sampler (sequential model-based / Bayesian optimisation),
            median pruner to stop clearly bad trials after 2 folds.
Objective:  mean observation-level top-1 accuracy over 3 trip-grouped folds of
            the train trips (3 instead of 5 to keep ~30 trials affordable on a
            laptop). Log loss and macro-F1 are recorded for every trial too.
Afterwards: the best configuration is re-scored on the SAME 5 trip-grouped
            folds as the phase-1 comparison (so tuned vs untuned use identical
            metrics and partitions), refit on all train rows, scored on
            student_test and compared with the untuned model.

Everything is kept: the Optuna study (sqlite), a trials CSV, plots and the
tuned-vs-untuned table under outputs/tuning/.

Run:  python scripts/05_bayesian_tuning.py --trials 40
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from pathsnap import config
from pathsnap.experiment import (evaluate_student_test, group_folds, grouped_top1, load_partition,
                                 training_rows, xy)
from pathsnap.metrics import candidate_metrics
from pathsnap.models import MODEL_ZOO

TUNE = config.OUT_DIR / "tuning"
PLOTS = TUNE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Search space (why these ranges):
#  learning_rate      0.02-0.3 log  : below 0.02 needs >1000 trees, above 0.3 is unstable
#  max_iter           100-800       : upper bound only, early stopping usually ends sooner
#  max_leaf_nodes     15-127 log    : tree complexity; untuned used 31
#  min_samples_leaf   10-300 log    : smoothing; rows are ~600k so 10 is still small
#  l2_regularization  1e-3-10 log   : shrinks leaf values
#  max_features       0.4-1.0       : feature sub-sampling per split (decorrelates trees)
SPACE = {
    "learning_rate": ("float", 0.02, 0.3, True),
    "max_iter": ("int", 100, 800, False),
    "max_leaf_nodes": ("int", 15, 127, True),
    "min_samples_leaf": ("int", 10, 300, True),
    "l2_regularization": ("float", 1e-3, 10.0, True),
    "max_features": ("float", 0.4, 1.0, False),
}


def build_hgb(params):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"], max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"], max_features=params.get("max_features", 1.0),
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=config.SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--search-folds", type=int, default=3)
    args = ap.parse_args()

    tr_cand, _, _ = load_partition("train")
    te_cand, te_fix, te_routes = load_partition("student_test")
    setup = json.loads((config.RESULTS_DIR / "experiment_setup.json").read_text())
    rows = training_rows(tr_cand, setup["train_stride"])
    X, y = xy(rows)
    folds = group_folds(rows, args.search_folds)
    folds_report = group_folds(rows, setup["folds"])  # identical to the phase-1 folds
    print(f"train rows {len(rows):,}, search folds {len(folds)}, report folds {len(folds_report)}, trials {args.trials}")

    def cv_report(params):
        """Score one configuration on the phase-1 folds."""
        out = {"top1": [], "f1_macro": [], "log_loss": []}
        for tr, va in folds_report:
            est = build_hgb(params).fit(X[tr], y[tr])
            p = est.predict_proba(X[va])[:, 1]
            m = candidate_metrics(y[va], p)
            out["top1"].append(grouped_top1(rows.iloc[va], p))
            out["f1_macro"].append(m["f1_macro"]); out["log_loss"].append(m["log_loss"])
        return {f"{k}_{s}": float(getattr(np, s)(v)) for k, v in out.items() for s in ("mean", "std")}

    def objective(trial):
        params = {}
        for name, (kind, lo, hi, log) in SPACE.items():
            params[name] = (trial.suggest_int(name, lo, hi, log=log) if kind == "int"
                            else trial.suggest_float(name, lo, hi, log=log))
        tops, f1s, lls, iters, fit_t = [], [], [], [], []
        for k, (tr, va) in enumerate(folds):
            est = build_hgb(params)
            t0 = time.perf_counter(); est.fit(X[tr], y[tr]); fit_t.append(time.perf_counter() - t0)
            p = est.predict_proba(X[va])[:, 1]
            m = candidate_metrics(y[va], p)
            tops.append(grouped_top1(rows.iloc[va], p)); f1s.append(m["f1_macro"]); lls.append(m["log_loss"])
            iters.append(int(est.n_iter_))
            trial.report(float(np.mean(tops)), k)
            if trial.should_prune():
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("folds_done", len(folds))
        trial.set_user_attr("cv_top1_std", float(np.std(tops, ddof=1)))
        trial.set_user_attr("cv_f1_macro", float(np.mean(f1s)))
        trial.set_user_attr("cv_f1_macro_std", float(np.std(f1s, ddof=1)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("mean_iterations", float(np.mean(iters)))
        trial.set_user_attr("fit_s_mean", float(np.mean(fit_t)))
        return float(np.mean(tops))

    db = TUNE / "optuna_hgb.db"
    if db.exists():  # every run is a fresh, complete search record
        db.unlink()
    storage = f"sqlite:///{db.as_posix()}"
    study = optuna.create_study(
        study_name="hgb_top1", direction="maximize", storage=storage, load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=10, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    # trial 0 = the untuned configuration (l2 0 -> 1e-3, the bottom of the log range), so the
    # search starts from it. The "untuned" row of the final table uses the exact phase-1 run.
    base = dict(MODEL_ZOO["hgb"].params)
    study.enqueue_trial({k: base[k] for k in SPACE if k in base} | {"l2_regularization": 1e-3, "max_features": 1.0})

    t_start = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number:3d} {t.state.name:8s} "
                                                 f"value={t.value if t.value is not None else float('nan'):.4f} "
                                                 f"params={ {k: round(v, 4) if isinstance(v, float) else v for k, v in t.params.items()} }",
                                                 flush=True)])
    search_s = time.perf_counter() - t_start

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials["duration_s"] = trials["duration"].dt.total_seconds()
    trials.drop(columns=["duration"]).to_csv(TUNE / "trials.csv", index=False)
    best = study.best_trial
    best_params = dict(best.params)
    (TUNE / "best_params.json").write_text(json.dumps({
        "best_trial": best.number, "cv_top1": best.value, **{f"cv_{k}": v for k, v in best.user_attrs.items()},
        "params": best_params, "fixed": {"early_stopping": True, "validation_fraction": 0.1,
                                         "n_iter_no_change": 15, "random_state": config.SEED}}, indent=2))

    # ------------------------------------------------ 5-fold re-score, refit, compare with untuned
    rep = cv_report(best_params)
    (TUNE / "best_params_5fold_cv.json").write_text(json.dumps(rep, indent=2))
    tuned = build_hgb(best_params)
    t0 = time.perf_counter(); tuned.fit(X, y); fit_tuned = time.perf_counter() - t0
    joblib.dump(tuned, config.MODEL_DIR / "hgb_tuned.joblib", compress=3)
    # re-fit the untuned config now so both fit times come from the same machine state
    t0 = time.perf_counter(); untuned = MODEL_ZOO["hgb"].build().fit(X, y); fit_untuned = time.perf_counter() - t0
    r_t = evaluate_student_test(tuned, te_cand, te_fix, te_routes)
    r_u = evaluate_student_test(untuned, te_cand, te_fix, te_routes)
    base_trial = study.trials[0]
    comp = pd.read_csv(config.RESULTS_DIR / "model_comparison.csv").set_index("model").loc["hgb"]
    table = pd.DataFrame([
        {"model": "HGB untuned", "cv_top1": comp["cv_top1_mean"], "cv_f1_macro": comp["cv_f1_macro_mean"],
         "cv_log_loss": comp["cv_log_loss_mean"], "fit_s_full": fit_untuned,
         **{k: v for k, v in r_u.items() if k not in ("proba", "confusion")}},
        {"model": "HGB tuned", "cv_top1": rep["top1_mean"], "cv_f1_macro": rep["f1_macro_mean"],
         "cv_log_loss": rep["log_loss_mean"], "fit_s_full": fit_tuned,
         **{k: v for k, v in r_t.items() if k not in ("proba", "confusion")}},
    ])
    table.to_csv(TUNE / "tuned_vs_untuned.csv", index=False)
    states = trials["state"].value_counts().to_dict()
    cost = {"trials": len(study.trials), "states": states, "search_wall_s": search_s,
            "mean_trial_s": float(trials["duration_s"].mean()), "refit_s": fit_tuned}
    (TUNE / "search_cost.json").write_text(json.dumps(cost, indent=2))
    (TUNE / "search_space.json").write_text(json.dumps(
        {k: {"type": v[0], "low": v[1], "high": v[2], "log": v[3]} for k, v in SPACE.items()}
        | {"_objective": f"maximise mean {args.search_folds}-fold GroupKFold top-1 (train trips); "
                         f"best re-scored on the {setup['folds']} phase-1 folds",
           "_sampler": "TPE (multivariate, 10 random start-up trials, seed 352)",
           "_pruner": "median pruner after fold 2, 8 start-up trials"}, indent=2))

    # ------------------------------------------------ plots
    done = trials[trials.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(done["number"], done["value"], s=22, color="#3b7dd8", label="completed trial")
    pr = trials[trials.state == "PRUNED"]
    ax.scatter(pr["number"], [done["value"].min()] * len(pr), s=18, marker="x", color="#adb5bd", label="pruned (shown at min)")
    ax.plot(done["number"], done["value"].cummax(), color="#e8590c", label="best so far")
    ax.axhline(base_trial.value, color="black", ls="--", lw=1, label=f"trial 0 = untuned config ({base_trial.value:.4f})")
    ax.set_xlabel("trial"); ax.set_ylabel(f"search objective: CV top-1 ({args.search_folds}-fold GroupKFold)"); ax.set_title("Bayesian optimisation history (TPE)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(PLOTS / "optimisation_history.png", dpi=130); plt.close(fig)

    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#3b7dd8")
        ax.set_xlabel("importance (fANOVA)"); ax.set_title("Which hyper-parameters mattered")
        fig.tight_layout(); fig.savefig(PLOTS / "param_importance.png", dpi=130); plt.close(fig)
        (TUNE / "param_importance.json").write_text(json.dumps(imp, indent=2))
    except Exception as exc:  # importance needs enough completed trials
        print("param importance skipped:", exc)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, name in zip(axes.ravel(), SPACE):
        ax.scatter(done[f"params_{name}"], done["value"], s=16, c=done["number"], cmap="viridis")
        ax.axvline(best_params[name], color="red", lw=1)
        if SPACE[name][3]:
            ax.set_xscale("log")
        ax.set_title(name, fontsize=9); ax.set_ylabel("CV top-1", fontsize=8)
    fig.suptitle("Objective vs each hyper-parameter (colour = trial number, red = best)")
    fig.tight_layout(); fig.savefig(PLOTS / "param_slices.png", dpi=130); plt.close(fig)

    print(json.dumps(cost, indent=2))
    print("best params:", best_params)
    print(table[["model", "cv_top1", "cv_f1_macro", "cv_log_loss", "test_top1", "test_f1_macro", "test_log_loss",
                 "infer_ms_per_1k_fixes", "fit_s_full"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
