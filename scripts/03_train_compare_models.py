"""Step 3 - train and compare the candidate classifiers.

Protocol
--------
* fitting + 5-fold GroupKFold CV on the 80 TRAIN trips (groups = trip id, so
  no trip is ever split between fit and validation)
* every model is then refit on all train rows and scored on the 10
  STUDENT_TEST trips (all fixes, no thinning)
* only fixes with a reliable label (label_ok) are used for fitting / scoring
* train rows are thinned to every `--stride`-th fix: at 1 Hz neighbouring fixes
  are ~15 m apart and almost duplicate rows, thinning halves the cost and
  reduces within-trip autocorrelation

Metrics per model: macro-F1 / precision / recall / log loss on the binary
candidate task, observation-level top-1 accuracy, candidate miss rate,
classifier-only route continuity and route-F1, fit and inference time.

Run:  python scripts/03_train_compare_models.py            (all models)
      python scripts/03_train_compare_models.py --models rf,hgb --folds 3
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
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import candidate_metrics, route_level_summary, top1_accuracy, top1_choice
from pathsnap.models import MODEL_ZOO

RES = config.RESULTS_DIR
PLOTS = RES / "plots"
for d in (RES, PLOTS, config.MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_partition(part):
    cand = pd.read_pickle(config.FEATURE_DIR / f"{part}_candidates.pkl")
    fixes = pd.read_pickle(config.FEATURE_DIR / f"{part}_fixes.pkl")
    routes = {k: np.asarray(v) for k, v in json.loads((config.FEATURE_DIR / f"{part}_routes.json").read_text()).items()}
    return cand, fixes, routes


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_ZOO))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--plots-only", action="store_true", help="redraw plots from the saved result files")
    args = ap.parse_args()
    keys = [k.strip() for k in args.models.split(",") if k.strip()]

    if args.plots_only:
        summ = pd.read_csv(RES / "model_comparison.csv")
        keys = [k for k in summ["model"] if k != "nearest_arc"]
        make_plots(summ, pd.read_csv(RES / "cv_folds.csv"),
                   pd.read_csv(RES / "student_test_per_trip.csv", dtype={"trip_id": str}),
                   json.loads((RES / "confusion_matrices.json").read_text()),
                   json.loads((RES / "training_histories.json").read_text()), keys)
        return

    tr_cand, tr_fix, _ = load_partition("train")
    te_cand, te_fix, te_routes = load_partition("student_test")

    train = tr_cand[tr_cand["label_ok"] & (tr_cand["obs_idx"] % args.stride == 0)].reset_index(drop=True)
    X = train[FEATURE_COLUMNS].to_numpy(np.float32)
    y = train["is_match"].to_numpy().astype(int)
    groups = train["trip_id"].to_numpy()
    print(f"train rows {len(train):,} (fixes {train.groupby(['trip_id','obs_idx']).ngroups:,}), "
          f"positive share {y.mean():.3f}, trips {len(np.unique(groups))}")

    X_te = te_cand[FEATURE_COLUMNS].to_numpy(np.float32)
    te_ok = te_cand["label_ok"].to_numpy()
    y_te = te_cand["is_match"].to_numpy().astype(int)
    n_te_fixes = len(te_fix)

    folds = list(GroupKFold(n_splits=args.folds).split(X, y, groups))
    fold_rows, summary_rows, per_trip_rows, histories, confusions = [], [], [], {}, {}

    # ------------------------------------------------------------ rule baseline (no learning)
    base_cv = [grouped_top1(train.iloc[va], -train["dist_m"].to_numpy()[va]) for _, va in folds]
    base_best = top1_choice(te_cand, -te_cand["dist_m"].to_numpy())
    base_t1 = top1_accuracy(te_fix, base_best)
    base_route, _ = route_level_summary(base_best, te_routes)
    summary_rows.append({
        "model": "nearest_arc", "name": "Nearest arc (rule, no training)", "family": "Baseline",
        "config": "score = -distance", "cv_top1_mean": np.mean(base_cv), "cv_top1_std": np.std(base_cv),
        "test_top1": base_t1["top1_acc"], "test_candidate_miss": base_t1["candidate_miss_rate"],
        **{f"test_{k}": v for k, v in base_route.items()},
    })
    print(f"baseline nearest arc: CV top1 {np.mean(base_cv):.4f}  test top1 {base_t1['top1_acc']:.4f}")

    rng = np.random.default_rng(config.SEED)
    for key in keys:
        spec = MODEL_ZOO[key]
        print(f"\n=== {spec.name} ({key}) params={spec.params}")
        for f_i, (tr, va) in enumerate(folds):
            est = spec.build()
            t0 = time.perf_counter(); est.fit(X[tr], y[tr]); fit_s = time.perf_counter() - t0
            t0 = time.perf_counter(); p_va = est.predict_proba(X[va])[:, 1]; pred_s = time.perf_counter() - t0
            m = candidate_metrics(y[va], p_va)
            # training-fold score on a sample, for the over/underfitting check
            sub = tr if len(tr) <= 150_000 else rng.choice(tr, 150_000, replace=False)
            m_tr = candidate_metrics(y[sub], est.predict_proba(X[sub])[:, 1])
            row = {"model": key, "fold": f_i, "n_fit": len(tr), "n_val": len(va),
                   "f1_macro": m["f1_macro"], "log_loss": m["log_loss"], "accuracy": m["accuracy"],
                   "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
                   "top1": grouped_top1(train.iloc[va], p_va),
                   "train_f1_macro": m_tr["f1_macro"], "train_log_loss": m_tr["log_loss"],
                   "fit_s": fit_s, "pred_s": pred_s}
            fold_rows.append(row)
            print(f"  fold {f_i}: F1m {row['f1_macro']:.4f}  top1 {row['top1']:.4f}  "
                  f"logloss {row['log_loss']:.4f}  (train F1m {row['train_f1_macro']:.4f})  fit {fit_s:.1f}s")

        # refit on all train rows, evaluate on student_test
        est = spec.build()
        t0 = time.perf_counter(); est.fit(X, y); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p_te = est.predict_proba(X_te)[:, 1]; infer_s = time.perf_counter() - t0
        cm = candidate_metrics(y_te[te_ok], p_te[te_ok])
        best = top1_choice(te_cand, p_te)
        t1 = top1_accuracy(te_fix, best)
        route_sum, route_trips = route_level_summary(best, te_routes)
        joblib.dump(est, config.MODEL_DIR / f"{key}_untuned.joblib", compress=3)

        pt = per_trip_top1(te_fix, best)
        for tid, acc in pt.items():
            rt = route_trips.set_index("trip_id").loc[tid]
            per_trip_rows.append({"model": key, "trip_id": tid, "top1": acc,
                                  "continuity": rt["continuity"], "route_f1": rt.get("route_f1", np.nan)})
        confusions[key] = cm["confusion"]
        if key == "mlp":
            histories[key] = est.history_
        if key == "hgb":
            histories[key] = [{"iter": i, "train_loss": -float(a), "val_loss": -float(b)}
                              for i, (a, b) in enumerate(zip(est.train_score_, est.validation_score_))]

        f = pd.DataFrame([r for r in fold_rows if r["model"] == key])
        summary_rows.append({
            "model": key, "name": spec.name, "family": spec.family,
            "config": json.dumps(spec.params),
            "cv_f1_macro_mean": f["f1_macro"].mean(), "cv_f1_macro_std": f["f1_macro"].std(),
            "cv_top1_mean": f["top1"].mean(), "cv_top1_std": f["top1"].std(),
            "cv_log_loss_mean": f["log_loss"].mean(),
            "train_f1_macro_mean": f["train_f1_macro"].mean(),
            "test_accuracy": cm["accuracy"], "test_precision_macro": cm["precision_macro"],
            "test_recall_macro": cm["recall_macro"], "test_f1_macro": cm["f1_macro"],
            "test_precision_pos": cm["precision_pos"], "test_recall_pos": cm["recall_pos"],
            "test_f1_pos": cm["f1_pos"], "test_log_loss": cm["log_loss"],
            "test_top1": t1["top1_acc"], "test_top1_when_covered": t1["top1_acc_when_covered"],
            "test_candidate_miss": t1["candidate_miss_rate"],
            **{f"test_{k}": v for k, v in route_sum.items()},
            "fit_s_full": fit_s, "cv_fit_s_mean": f["fit_s"].mean(),
            "infer_ms_per_1k_fixes": infer_s * 1000 / (n_te_fixes / 1000),
        })
        s = summary_rows[-1]
        print(f"  student_test: F1m {s['test_f1_macro']:.4f}  top1 {s['test_top1']:.4f}  "
              f"logloss {s['test_log_loss']:.4f}  continuity {s['test_route_continuity']:.4f}  "
              f"route F1 {s['test_route_f1']:.4f}  fit {fit_s:.1f}s  infer {s['infer_ms_per_1k_fixes']:.1f} ms/1k fixes")

    # ------------------------------------------------------------ save tables
    summ = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)
    trips_df = pd.DataFrame(per_trip_rows)
    summ.to_csv(RES / "model_comparison.csv", index=False)
    folds_df.to_csv(RES / "cv_folds.csv", index=False)
    trips_df.to_csv(RES / "student_test_per_trip.csv", index=False)
    (RES / "confusion_matrices.json").write_text(json.dumps(confusions, indent=2))
    (RES / "training_histories.json").write_text(json.dumps(histories, indent=2))
    (RES / "experiment_setup.json").write_text(json.dumps({
        "seed": config.SEED, "folds": args.folds, "cv": "GroupKFold by trip_id",
        "train_stride": args.stride, "train_rows": int(len(train)), "features": FEATURE_COLUMNS,
        "student_test_rows": int(len(te_cand)), "student_test_fixes": int(n_te_fixes),
        "models": {k: {"name": MODEL_ZOO[k].name, "family": MODEL_ZOO[k].family, "params": MODEL_ZOO[k].params}
                   for k in keys},
        "threshold_for_binary_metrics": 0.5,
    }, indent=2))

    make_plots(summ, folds_df, trips_df, confusions, histories, keys)

    ranked = summ[summ.model != "nearest_arc"].sort_values("cv_top1_mean", ascending=False)
    cols = ["model", "cv_f1_macro_mean", "cv_top1_mean", "cv_log_loss_mean", "test_f1_macro",
            "test_top1", "test_log_loss", "test_route_continuity", "test_route_f1", "infer_ms_per_1k_fixes"]
    print("\nRanking by CV top-1 (primary selection metric):")
    print(ranked[cols].round(4).to_string(index=False))


def make_plots(summ, folds_df, trips_df, confusions, histories, keys):
    ml = summ[summ.model != "nearest_arc"].set_index("model").loc[keys]
    base = summ[summ.model == "nearest_arc"].iloc[0]
    order = ml.sort_values("cv_f1_macro_mean").index
    names = {k: MODEL_ZOO[k].name for k in keys}

    # CV macro F1 (+-std over folds)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([names[k] for k in order], ml.loc[order, "cv_f1_macro_mean"],
            xerr=ml.loc[order, "cv_f1_macro_std"], color="#3b7dd8", capsize=3)
    ax.set_xlabel("macro F1 (5-fold GroupKFold, train trips)")
    lo = max(0.0, ml["cv_f1_macro_mean"].min() - 0.05)
    ax.set_xlim(lo, 1.0); ax.set_title("Cross-validation macro F1 per model")
    fig.tight_layout(); fig.savefig(PLOTS / "cv_macro_f1.png", dpi=130); plt.close(fig)

    # student test top-1 + macro F1
    o2 = ml.sort_values("test_top1").index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    yy = np.arange(len(o2))
    ax.barh(yy - 0.2, ml.loc[o2, "test_top1"], height=0.4, label="top-1 accuracy (fixes)", color="#2f9e44")
    ax.barh(yy + 0.2, ml.loc[o2, "test_f1_macro"], height=0.4, label="macro F1 (candidates)", color="#e8590c")
    ax.set_yticks(yy); ax.set_yticklabels([names[k] for k in o2])
    ax.set_xlim(max(0.0, min(ml["test_f1_macro"].min(), ml["test_top1"].min()) - 0.05), 1.0)
    ax.set_title(f"Student-test performance (10 trips)  -  nearest-arc rule top-1 = {base['test_top1']:.3f}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "student_test_top1_f1.png", dpi=130); plt.close(fig)

    # log loss
    o3 = ml.sort_values("test_log_loss").index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    yy = np.arange(len(o3))
    ax.barh(yy - 0.2, ml.loc[o3, "cv_log_loss_mean"], height=0.4, label="CV", color="#7048e8")
    ax.barh(yy + 0.2, ml.loc[o3, "test_log_loss"], height=0.4, label="student test", color="#f59f00")
    ax.set_yticks(yy); ax.set_yticklabels([names[k] for k in o3]); ax.set_xscale("log")
    ax.set_xlabel("log loss (log scale, lower is better)"); ax.set_title("Probability quality"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "log_loss.png", dpi=130); plt.close(fig)

    # train vs CV F1 (over/underfitting)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    yy = np.arange(len(order))
    ax.barh(yy - 0.2, ml.loc[order, "train_f1_macro_mean"], height=0.4, label="fit folds", color="#adb5bd")
    ax.barh(yy + 0.2, ml.loc[order, "cv_f1_macro_mean"], height=0.4, label="held-out folds", color="#3b7dd8")
    ax.set_yticks(yy); ax.set_yticklabels([names[k] for k in order]); ax.set_xlim(lo, 1.0)
    ax.set_xlabel("macro F1"); ax.set_title("Fit vs held-out macro F1 (gap = overfitting)"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "train_vs_cv_f1.png", dpi=130); plt.close(fig)

    # speed vs quality
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(ml["infer_ms_per_1k_fixes"], ml["test_top1"], s=40, color="#3b7dd8")
    for k in keys:
        ax.annotate(k, (ml.loc[k, "infer_ms_per_1k_fixes"], ml.loc[k, "test_top1"]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("inference time (ms per 1000 GPS fixes, log)")
    ax.set_ylabel("student-test top-1"); ax.set_title("Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(PLOTS / "speed_vs_top1.png", dpi=130); plt.close(fig)

    # confusion matrices (row-normalised)
    n = len(keys); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    for ax, k in zip(np.ravel(axes), keys):
        c = np.asarray(confusions[k], dtype=float)
        cn = c / c.sum(axis=1, keepdims=True)
        ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cn):
            ax.text(j, i, f"{v:.2f}\n({int(c[i, j]):,})", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7); ax.set_yticklabels(["true 0", "true 1"], fontsize=7)
        ax.set_title(names[k], fontsize=8)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Student-test confusion matrices (candidate level, threshold 0.5)")
    fig.tight_layout(); fig.savefig(PLOTS / "confusion_matrices.png", dpi=130); plt.close(fig)

    # learning curves - only models that really train iteratively
    if histories:
        fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
        for ax, (k, h) in zip(np.atleast_1d(axes), histories.items()):
            h = pd.DataFrame(h)
            xcol = "epoch" if "epoch" in h else "iter"
            ax.plot(h[xcol], h["train_loss"], label="train loss")
            ax.plot(h[xcol], h["val_loss"], label="validation loss")
            ax.set_xlabel(xcol); ax.set_ylabel("log loss"); ax.set_title(f"{names[k]} (final fit)")
            ax.legend()
        fig.tight_layout(); fig.savefig(PLOTS / "learning_curves.png", dpi=130); plt.close(fig)

    # per-trip top-1 heatmap on student test
    hm = trips_df.pivot(index="model", columns="trip_id", values="top1").loc[keys]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(keys) + 1.5))
    im = ax.imshow(hm.values, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([names[k] for k in keys], fontsize=8)
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, rotation=45, fontsize=8)
    for (i, j), v in np.ndenumerate(hm.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < 0.85 else "black")
    fig.colorbar(im, ax=ax, label="top-1 accuracy")
    ax.set_title("Student-test top-1 accuracy per trip")
    fig.tight_layout(); fig.savefig(PLOTS / "student_test_per_trip_top1.png", dpi=130); plt.close(fig)

    # impurity importance for the bagged tree ensembles
    for k in ("rf", "extratrees"):
        path = config.MODEL_DIR / f"{k}_untuned.joblib"
        if k in keys and path.exists():
            est = joblib.load(path)
            imp = pd.Series(est.feature_importances_, index=FEATURE_COLUMNS).sort_values()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(imp.index, imp.values, color="#3b7dd8")
            ax.set_title(f"{names[k]}: impurity-based feature importance"); ax.set_xlabel("mean decrease in impurity")
            fig.tight_layout(); fig.savefig(PLOTS / f"feature_importance_{k}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
