"""Step 6 - HMM + Viterbi route reconstruction and comparison with classifier-only.

1. Emissions = probabilities of the TUNED Gradient Boosting model.
   * train trips: out-of-fold probabilities (5 trip-grouped folds, same as
     phase 1), so the decoder is tuned on honest, not memorised, scores
   * student_test: the tuned model refit on all train rows
2. Decoder parameters (beta, emission weight, unreachable penalty, gap split)
   are chosen by a grid search on the TRAIN trips only (objective: top-1).
3. The chosen decoder is compared with classifier-only decoding on train (OOF)
   and student_test: top-1, route continuity, route-F1, breaks, timing.
4. Folium maps per student_test trip: GPS, reference, classifier-only, HMM.

Run:  python scripts/06_hmm_viterbi.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config, viz
from pathsnap.experiment import group_folds, load_partition, training_rows, xy
from pathsnap.features import FEATURE_COLUMNS, trip_network
from pathsnap.hmm import HMMParams, TripLattice
from pathsnap.metrics import route_overlap
from pathsnap.models import hgb_from_params
from pathsnap.trip_io import load_trip

OUT = config.OUT_DIR / "hmm"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
config.MAPS_DIR.mkdir(parents=True, exist_ok=True)

GRID = {
    "beta_m": [5.0, 10.0, 20.0, 40.0, 80.0],
    "emission_weight": [0.5, 1.0, 2.0, 4.0],
    "log_unreachable": [-15.0, -40.0],
    "gap_split_s": [30.0, 120.0],
}


def oof_train_proba(tr_cand, best_params, stride, n_folds):
    cache = OUT / "train_oof_proba.npy"
    if cache.exists():
        p = np.load(cache)
        if len(p) == len(tr_cand):
            return p
    rows = training_rows(tr_cand, stride)
    X, y = xy(rows)
    all_X = tr_cand[FEATURE_COLUMNS].to_numpy(np.float32)
    trip = tr_cand["trip_id"].to_numpy()
    p = np.full(len(tr_cand), np.nan)
    for k, (tr, va) in enumerate(group_folds(rows, n_folds)):
        held = np.isin(trip, rows["trip_id"].iloc[va].unique())
        est = hgb_from_params(best_params).fit(X[tr], y[tr])
        p[held] = est.predict_proba(all_X[held])[:, 1]
        print(f"  OOF fold {k}: {held.sum():,} rows")
    assert not np.isnan(p).any()
    np.save(cache, p)
    return p


def build_lattices(part, cand, fixes):
    out = {}
    t0 = time.perf_counter()
    for tid, c in cand.groupby("trip_id", sort=True):
        trip = load_trip(part, tid, with_route=False)
        _, _, _, net = trip_network(trip)
        f = fixes[fixes["trip_id"] == tid].reset_index(drop=True)
        out[tid] = (TripLattice(c.reset_index(drop=True), f, net), c.index.to_numpy(), f)
    print(f"  {part}: {len(out)} lattices in {time.perf_counter() - t0:.0f}s")
    return out


def strict_continuity(arcs, net):
    s = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
    return 1.0 if len(s) < 2 else float(np.mean(net.dst[s[:-1]] == net.src[s[1:]]))


def score_trip(lat, idx, fixes, proba, params, route=None, expand=False):
    p = proba[idx]
    if params is None:
        t0 = time.perf_counter(); rows = lat.classifier_only(p); segs = [0]
    else:
        t0 = time.perf_counter(); rows, segs = lat.decode(p, params)
    dec_s = time.perf_counter() - t0
    arcs = lat.arc[rows]
    ok = fixes["label_ok"].to_numpy()
    lab = fixes["label_arc"].to_numpy()
    r = {"n_fix": len(arcs), "n_scored": int(ok.sum()), "correct": int((arcs[ok] == lab[ok]).sum()),
         "continuity": strict_continuity(arcs, lat.net), "segments": len(segs), "decode_s": dec_s}
    if route is not None:
        r["route_f1"] = route_overlap(arcs, route)["route_f1"]
    if expand:
        chains, br = lat.expand(arcs, segs)
        r["unfillable_breaks"] = br
        if route is not None:
            r["route_f1_expanded"] = route_overlap(np.array([a for ch in chains for a in ch]), route)["route_f1"]
        r["_arcs"], r["_chains"] = arcs, chains
    return r


def summarise(per_trip: pd.DataFrame) -> dict:
    out = {"top1": per_trip["correct"].sum() / per_trip["n_scored"].sum(),
           "continuity": per_trip["continuity"].mean(), "decode_ms_per_1k_fixes":
           per_trip["decode_s"].sum() * 1000 / (per_trip["n_fix"].sum() / 1000)}
    for c in ("route_f1", "route_f1_expanded", "unfillable_breaks"):
        if c in per_trip:
            out[c] = per_trip[c].mean() if c != "unfillable_breaks" else int(per_trip[c].sum())
    return out


def main():
    best = json.loads((config.OUT_DIR / "tuning" / "best_params.json").read_text())["params"]
    setup = json.loads((config.RESULTS_DIR / "experiment_setup.json").read_text())
    tr_cand, tr_fix, tr_routes = load_partition("train")
    te_cand, te_fix, te_routes = load_partition("student_test")

    print("emission probabilities ...")
    p_tr = oof_train_proba(tr_cand, best, setup["train_stride"], setup["folds"])
    tuned = joblib.load(config.MODEL_DIR / "hgb_tuned.joblib")
    p_te = tuned.predict_proba(te_cand[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]

    print("building lattices (shortest paths) ...")
    lat_tr = build_lattices("train", tr_cand, tr_fix)
    lat_te = build_lattices("student_test", te_cand, te_fix)

    # ------------------------------------------------ decoder grid search (train only)
    print("grid search on train (OOF emissions) ...")
    grid_rows = []
    t0 = time.perf_counter()
    for vals in itertools.product(*GRID.values()):
        params = HMMParams(**dict(zip(GRID, vals)))
        res = pd.DataFrame([score_trip(lat, idx, f, p_tr, params) for lat, idx, f in lat_tr.values()])
        s = summarise(res)
        grid_rows.append({**params.to_dict(), "train_top1": s["top1"], "train_continuity": s["continuity"]})
    grid = pd.DataFrame(grid_rows).sort_values(["train_top1", "train_continuity"], ascending=False)
    grid.to_csv(OUT / "decoder_grid_train.csv", index=False)
    best_row = grid.iloc[0]
    params = HMMParams(**{k: float(best_row[k]) for k in GRID})
    (OUT / "hmm_params.json").write_text(json.dumps({
        "params": params.to_dict(), "chosen_on": "train trips, out-of-fold emissions, max top-1",
        "grid": GRID, "grid_size": len(grid), "grid_search_s": time.perf_counter() - t0}, indent=2))
    print("best decoder:", params.to_dict(), f"train top-1 {best_row['train_top1']:.4f}")

    # ------------------------------------------------ classifier-only vs HMM
    summary, per_trip_all, keep = [], [], {}
    for part, lats, proba, routes in (("train_oof", lat_tr, p_tr, tr_routes), ("student_test", lat_te, p_te, te_routes)):
        for name, prm in (("classifier_only", None), ("hmm_viterbi", params)):
            rows = []
            for tid, (lat, idx, f) in lats.items():
                r = score_trip(lat, idx, f, proba, prm, routes[tid], expand=(prm is not None))
                if part == "student_test":
                    keep[(tid, name)] = (r.pop("_arcs", lat.arc[lat.classifier_only(proba[idx])]), r.pop("_chains", None))
                else:
                    r.pop("_arcs", None); r.pop("_chains", None)
                rows.append({"partition": part, "decoder": name, "trip_id": tid, **r,
                             "top1": r["correct"] / max(r["n_scored"], 1)})
            df = pd.DataFrame(rows)
            per_trip_all.append(df)
            summary.append({"partition": part, "decoder": name, **summarise(df)})
    summ = pd.DataFrame(summary)
    per_trip = pd.concat(per_trip_all, ignore_index=True)
    summ.to_csv(OUT / "classifier_vs_hmm.csv", index=False)
    per_trip.to_csv(OUT / "classifier_vs_hmm_per_trip.csv", index=False)

    # where did the HMM help / hurt (student_test)
    st = per_trip[per_trip.partition == "student_test"].pivot(index="trip_id", columns="decoder", values="top1")
    st["delta"] = st["hmm_viterbi"] - st["classifier_only"]
    st.to_csv(OUT / "student_test_top1_delta.csv")

    # ------------------------------------------------ plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(st))
    for ax, col, title in ((axes[0], "top1", "top-1 accuracy"), (axes[1], "continuity", "route continuity")):
        piv = per_trip[per_trip.partition == "student_test"].pivot(index="trip_id", columns="decoder", values=col)
        ax.bar(x - 0.2, piv["classifier_only"], 0.4, label="classifier only", color="#7048e8")
        ax.bar(x + 0.2, piv["hmm_viterbi"], 0.4, label="classifier + HMM", color="#e8590c")
        ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=45, fontsize=8)
        ax.set_ylim(max(0, piv.values.min() - 0.1), 1.0); ax.set_title(f"student_test: {title} per trip"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "student_test_classifier_vs_hmm.png", dpi=130); plt.close(fig)

    g = grid[(grid.log_unreachable == params.log_unreachable) & (grid.gap_split_s == params.gap_split_s)]
    hm = g.pivot(index="emission_weight", columns="beta_m", values="train_top1")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(hm.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns); ax.set_xlabel("beta (m)")
    ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index); ax.set_ylabel("emission weight")
    for (i, j), v in np.ndenumerate(hm.values):
        ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8, color="white" if v < hm.values.mean() else "black")
    ax.set_title(f"train top-1 (OOF) - unreachable {params.log_unreachable:g}, gap split {params.gap_split_s:g}s")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(PLOTS / "decoder_grid_train.png", dpi=130); plt.close(fig)

    # ------------------------------------------------ maps
    for tid in sorted(te_routes):
        trip = load_trip("student_test", tid)
        lat = lat_te[tid][0]
        clf_arcs, _ = keep[(tid, "classifier_only")]
        _, chains = keep[(tid, "hmm_viterbi")]
        r_c = per_trip[(per_trip.partition == "student_test") & (per_trip.trip_id == tid) & (per_trip.decoder == "classifier_only")].iloc[0]
        r_h = per_trip[(per_trip.partition == "student_test") & (per_trip.trip_id == tid) & (per_trip.decoder == "hmm_viterbi")].iloc[0]
        m = viz.base_map(trip)
        viz.add_network_layer(m, trip, np.unique(lat.arc), name="Candidate arcs", show=False)
        viz.add_route_layer(m, trip, trip.route, "Reference route", "#2f9e44", weight=8, opacity=0.5)
        viz.add_route_layer(m, trip, clf_arcs, "Classifier only", "#7048e8", weight=3, opacity=0.9)
        fg = folium.FeatureGroup(name="Classifier + HMM (Viterbi)")
        for ch in chains:
            for line in viz._chains(trip, ch):
                folium.PolyLine(line, color="#e8590c", weight=4, opacity=0.95).add_to(fg)
        fg.add_to(m)
        viz.add_gps_layer(m, trip)
        folium.LayerControl(collapsed=False).add_to(m)
        viz.add_legend(m, [("Raw GPS fixes", "#1f4fd6"), ("Reference route", "#2f9e44"),
                           (f"Classifier only (top-1 {r_c.top1:.3f})", "#7048e8"),
                           (f"Classifier + HMM (top-1 {r_h.top1:.3f})", "#e8590c")], title=f"student_test/{tid}")
        m.save(str(config.MAPS_DIR / f"hmm_{tid}.html"))

    print(summ.round(4).to_string(index=False))
    print(st.round(4).to_string())


if __name__ == "__main__":
    main()
