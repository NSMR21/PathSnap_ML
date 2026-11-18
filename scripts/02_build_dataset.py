"""Step 2 - labels, candidate sets and the feature tables.

For every trip in train and student_test:
  1. project to metres, index the road network
  2. derive per-fix labels from the .route file (monotone alignment)
  3. generate candidate arcs + features
and save the tables under outputs/features/.

Also runs two checks on the TRAIN partition only (so they can be used to pick
settings without touching any test data):
  * label distance distribution   -> LABEL_MAX_DIST_M
  * candidate radius / cap sweep  -> CANDIDATE_RADIUS_M, MAX_CANDIDATES

Run:  python scripts/02_build_dataset.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.candidates import generate_candidates
from pathsnap.features import FEATURE_COLUMNS, build_trip_tables
from pathsnap.trip_io import list_trips, load_trip

OUT = config.FEATURE_DIR
OUT.mkdir(parents=True, exist_ok=True)
EDA = config.EDA_DIR
EDA.mkdir(parents=True, exist_ok=True)

# NOTE: only fixes with label_dist <= LABEL_MAX_DIST_M are scored, so every radius at or
# above that threshold gives the same miss rate - the 75/100 m rows are kept to show this.
SWEEP_RADII = [15, 25, 35, 50, 75, 100]
SWEEP_CAPS = [3, 5, 8, 10, 15, 25]


def sweep_one_trip(tt) -> pd.DataFrame:
    """For each reliable fix: distance-rank of the true arc among all arcs within 100 m."""
    f = tt.fixes
    ok = f["label_ok"].to_numpy()
    px, py = f["x_m"].to_numpy(), f["y_m"].to_numpy()
    cs = generate_candidates(px, py, tt.network, radius=max(SWEEP_RADII), k_max=200, k_fallback=0)
    lab = f["label_arc"].to_numpy()
    hit = cs.arc == lab[cs.obs]
    rank = np.full(len(f), 10_000)
    rank[cs.obs[hit]] = cs.rank[hit]
    n_within = {r: np.bincount(cs.obs[cs.dist <= r], minlength=len(f)) for r in SWEEP_RADII}
    out = pd.DataFrame({"label_dist": f["label_dist_m"].to_numpy(), "label_rank": rank})
    for r in SWEEP_RADII:
        out[f"n_within_{r}"] = n_within[r]
    return out[ok]


def main():
    cfg_used = {k: getattr(config, k) for k in (
        "CANDIDATE_RADIUS_M", "MAX_CANDIDATES", "FALLBACK_CANDIDATES", "LABEL_MAX_DIST_M",
        "INDEX_STEP_M", "INDEX_MARGIN_M", "HEADING_HALF_WINDOW", "SUPPORT_OFFSETS", "SEED")}
    (OUT / "build_config.json").write_text(json.dumps(cfg_used, indent=2, default=list))

    trip_rows, sweep_parts, label_d = [], [], []
    for part in config.PARTITIONS:
        cands, fixes, routes = [], [], {}
        for tid in list_trips(part):
            t0 = time.perf_counter()
            trip = load_trip(part, tid)
            tt = build_trip_tables(trip)
            dt = time.perf_counter() - t0
            c, f = tt.candidates, tt.fixes
            c[FEATURE_COLUMNS] = c[FEATURE_COLUMNS].astype(np.float32)
            cands.append(c)
            fixes.append(f)
            routes[tid] = trip.route.tolist()
            ok = f["label_ok"].to_numpy()
            trip_rows.append({
                "partition": part, "trip_id": tid, "fixes": len(f),
                "label_ok_share": ok.mean(),
                "label_dist_median": float(f["label_dist_m"].median()),
                "candidate_miss_rate": float(1 - f.loc[ok, "label_in_cands"].mean()) if ok.any() else np.nan,
                "cands_per_fix": len(c) / len(f),
                "fallback_share": float(c.groupby("obs_idx")["is_fallback"].first().mean()),
                "build_s": dt,
            })
            if part == "train":
                label_d.append(f["label_dist_m"].to_numpy())
                sweep_parts.append(sweep_one_trip(tt))
            r = trip_rows[-1]
            print(f"{part:>12} {tid}  fixes={r['fixes']:6d}  labelled={r['label_ok_share']:.3f}  "
                  f"miss={r['candidate_miss_rate']:.3f}  cands/fix={r['cands_per_fix']:.1f}  {dt:.1f}s")

        cand_df = pd.concat(cands, ignore_index=True)
        fix_df = pd.concat(fixes, ignore_index=True)
        cand_df.to_pickle(OUT / f"{part}_candidates.pkl")
        fix_df.to_pickle(OUT / f"{part}_fixes.pkl")
        (OUT / f"{part}_routes.json").write_text(json.dumps(routes))
        print(f"[{part}] candidate rows={len(cand_df):,}  fixes={len(fix_df):,}  "
              f"positives={int(cand_df['is_match'].sum()):,}")

    per_trip = pd.DataFrame(trip_rows)
    per_trip.to_csv(OUT / "per_trip_build_summary.csv", index=False)

    # ---------------- label distance distribution (train only)
    d = np.concatenate(label_d)
    q = {f"p{p}": float(np.percentile(d, p)) for p in (50, 75, 90, 95, 98, 99)}
    q["share_le_threshold"] = float(np.mean(d <= config.LABEL_MAX_DIST_M))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(d, 0, 200), bins=100, color="#2b6cb0")
    ax.axvline(config.LABEL_MAX_DIST_M, color="red", ls="--", label=f"threshold {config.LABEL_MAX_DIST_M:.0f} m")
    ax.set_yscale("log"); ax.set_xlabel("distance GPS fix -> aligned reference arc (m, clipped 200)")
    ax.set_ylabel("fixes (log)"); ax.set_title("train: label distance after monotone alignment"); ax.legend()
    fig.tight_layout(); fig.savefig(EDA / "label_distance_train.png", dpi=130); plt.close(fig)

    # ---------------- radius / cap sweep (train only)
    sw = pd.concat(sweep_parts, ignore_index=True)
    rows = []
    for r in SWEEP_RADII:
        for k in SWEEP_CAPS:
            covered = (sw["label_dist"] <= r) & (sw["label_rank"] < k)
            rows.append({"radius_m": r, "max_cands": k, "miss_rate": float(1 - covered.mean()),
                         "mean_cands": float(np.minimum(sw[f"n_within_{r}"], k).mean())})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(EDA / "candidate_sweep_train.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, g in sweep.groupby("max_cands"):
        ax.plot(g["radius_m"], g["miss_rate"] * 100, marker="o", label=f"cap {k}")
    ax.axvline(config.CANDIDATE_RADIUS_M, color="grey", ls=":")
    ax.set_yscale("log"); ax.set_xlabel("candidate radius (m)"); ax.set_ylabel("candidate miss rate (%)")
    ax.set_title("train: true arc missing from candidate set"); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(EDA / "candidate_sweep_train.png", dpi=130); plt.close(fig)

    summary = {"label_distance_train": q, "chosen": cfg_used,
               "chosen_point_in_sweep": sweep[(sweep.radius_m == config.CANDIDATE_RADIUS_M)
                                              & (sweep.max_cands == config.MAX_CANDIDATES)].to_dict("records")}
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2, default=list))
    print(json.dumps(summary, indent=2, default=list))
    print(sweep.pivot(index="radius_m", columns="max_cands", values="miss_rate").round(4))


if __name__ == "__main__":
    main()
