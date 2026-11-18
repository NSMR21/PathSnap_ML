"""Step 4 - folium map of classifier-only predictions for a student_test trip.

Shows raw GPS, nearby road network, reference route and the route obtained by
taking the top-1 candidate of every fix independently (no HMM yet). Arc
changes where the predicted arcs are NOT connected are marked in red - those
breaks are exactly what the HMM/Viterbi stage has to fix.

Run:  python scripts/04_map_classifier_route.py                      (best model by CV top-1)
      python scripts/04_map_classifier_route.py --model rf --trip 00000040
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import joblib
import numpy as np
import pandas as pd

from pathsnap import config, viz
from pathsnap.features import FEATURE_COLUMNS, build_trip_tables
from pathsnap.metrics import route_continuity, top1_choice
from pathsnap.trip_io import list_trips, load_trip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--trip", default=None, help="student_test trip id (default: all)")
    args = ap.parse_args()

    key = args.model
    if key is None:
        comp = pd.read_csv(config.RESULTS_DIR / "model_comparison.csv")
        key = comp[comp.model != "nearest_arc"].sort_values("cv_top1_mean", ascending=False).iloc[0]["model"]
    est = joblib.load(config.MODEL_DIR / f"{key}_untuned.joblib")
    trips = [args.trip] if args.trip else list_trips("student_test")
    config.MAPS_DIR.mkdir(parents=True, exist_ok=True)

    for tid in trips:
        trip = load_trip("student_test", tid)
        # features are built WITHOUT the route; the route is only drawn for comparison
        tt = build_trip_tables(trip, use_route=False)
        cand = tt.candidates
        p = est.predict_proba(cand[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
        best = top1_choice(cand, p).sort_values("obs_idx")
        seq = best["pred_arc"].to_numpy()

        m = viz.base_map(trip)
        viz.add_network_layer(m, trip, np.unique(cand["arc_id"].to_numpy()), name="Candidate arcs", show=False)
        viz.add_route_layer(m, trip, trip.route, "Reference route", "#2f9e44", weight=7, opacity=0.6)
        viz.add_route_layer(m, trip, seq, f"Classifier-only route ({key})", "#7048e8", weight=3, opacity=0.95)
        viz.add_gps_layer(m, trip)

        # mark breaks: consecutive distinct predicted arcs that are not connected
        runs = np.r_[True, seq[1:] != seq[:-1]]
        arcs = seq[runs]
        obs_at = best["obs_idx"].to_numpy()[runs]
        brk = folium.FeatureGroup(name="Route breaks (not connected)")
        n_breaks = 0
        for a, b, o in zip(arcs[:-1], arcs[1:], obs_at[1:]):
            if trip.arcs[a, 1] != trip.arcs[b, 0]:
                n_breaks += 1
                folium.CircleMarker([trip.lat[o], trip.lon[o]], radius=5, color="#e03131", fill=True,
                                    fill_opacity=0.9, tooltip=f"break at fix {o}: arc {a} -> {b}").add_to(brk)
        brk.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        cont = route_continuity(best)
        viz.add_legend(m, [("Raw GPS fixes", "#1f4fd6"), ("Reference route", "#2f9e44"),
                           (f"Classifier-only ({key})", "#7048e8"), ("Break", "#e03131")],
                       title=f"student_test/{tid} - continuity {cont:.3f}")
        out = config.MAPS_DIR / f"classifier_only_{key}_{tid}.html"
        m.save(str(out))
        print(f"{tid}: {n_breaks} breaks, continuity {cont:.3f} -> {out.name}")


if __name__ == "__main__":
    main()
