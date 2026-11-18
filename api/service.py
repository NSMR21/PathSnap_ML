"""Prediction service behind the API (runs on laptop 1).

Ground-truth separation
-----------------------
* predict_route() loads the trip WITHOUT its .route file and runs the full
  pipeline: candidates -> features -> tuned classifier -> HMM/Viterbi.
* Only afterwards score() reads the .route file (server-side) to compute
  accuracy. The reference route itself is never put into a response.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from pathsnap import config, viz
from pathsnap.features import FEATURE_COLUMNS, build_trip_tables
from pathsnap.hmm import HMMParams, TripLattice
from pathsnap.labels import align_fixes_to_route
from pathsnap.metrics import route_overlap
from pathsnap.trip_io import list_trips_in, load_trip_dir, read_route

MAX_GPS_POINTS = 2500  # points sent to the UI for drawing


class PathSnapService:
    def __init__(self, artifacts_dir: Path, trips_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.trips_dir = Path(trips_dir)
        self.pipeline = json.loads((self.artifacts_dir / "pipeline.json").read_text())
        if self.pipeline["features"] != FEATURE_COLUMNS:
            raise RuntimeError("feature list in pipeline.json does not match the code")
        self.model = joblib.load(self.artifacts_dir / self.pipeline["model"]["file"])
        self.hmm_params = HMMParams(**self.pipeline["hmm"]["params"])
        cand = self.pipeline["candidates"]
        self.radius, self.k_max = cand["radius_m"], cand["max_candidates"]
        self.label_max = self.pipeline["scoring"]["label_max_dist_m"]

    # ------------------------------------------------------------------ listing
    def list_trips(self):
        out = []
        for tid in list_trips_in(self.trips_dir):
            with open(self.trips_dir / tid / f"{tid}.track", "rb") as fh:
                n = sum(1 for line in fh if line.strip())
            out.append({"trip_id": tid, "n_fixes": n})
        return out

    # ------------------------------------------------------------------ prediction (no ground truth)
    def predict_route(self, trip_id: str):
        t = {}
        t0 = time.perf_counter()
        trip = load_trip_dir(self.trips_dir, trip_id, with_route=False)
        t["load_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tt = build_trip_tables(trip, radius=self.radius, k_max=self.k_max, use_route=False)
        t["features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        proba = self.model.predict_proba(tt.candidates[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
        t["classifier_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        lat = TripLattice(tt.candidates, tt.fixes, tt.network)
        rows, segs = lat.decode(proba, self.hmm_params)
        hmm_arcs = lat.arc[rows]
        clf_arcs = lat.arc[lat.classifier_only(proba)]
        chains, breaks = lat.expand(hmm_arcs, segs)
        t["hmm_ms"] = (time.perf_counter() - t0) * 1000
        return trip, tt, hmm_arcs, clf_arcs, chains, breaks, segs, t

    # ------------------------------------------------------------------ scoring (server-side only)
    def score(self, trip_id, tt, hmm_arcs, clf_arcs, chains):
        route = read_route(self.trips_dir, trip_id)
        if route is None:
            return {"available": False}
        f = tt.fixes
        _, lab_arc, lab_dist = align_fixes_to_route(f["x_m"].to_numpy(), f["y_m"].to_numpy(), tt.network, route)
        ok = lab_dist <= self.label_max
        expanded = np.array([a for ch in chains for a in ch])
        return {
            "available": True,
            "n_scored_fixes": int(ok.sum()),
            "top1_hmm": float(np.mean(hmm_arcs[ok] == lab_arc[ok])) if ok.any() else None,
            "top1_classifier_only": float(np.mean(clf_arcs[ok] == lab_arc[ok])) if ok.any() else None,
            "route_f1_hmm": route_overlap(expanded, route)["route_f1"],
            "route_precision_hmm": route_overlap(expanded, route)["route_precision"],
            "route_recall_hmm": route_overlap(expanded, route)["route_recall"],
            "route_f1_classifier_only": route_overlap(clf_arcs, route)["route_f1"],
        }

    @staticmethod
    def continuity(arcs, net):
        s = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
        return 1.0 if len(s) < 2 else float(np.mean(net.dst[s[:-1]] == net.src[s[1:]]))

    # ------------------------------------------------------------------ full request
    def run(self, trip_id: str) -> dict:
        t_all = time.perf_counter()
        trip, tt, hmm_arcs, clf_arcs, chains, breaks, segs, timing = self.predict_route(trip_id)
        t0 = time.perf_counter()
        metrics = self.score(trip_id, tt, hmm_arcs, clf_arcs, chains)
        timing["scoring_ms"] = (time.perf_counter() - t0) * 1000
        net = tt.network
        metrics.update({
            "continuity_hmm": self.continuity(hmm_arcs, net),
            "continuity_classifier_only": self.continuity(clf_arcs, net),
            "unfillable_breaks_hmm": int(breaks),
            "segments": len(segs),
        })
        n = len(trip.track)
        step = max(1, int(np.ceil(n / MAX_GPS_POINTS)))
        gps = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(trip.lat[::step], trip.lon[::step])]
        hmm_lines = [line for ch in chains for line in viz.arc_polylines(trip, ch)]
        timing["total_ms"] = (time.perf_counter() - t_all) * 1000
        return {
            "trip_id": trip_id,
            "n_fixes": n,
            "n_candidates": int(len(tt.candidates)),
            "gps": gps,
            "predicted_route": {"polylines": hmm_lines, "n_arcs": int(sum(len(c) for c in chains))},
            "classifier_only_route": {"polylines": viz.arc_polylines(trip, clf_arcs)},
            "metrics": metrics,
            "timing_ms": {k: round(v, 1) for k, v in timing.items()},
            "model": {"type": self.pipeline["model"]["type"], "created": self.pipeline["created"]},
        }
