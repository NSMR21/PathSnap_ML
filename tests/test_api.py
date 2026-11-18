"""End-to-end API test on the student_test trips (development stand-in for evaluator trips).

For every trip:  POST /predict through FastAPI's TestClient, then check
  * the response only has the expected top-level keys
  * the reference route never appears in the response (ids or coordinates)
  * top-1 accuracy equals the offline HMM experiment (outputs/hmm) exactly

Run:  python tests/test_api.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config  # noqa: E402

os.environ["PATHSNAP_TRIPS_DIR"] = str(config.PARTITIONS["student_test"])

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from pathsnap.trip_io import read_route  # noqa: E402

ALLOWED = {"trip_id", "n_fixes", "n_candidates", "gps", "predicted_route", "classifier_only_route",
           "metrics", "timing_ms", "model"}


def main():
    client = TestClient(app)
    trips = client.get("/trips").json()["trips"]
    assert len(trips) == 10, trips
    offline = pd.read_csv(config.OUT_DIR / "hmm" / "classifier_vs_hmm_per_trip.csv", dtype={"trip_id": str})
    offline = offline[(offline.partition == "student_test")].pivot(index="trip_id", columns="decoder", values="top1")
    rows, failures = [], []
    for t in trips:
        tid = t["trip_id"]
        r = client.post("/predict", json={"trip_id": tid})
        assert r.status_code == 200, r.text
        res = r.json()
        text = json.dumps(res)
        route = read_route(config.PARTITIONS["student_test"], tid)
        leak = set(res) - ALLOWED or "reference" in text.lower() or json.dumps(route.tolist()) in text
        m = res["metrics"]
        d_hmm = abs(m["top1_hmm"] - offline.loc[tid, "hmm_viterbi"])
        d_clf = abs(m["top1_classifier_only"] - offline.loc[tid, "classifier_only"])
        ok = not leak and d_hmm < 1e-12 and d_clf < 1e-12
        if not ok:
            failures.append(tid)
        rows.append({"trip": tid, "fixes": res["n_fixes"], "top1_hmm": m["top1_hmm"],
                     "top1_clf": m["top1_classifier_only"], "route_f1": m["route_f1_hmm"],
                     "continuity": m["continuity_hmm"], "server_s": res["timing_ms"]["total_ms"] / 1000,
                     "leak": bool(leak), "matches_offline": d_hmm < 1e-12 and d_clf < 1e-12})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("404 on unknown trip:", client.post("/predict", json={"trip_id": "nope"}).status_code)
    print("FAILED:" if failures else "ALL OK", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
