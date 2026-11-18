"""PathSnap FastAPI backend - runs on laptop 1.

Start (from the repository root):
    run_api.bat
or  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Settings (environment variables):
    PATHSNAP_TRIPS_DIR   folder holding the trips to serve, laid out as <dir>/<id>/<id>.track ...
                         default: materials/evaluator_test if it exists, otherwise the
                         student_test partition (development stand-in, logged as a warning)
    PATHSNAP_ARTIFACTS   folder with candidate_model.joblib + pipeline.json (default: artifacts/)

Endpoints:
    GET  /health           model + data folder info
    GET  /trips            the trips the UI may offer (ids + fix counts, no labels)
    POST /predict          {"trip_id": "..."} -> predicted route, metrics, timings
"""
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.service import PathSnapService
from pathsnap import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")


def _trips_dir() -> Path:
    env = os.environ.get("PATHSNAP_TRIPS_DIR")
    if env:
        return Path(env)
    ev = ROOT / "materials" / "evaluator_test"
    if ev.is_dir():
        return ev
    log.warning("PATHSNAP_TRIPS_DIR not set and materials/evaluator_test missing - "
                "serving student_test as a development stand-in")
    return config.PARTITIONS["student_test"]


TRIPS_DIR = _trips_dir()
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
service = PathSnapService(ARTIFACTS, TRIPS_DIR)
log.info("model loaded from %s", ARTIFACTS)
log.info("serving trips from %s", TRIPS_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path, client,
             response.status_code, (time.perf_counter() - t0) * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": service.pipeline["model"]["type"], "created": service.pipeline["created"],
            "hmm": service.pipeline["hmm"]["params"], "trips_dir": TRIPS_DIR.name}


@app.get("/trips")
def trips():
    items = service.list_trips()
    log.info("listed %d trips", len(items))
    return {"trips": items}


@app.post("/predict")
def predict(req: PredictRequest):
    valid = {t["trip_id"] for t in service.list_trips()}
    if req.trip_id not in valid:
        raise HTTPException(status_code=404, detail=f"unknown trip {req.trip_id}")
    log.info("predict trip %s: running candidates -> features -> classifier -> HMM/Viterbi", req.trip_id)
    out = service.run(req.trip_id)
    m, t = out["metrics"], out["timing_ms"]
    if m.get("available"):
        log.info("trip %s | fixes %d | top-1 HMM %.4f (classifier only %.4f) | route-F1 %.4f | "
                 "continuity %.3f | breaks %d", req.trip_id, out["n_fixes"], m["top1_hmm"],
                 m["top1_classifier_only"], m["route_f1_hmm"], m["continuity_hmm"], m["unfillable_breaks_hmm"])
    else:
        log.info("trip %s | fixes %d | no reference route on server - accuracy not available", req.trip_id, out["n_fixes"])
    log.info("trip %s | timing ms: %s", req.trip_id, ", ".join(f"{k}={v:.0f}" for k, v in t.items()))
    return out
