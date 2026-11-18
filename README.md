# PathSnap: map-matching noisy GPS tracks to a road network

UE24CS352A Machine Learning hackathon (PES University).

## Status

| Stage | State |
|---|---|
| 1-2. Problem + raw data exploration | done (`scripts/01_explore_dataset.py`) |
| 3. Pre-processing, labels, candidates, features | done (`scripts/02_build_dataset.py`) |
| 4-5. Train + compare 8 models (grouped CV + student_test) | done (`scripts/03_train_compare_models.py`) |
| Classifier-only route maps | done (`scripts/04_map_classifier_route.py`) |
| 6. Bayesian hyper-parameter optimisation | done (`scripts/05_bayesian_tuning.py`, `05b_refit_tuned.py`, `outputs/tuning/`) |
| 7. HMM + Viterbi route reconstruction | done (`pathsnap/hmm.py`, `scripts/06_hmm_viterbi.py`, `outputs/hmm/`) |
| 8. Export final artifacts | done (`scripts/07_export_artifacts.py`, `artifacts/`) |
| 9-10. FastAPI backend (laptop 1) + evaluator UI (laptop 2) | done (`api/`, `ui/`, `run_api.bat`, `run_ui.bat`) |
| 11. Integration test | `tests/test_api.py` (all student_test trips through the API) |

## Folder layout

```
hackathon/
  pathsnap/            library code (imported by all scripts, later by the API)
    config.py          paths, radius, thresholds, seed
    trip_io.py         .track/.nodes/.arcs/.route readers
    geo.py             local metric projection, point-to-segment distance, bearings
    network.py         road graph + spatial index over arcs
    labels.py          per-fix labels from the arc-level route (monotone alignment)
    candidates.py      candidate arcs per GPS fix
    features.py        feature table (fix x candidate)
    metrics.py         candidate / top-1 / route metrics
    models.py          model zoo (8 classifiers, untuned configs)
    experiment.py      shared data loading / scoring for the experiment scripts
    hmm.py             HMM + Viterbi decoder, shortest-path router, route expansion
    viz.py             folium helpers
  scripts/             numbered pipeline steps
  api/                 FastAPI backend (main.py = endpoints + logging, service.py = pipeline)
  ui/                  evaluator web page (index.html, app.js, style.css)
  artifacts/           exported tuned model + pipeline.json used by the API
  tests/               end-to-end API test
  outputs/
    eda/               inventory, data-quality facts, EDA plots, sweeps
    features/          feature tables (pkl, git-ignored) + build summaries
    models/            fitted models (*.joblib; untuned RF/ExtraTrees are git-ignored, >40 MB)
    results/           comparison tables, CV folds, per-trip scores, logs, plots/
    maps/              folium HTML maps
  materials/           raw data + handout PDFs (git-ignored)
```

## Setup

```bash
python -m pip install -r requirements.txt
```

The raw data is expected in `materials/data/Student folder/{train,student_test}/`.
To keep it elsewhere set `PATHSNAP_DATA` to the folder that contains `train/` and `student_test/`.

## Running the pipeline

```bash
python scripts/01_explore_dataset.py        # ~1 min   EDA tables, plots, folium maps
python scripts/02_build_dataset.py          # ~1-2 min labels + candidates + features
python scripts/03_train_compare_models.py   # longest  8 models, 5-fold GroupKFold + student_test
python scripts/04_map_classifier_route.py   # maps of classifier-only routes (student_test)
```

```bash
python scripts/05_bayesian_tuning.py        # ~35 min  Optuna TPE search for Gradient Boosting
python scripts/05b_refit_tuned.py           # ~1 min   refit + re-time tuned vs untuned
python scripts/06_hmm_viterbi.py            # ~15 min  decoder grid (train), classifier vs HMM, maps
python scripts/07_export_artifacts.py       # seconds  artifacts/ for the API
python tests/test_api.py                    # ~1 min   all student_test trips through the API
```

`03` accepts `--models rf,hgb` and `--folds 3` for quicker partial runs.

## Two-laptop deployment

**Laptop 1 (API).** Install the requirements, keep `artifacts/` from the repo, and extract the
evaluation trips so the folder looks like `materials/evaluator_test/<id>/<id>.track ...`
(or set `PATHSNAP_TRIPS_DIR` to wherever they are). Then:

```bash
run_api.bat
```

The console shows every request with the trip, accuracy and timings. Allow Python through
Windows Firewall on port 8000 and note the laptop's IP address (`ipconfig`).

**Laptop 2 (UI).** Same Wi-Fi network. From the repository folder:

```bash
run_ui.bat
```

The page opens at `http://localhost:8501`. Type laptop 1's address in the API box
(for example `http://192.168.1.23:8000`) and press Connect. The ten trips appear; click one to
run it. The page needs internet for the map tiles; without it the routes are drawn on a plain
background instead.

The API only returns the predicted route and the accuracy numbers. The reference route stays on
laptop 1 and is read only after prediction, to score it.

Before submitting, run the integrity self-audit. It checks that partitions are disjoint, that
features don't depend on the route, that results are reproducible and match the saved
models, and that the code has no secrets or hard-coded trips:

```bash
python scripts/integrity_audit.py           # writes outputs/results/integrity_audit.json
```

## How the pipeline works (short version)

1. **Projection.** Every trip is projected to a local east/north plane in metres around its
   median fix (equirectangular). Only short distances are measured (fix to nearby road),
   so the flat-earth error is negligible and no extra GIS dependency is needed.
2. **Labels.** `.route` lists arcs, not fixes. Each fix gets the route arc chosen by a
   *monotone* alignment (DTW-style dynamic programme): the route position may only move
   forward in time, and total fix-to-arc distance is minimised. Fixes more than 50 m from
   their aligned arc are marked unreliable and skipped for fitting and scoring.
3. **Candidates.** All arcs within 50 m of a fix, at most 10, closest first. If there are
   none, the 3 nearest arcs are kept so no fix has an empty set. The radius and cap come
   from a sweep on train only (`outputs/eda/candidate_sweep_train.*`).
4. **Features.** 21 per (fix, candidate): distance (absolute and relative to the fix's
   other candidates), heading agreement (separates the two directions of a street),
   motion (speed, turning, time gap, jump speed), arc topology, and *support*: how close
   neighbouring fixes are to the same arc. No feature uses the route.
5. **Models.** Logistic regression, Gaussian NB, distance-weighted KNN, decision tree,
   random forest, extra trees, histogram gradient boosting and an MLP with epoch-wise
   train/validation loss. Each predicts P(candidate is the true arc).
6. **Evaluation.** 5-fold GroupKFold on train trips (one trip never spans folds), then a
   refit on all train rows and scoring on student_test. The script reports binary metrics,
   observation-level top-1, candidate miss rate, route continuity, route-F1 and timings.

## Leakage rules followed

* The evaluator-test partition is not part of this repo and is never read.
* student_test is only predicted on, never fitted. Radius, cap and label threshold were
  chosen on train only.
* Features never use the reference route. It is used only to build training labels and
  to score.
