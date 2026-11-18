# PathSnap: map-matching noisy GPS tracks to a road network

UE24CS352A Machine Learning hackathon (PES University).

## Status

| Stage | State |
|---|---|
| 1-2. Problem + raw data exploration | done (`scripts/01_explore_dataset.py`) |
| 3. Pre-processing, labels, candidates, features | done (`scripts/02_build_dataset.py`) |
| 4-5. Train + compare 8 models (grouped CV + student_test) | done (`scripts/03_train_compare_models.py`) |
| Classifier-only route maps | done (`scripts/04_map_classifier_route.py`) |
| 6. Bayesian hyper-parameter optimisation | **in progress**: `scripts/05_bayesian_tuning.py` written, search results not committed yet |
| 7. HMM + Viterbi route reconstruction | **in progress**: decoder in `pathsnap/hmm.py`, experiment script next |
| 8. Export final artifacts | **next** |
| 9-10. FastAPI backend (laptop 1) + evaluator UI (laptop 2) | **next** |
| 12. Technical report | **next** |

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

`03` accepts `--models rf,hgb` and `--folds 3` for quicker partial runs.

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
