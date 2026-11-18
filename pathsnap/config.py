"""Central configuration for the PathSnap pipeline.

Everything that is a "decision" (radius, thresholds, seeds) lives here so the
experiment scripts and later the API all read the same numbers.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data can be moved; point PATHSNAP_DATA at the folder holding train/ and student_test/
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "materials" / "data" / "Student folder"))
PARTITIONS = {
    "train": DATA_DIR / "train",
    "student_test": DATA_DIR / "student_test",
}

OUT_DIR = ROOT / "outputs"
EDA_DIR = OUT_DIR / "eda"
FEATURE_DIR = OUT_DIR / "features"
MODEL_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
MAPS_DIR = OUT_DIR / "maps"

# Seed used everywhere (course code UE24CS352A -> 352)
SEED = 352

# ---------------------------------------------------------------- geometry
# Densification step for the arc spatial index (metres). A fix within r of a
# segment is guaranteed to be within r + step/2 of one of its sample points.
INDEX_STEP_M = 10.0
# Only arcs whose bounding box comes within this margin of the track bbox are indexed.
INDEX_MARGIN_M = 400.0

# ---------------------------------------------------------------- candidates
# Search radius around each GPS fix and the cap on candidates per fix.
# Chosen from the radius sweep in scripts/02_build_dataset.py (train partition only).
CANDIDATE_RADIUS_M = 50.0
MAX_CANDIDATES = 10
# If nothing lies inside the radius we still keep this many nearest arcs, so a fix
# never ends up with an empty candidate set (matters for the HMM later).
FALLBACK_CANDIDATES = 3

# ---------------------------------------------------------------- labels
# A fix whose aligned reference arc is further than this is treated as an
# unreliable label (GPS outlier or reference-route problem) and skipped in
# training / scoring. Checked against the distance histogram in the EDA.
LABEL_MAX_DIST_M = 50.0

# ---------------------------------------------------------------- motion features
HEADING_HALF_WINDOW = 2          # fixes either side used for heading/speed
SUPPORT_OFFSETS = (-6, -3, 3, 6)  # neighbouring fixes used for the "support" feature
SUPPORT_CLIP_M = 200.0
