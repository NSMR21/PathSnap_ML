"""Step 7 - export everything the API needs into artifacts/.

artifacts/candidate_model.joblib   tuned Gradient Boosting (scripts/05, 05b)
artifacts/pipeline.json            feature list, candidate + geometry settings, HMM
                                   parameters (scripts/06), library versions and the
                                   development results the model was selected on

Run:  python scripts/07_export_artifacts.py
"""
import json
import platform
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy
import sklearn

from pathsnap import config
from pathsnap.features import FEATURE_COLUMNS

ART = config.ROOT / "artifacts"


def main():
    ART.mkdir(exist_ok=True)
    shutil.copy2(config.MODEL_DIR / "hgb_tuned.joblib", ART / "candidate_model.joblib")
    tuning = json.loads((config.OUT_DIR / "tuning" / "best_params.json").read_text())
    hmm = json.loads((config.OUT_DIR / "hmm" / "hmm_params.json").read_text())
    dev = pd.read_csv(config.OUT_DIR / "hmm" / "classifier_vs_hmm.csv")
    pipeline = {
        "created": date.today().isoformat(),
        "model": {"file": "candidate_model.joblib", "type": "sklearn HistGradientBoostingClassifier",
                  "params": tuning["params"], "selected_by": "Optuna TPE, CV top-1 on train trips",
                  "trained_on": "80 train trips, labelled fixes, every 2nd fix", "seed": config.SEED},
        "features": FEATURE_COLUMNS,
        "candidates": {"radius_m": config.CANDIDATE_RADIUS_M, "max_candidates": config.MAX_CANDIDATES,
                       "fallback_candidates": config.FALLBACK_CANDIDATES},
        "geometry": {"index_step_m": config.INDEX_STEP_M, "index_margin_m": config.INDEX_MARGIN_M,
                     "heading_half_window": config.HEADING_HALF_WINDOW,
                     "support_offsets": list(config.SUPPORT_OFFSETS), "support_clip_m": config.SUPPORT_CLIP_M},
        "scoring": {"label_max_dist_m": config.LABEL_MAX_DIST_M},
        "hmm": {"params": hmm["params"], "chosen_on": hmm["chosen_on"]},
        "development_results": dev.to_dict("records"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
    }
    (ART / "pipeline.json").write_text(json.dumps(pipeline, indent=2))
    print(json.dumps({k: pipeline[k] for k in ("model", "candidates", "hmm", "versions")}, indent=2))


if __name__ == "__main__":
    main()
