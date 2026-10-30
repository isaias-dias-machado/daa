"""Project-wide paths and the single random seed.

Every experiment imports ``SEED`` from here. Do not pass ``random_state=None``
anywhere: comparable results depend on one seed and one set of folds.

Paths can be overridden with environment variables, which is useful for tests
and for running notebooks against a temporary output directory:

- ``DAA_DATA_DIR``
- ``DAA_OUTPUT_DIR``
- ``DAA_SPLITS_PATH``
"""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DAA_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("DAA_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
RUNS_DIR = OUTPUTS_DIR / "runs"
SPLITS_PATH = Path(os.environ.get("DAA_SPLITS_PATH", PROJECT_ROOT / "splits" / "folds.npz"))

TRAIN_PATH = DATA_DIR / "dataset_train.csv"
TEST_PATH = DATA_DIR / "dataset_test_X.csv"
