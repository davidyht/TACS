import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rebuttal"))
import summarize_specificity_calibration as ssc  # noqa: E402


def _rows(lrs, depths, folds=(1, 2, 3)):
    # Specificity grows with depth at lr 1e-5 and is flat elsewhere, so the pick is (1e-5, max depth).
    return [{"fold": f, "lr": lr, "depth": d, "spec_train": 0.1 * d if lr == "1e-5" else 0.0}
            for f in folds for lr in lrs for d in depths]


def test_expect_grid_accepts_complete_grid():
    grid = {"bbh": (["1e-5", "2e-5"], [2, 4])}
    _, raw = ssc.summarize({"bbh": _rows(["1e-5", "2e-5"], [2, 4])}, 3, expect_grid=grid)
    assert (raw["bbh"]["lr"], raw["bbh"]["depth"]) == ("1e-5", 4)


def test_expect_grid_refuses_missing_learning_rate():
    grid = {"bbh": (["1e-5", "2e-5"], [2, 4])}
    with pytest.raises(ssc.CalibrationError, match="missing cells"):
        ssc.summarize({"bbh": _rows(["1e-5"], [2, 4])}, 3, expect_grid=grid)


def test_expect_grid_refuses_unexpected_depth():
    grid = {"tydiqa": (["1e-5"], [2, 4])}
    with pytest.raises(ssc.CalibrationError, match="unexpected cells"):
        ssc.summarize({"tydiqa": _rows(["1e-5"], [2, 4, 128])}, 3, expect_grid=grid)
