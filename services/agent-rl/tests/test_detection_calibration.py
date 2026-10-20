"""A reference that cannot decide is not evidence of specificity or zero power."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("calibrate_detection", SCRIPTS / "calibrate_detection.py")
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)
from nano_rl.explain.sanity import UNRESOLVED, INFORMATIVE, NOT_DISTINGUISHABLE


def row(verdict, z=0.):
    return {"result": {"verdict": verdict, "passes": verdict == INFORMATIVE, "z_score": z}}


def test_unresolved_trials_do_not_count_as_successful_null_checks():
    result = calibration.summarize([row(UNRESOLVED)] * 5)
    assert result["n_unresolved"] == 5
    assert result["rejection_rate_resolved"] is None
    assert result["rejection_rate_ci95"] is None


def test_positive_detection_is_distinct_from_two_sided_rejection():
    result = calibration.summarize([row(INFORMATIVE, 5.), row(INFORMATIVE, -5.),
                                     row(NOT_DISTINGUISHABLE)])
    assert result["rejection_rate_resolved"] == pytest.approx(2 / 3)
    assert result["positive_detection_rate_resolved"] == pytest.approx(1 / 3)


def test_zero_empirical_errors_does_not_claim_zero_population_error():
    lo, hi = calibration.wilson_interval(0, 30)
    assert lo == pytest.approx(0., abs=1e-15)
    assert .10 < hi < .12
