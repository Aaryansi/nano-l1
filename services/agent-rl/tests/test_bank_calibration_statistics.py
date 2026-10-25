"""Bank reuse cannot turn dependent trials into independent precision."""
from itertools import product
from math import comb
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bank_calibration_statistics import METHODS, bounded_mean_interval, summarize_banks


def banks(trials=2, unresolved=False):
    return [{"bank": index, "cases": [{"signal_strength": 0., "trials": [
        {"methods": {method: {"flagged": False, "resolved": not unresolved} for method in METHODS}}
        for _ in range(trials)]}]} for index in range(4)]


def test_zero_flags_leave_positive_bank_level_uncertainty():
    lower, upper = bounded_mean_interval([0.] * 20)
    assert lower == 0.
    assert upper == pytest.approx(1 - .025 ** (1 / 20))
    assert upper > .16
    lo, hi = bounded_mean_interval([1.] * 20)
    assert hi == 1.
    assert lo == pytest.approx(1 - upper)


def test_more_trials_in_the_same_banks_do_not_invent_independent_units():
    a, b = [summarize_banks(banks(trials=n))["cases"][0]["methods"]["legacy"] for n in (2, 1000)]
    assert a["ci95_bounded_banks"] == b["ci95_bounded_banks"]
    assert a["n_independent_banks"] == b["n_independent_banks"] == 4
    assert a["n_trials"] == 8
    assert b["n_trials"] == 4000


def test_unresolved_is_reported_separately_from_flags_and_conditional_rates():
    result = summarize_banks(banks(unresolved=True))["cases"][0]["methods"]["legacy"]
    assert result["n_unresolved"] == result["n_trials"] == 8
    assert result["n_resolved"] == 0
    assert result["flag_fraction_resolved_descriptive"] is None
    assert result["mean"] == 0.
    assert result["unresolved"]["mean"] == 1.
    assert result["ci95_bounded_banks"][1] > 0.


def test_method_contrasts_pair_bank_fractions_and_keep_failures():
    data = banks()
    data[0]["cases"][0]["trials"][0]["methods"]["legacy"]["flagged"] = True
    data[1]["cases"][0]["trials"][1]["methods"]["greater"]["flagged"] = True
    contrast = summarize_banks(data)["cases"][0]["paired_contrasts"]["legacy_minus_greater"]
    assert contrast["bank_differences"] == [.5, -.5, 0., 0.]
    assert contrast["mean_difference"] == 0.
    assert contrast["ci95_bounded_banks"][0] < 0 < contrast["ci95_bounded_banks"][1]
    data[0]["cases"][0]["trials"][0]["methods"]["legacy"]["resolved"] = False
    with pytest.raises(ValueError, match="cannot issue flags"):
        summarize_banks(data)


@pytest.mark.parametrize("p", [0., .01, .05, .2, .5, .9, .99, 1.])
@pytest.mark.parametrize("n", [1, 5, 12])
def test_interval_coverage_by_exact_binomial_enumeration(p, n):
    failure = 0.
    for k in range(n + 1):
        lo, hi = bounded_mean_interval([1.] * k + [0.] * (n - k))
        if not lo <= p <= hi:
            failure += comb(n, k) * p ** k * (1 - p) ** (n - k)
    assert failure <= .05 + 1e-12


def test_bounded_bank_fractions_need_not_be_bernoulli():
    support, probabilities = [0., .2, 1.], [.2, .6, .2]
    mean = sum(x * p for x, p in zip(support, probabilities, strict=True))
    failure = 0.
    for indices in product(range(3), repeat=5):
        lo, hi = bounded_mean_interval([support[index] for index in indices])
        if not lo <= mean <= hi:
            failure += float(np.prod([probabilities[index] for index in indices]))
    assert failure <= .05 + 1e-12


@pytest.mark.parametrize("values", [[], [float("nan")], [-.1], [1.1], [[0., 1.]]])
def test_invalid_bank_averages_are_rejected(values):
    with pytest.raises(ValueError, match="bank averages"):
        bounded_mean_interval(values)
