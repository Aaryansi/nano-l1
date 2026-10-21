"""Unresolved references cannot establish agreement or absence of rejection."""
import pytest

from nano_rl.explain.sanity import (INFORMATIVE, NOT_DISTINGUISHABLE, UNRESOLVED,
                                    summarize_verdicts)


@pytest.mark.parametrize("verdicts,agreement,all_rejected,none_rejected", [
    ([UNRESOLVED] * 3, None, None, None),
    ([INFORMATIVE, UNRESOLVED], None, None, None),
    ([NOT_DISTINGUISHABLE, UNRESOLVED], None, None, None),
    ([INFORMATIVE] * 3, True, True, False),
    ([NOT_DISTINGUISHABLE] * 3, True, False, True),
    ([INFORMATIVE, NOT_DISTINGUISHABLE], False, False, False),
])
def test_summary_preserves_three_states(verdicts, agreement, all_rejected, none_rejected):
    result = summarize_verdicts([{"verdict": v} for v in verdicts])
    assert result["agreement"] is agreement
    assert result["all_rejected"] is all_rejected
    assert result["none_rejected"] is none_rejected
    assert result["n_unresolved"] == verdicts.count(UNRESOLVED)
    assert result["n_rejected"] == verdicts.count(INFORMATIVE)
    assert result["n_not_rejected"] == verdicts.count(NOT_DISTINGUISHABLE)


@pytest.mark.parametrize("rows", [[], [{"verdict": "unknown"}]])
def test_invalid_summary_is_rejected(rows):
    with pytest.raises(ValueError):
        summarize_verdicts(rows)
