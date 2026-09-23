"""every null in the pipeline must be large enough to decide at alpha.

a rank test with n reference draws cannot return a p-value below 1/(n+1). if
that floor sits above alpha the comparison is not a weak test, it is not a test
at all: no statistic whatsoever can be called informative against it.

this is pinned as a test because the failure is invisible at runtime. three
pipeline steps ran at n=12 (floor 0.0769) against alpha=0.05 for weeks. they
completed, wrote well-formed json, and reported "not distinguishable from null"
for observations sitting 89 standard deviations outside their reference. the
paper then recorded those as negative results.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nano_rl.explain.sanity import check_resolving_power, min_nulls_for

ALPHA = 0.05
REPRODUCE = Path(__file__).resolve().parents[3] / "reproduce.sh"

# scripts that take --n-null but never run a rank test against it.
# horizon_scaling uses its draws to estimate the WIDTH of two null
# distributions and compares those widths, so the rank floor does not apply.
NO_RANK_TEST = {"horizon_scaling"}


def pipeline_budgets() -> dict[str, int]:
    """the full-run --n-null for every script reproduce.sh invokes."""
    text = REPRODUCE.read_text()
    default = int(re.search(r"^N_NULL=(\d+)", text, re.M).group(1))

    budgets: dict[str, int] = {}

    def note(name: str, n: int) -> None:
        # a script can be invoked more than once (the headline null test runs
        # at 24 and again at 99). keep the smallest, so the check is about the
        # weakest reference the pipeline actually uses.
        budgets[name] = min(budgets.get(name, n), n)

    for chunk in text.split('"$PY" scripts/')[1:]:
        name = chunk.split(".py")[0]
        # one invocation ends where the subshell closes
        cmd = chunk.split("\n\n")[0]
        m = re.search(r'--n-null\s+"\$N_NULL"', cmd)
        if m:
            note(name, default)
            continue
        # --n-null $([ "$QUICK" = 1 ] && echo 6 || echo 24): take the full-run arm
        m = re.search(r'--n-null\s+\$\(\[ "\$QUICK" = 1 \] && echo \d+ \|\| echo (\d+)\)', cmd)
        if m:
            note(name, int(m.group(1)))
    return budgets


class TestTheFloorIsDocumented:
    def test_alpha_of_five_percent_needs_nineteen_draws(self):
        assert min_nulls_for(0.05) == 19
        assert 1.0 / (19 + 1) <= 0.05

    def test_eighteen_draws_is_not_enough(self):
        assert 1.0 / (18 + 1) > 0.05


class TestTheGuardRefuses:
    @pytest.mark.parametrize("n", [2, 8, 12, 16, 18])
    def test_a_reference_below_the_floor_is_a_configuration_error(self, n, monkeypatch):
        # explicitly cleared: reproduce.sh --quick exports this, and a test that
        # silently changes meaning with the ambient environment is not a test.
        monkeypatch.delenv("ALLOW_UNRESOLVED_NULLS", raising=False)
        with pytest.raises(SystemExit, match="resolving power"):
            check_resolving_power(n, ALPHA)

    @pytest.mark.parametrize("n", [19, 20, 24, 32, 99])
    def test_a_reference_at_or_above_the_floor_is_accepted(self, n, monkeypatch):
        monkeypatch.delenv("ALLOW_UNRESOLVED_NULLS", raising=False)
        check_resolving_power(n, ALPHA)

    def test_the_smoke_test_can_opt_out_explicitly(self, monkeypatch):
        monkeypatch.setenv("ALLOW_UNRESOLVED_NULLS", "1")
        check_resolving_power(4, ALPHA)  # warns, does not raise

    def test_opting_out_requires_exactly_that_value(self, monkeypatch):
        monkeypatch.setenv("ALLOW_UNRESOLVED_NULLS", "yes")
        with pytest.raises(SystemExit):
            check_resolving_power(4, ALPHA)


class TestThePipelineCanDecide:
    def test_reproduce_sh_was_found(self):
        assert REPRODUCE.exists()
        assert pipeline_budgets(), "parsed no budgets, the parser has drifted"

    def test_every_rank_test_budget_can_resolve_alpha(self):
        need = min_nulls_for(ALPHA)
        too_small = {
            name: n
            for name, n in pipeline_budgets().items()
            if name not in NO_RANK_TEST and n < need
        }
        assert not too_small, (
            f"these steps cannot return INFORMATIVE at alpha={ALPHA} no matter "
            f"what the data says, because their rank floor exceeds it: "
            f"{too_small}. at least {need} draws are needed."
        )


# ---------------------------------------------------------------------------
# the artifacts record the budget they were built with. the pipeline declares
# the budget it passes. nothing checked that these agreed, and for a long time
# they did not: three steps were invoked with a budget different from the one
# that produced the committed artifact, so running the documented single command
# would have silently produced different numbers from the ones the paper
# reports. that is a reproducibility failure invisible to every other test here.

REPORTS = Path(__file__).resolve().parents[3] / "reports"

# which artifact each step writes, and how to dig the recorded budget out of it
ARTIFACTS = {
    "sanity_check_explanations": ("sanity_test.json", None),
    "matched_null_test": ("matched_null_test.json", None),
    "permuted_null_test": ("permuted_null_test.json", None),
    "sverl_targets": ("sverl_targets.json", None),
    "null_budget_check": ("null_budget_check.json", None),
    "positive_control": ("positive_control.json", None),
    "null_corpus_check": ("null_corpus_check.json", None),
    "manifold_masking": ("manifold_masking.json", None),
    "scheme_robustness": ("scheme_robustness.json", None),
    "second_method": ("second_method.json", None),
}


def recorded_budgets(obj) -> set[int]:
    """every n_null_samples anywhere in an artifact."""
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "n_null_samples" and isinstance(v, int):
                found.add(v)
            else:
                found |= recorded_budgets(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= recorded_budgets(v)
    return found


@pytest.mark.skipif(not REPORTS.exists(), reason="no artifacts in this checkout")
@pytest.mark.parametrize("step", sorted(ARTIFACTS))
def test_artifact_budget_matches_the_pipeline(step):
    import json

    name, _ = ARTIFACTS[step]
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{name} not built")

    declared = pipeline_budgets().get(step)
    assert declared is not None, f"reproduce.sh does not invoke {step}"

    found = recorded_budgets(json.loads(path.read_text()))
    if not found:
        pytest.skip(f"{name} records no n_null_samples")

    assert found == {declared}, (
        f"{name} was built with n_null={sorted(found)} but reproduce.sh passes "
        f"{declared} to {step}.py. one of the two is wrong, and until they "
        f"agree the documented pipeline does not regenerate this artifact."
    )


@pytest.mark.skipif(not REPORTS.exists(), reason="no artifacts in this checkout")
def test_the_gym_sweep_budget_matches_the_pipeline():
    import json

    path = REPORTS / "generalize_gym.json"
    if not path.exists():
        pytest.skip("generalize_gym.json not built")
    declared = pipeline_budgets()["generalize_gym"]
    drawn = {len(r["env_null"]["spans"]) for r in json.loads(path.read_text())}
    assert drawn == {declared}, (
        f"generalize_gym.json holds {sorted(drawn)} environment-null draws but "
        f"reproduce.sh passes {declared}"
    )
