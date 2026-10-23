"""Publication output must retain complete evidence and honest plot geometry."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("manuscript_build_tests", ROOT / "docs/paper/build.py")
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)
import paper_assets as assets


@pytest.mark.parametrize("phase,quick,error", [
    ("started", False, "run has not completed"),
    ("complete", True, "quick runs cannot support manuscript results"),
])
def test_default_build_rejects_incomplete_or_smoke_evidence_before_writing(tmp_path, phase, quick, error):
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    (run / "reports/run_manifest.json").write_text(json.dumps({
        "phase": phase, "quick": quick, "tracked_source_dirty": False}))
    output = tmp_path / "paper"
    result = subprocess.run([sys.executable, str(ROOT / "docs/paper/build.py"),
                             "--run", str(run), "--out", str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert error in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("flag", [False, 1, "true"])
def test_assets_require_an_explicit_verified_scientific_flag(tmp_path, flag):
    with pytest.raises(ValueError, match="verified scientific budgets"):
        assets.write_assets({"verification": {"scientific_budgets_checked": flag}}, {}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def publication_manifest(output):
    names = (*paper.ASSET_NAMES, "main.pdf")
    for name in names:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test artifact")
    manifest = {"schema_version": 1, "scientific_budgets_checked": True,
                "layout_fixture": False, "profiles": {"full": {"pdf": "main.pdf", "pages": 1}},
                "manuscript_source_sha256": {
                    str((paper.HERE / name).relative_to(ROOT)): paper.digest(paper.HERE / name)
                    for name in paper.SOURCE_NAMES},
                "artifacts_sha256": {name: paper.digest(output / name) for name in names}}
    (output / "manuscript_manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("alteration,error", [
    ("missing_artifact", "artifact membership"),
    ("missing_source", "source membership"),
    ("fixture", "layout or smoke"),
    ("changed_pdf", "content hash mismatch"),
    ("changed_statistics", "content hash mismatch"),
    ("profile_path", "invalid manuscript profile"),
])
def test_publication_verifier_rejects_incomplete_or_altered_manifests(tmp_path, alteration, error):
    manifest = publication_manifest(tmp_path)
    assert paper.verify_artifacts(tmp_path) == manifest
    if alteration == "missing_artifact":
        del manifest["artifacts_sha256"]["main.pdf"]
    elif alteration == "missing_source":
        manifest["manuscript_source_sha256"].pop(next(iter(manifest["manuscript_source_sha256"])))
    elif alteration == "fixture":
        manifest["layout_fixture"] = True
    elif alteration == "changed_pdf":
        (tmp_path / "main.pdf").write_bytes(b"altered")
    elif alteration == "changed_statistics":
        (tmp_path / "statistics.json").write_bytes(b"altered")
    else:
        manifest["profiles"]["full"]["pdf"] = "other.pdf"
    (tmp_path / "manuscript_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        paper.verify_artifacts(tmp_path)


def test_failed_compilation_does_not_replace_existing_publication(tmp_path, monkeypatch):
    existing = tmp_path / "main.pdf"
    existing.write_bytes(b"existing publication")
    monkeypatch.setattr(paper, "write_assets", lambda *args, **kwargs: None)
    def fail(*args):
        raise ValueError("failed compilation")
    monkeypatch.setattr(paper, "render_profiles", fail)
    with pytest.raises(ValueError, match="failed compilation"):
        paper.build_from_statistics({"verification": {"scientific_budgets_checked": True}},
                                    {}, tmp_path, ["full"])
    assert existing.read_bytes() == b"existing publication"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["main.pdf"]


def test_every_layout_profile_is_marked_and_inputs_share_one_article(tmp_path):
    paper.stage_sources(tmp_path, fixture=True)
    for name in paper.PROFILES.values():
        source = (tmp_path / name).read_text()
        assert source.startswith("\\def\\layoutfixture{1}\n")
        assert "article.tex}" in source
    article = (tmp_path / "article.tex").read_text()
    assert "LAYOUT CHECK: SYNTHETIC NUMBERS" in article
    assert "\\input{\\paperroot/metrics.tex}" in article


def test_source_changes_during_compilation_cannot_publish_old_pdf_with_new_hash(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "article.tex").write_text("original source")
    monkeypatch.setattr(paper, "HERE", source)
    monkeypatch.setattr(paper, "ROOT", tmp_path)
    monkeypatch.setattr(paper, "SOURCE_NAMES", ("article.tex",))
    monkeypatch.setattr(paper, "write_assets", lambda *args, **kwargs: None)
    def render(stage, targets):
        (stage / "main.pdf").write_bytes(b"compiled from original source")
        (source / "article.tex").write_text("modified source")
        return {"full": {"pdf": "main.pdf", "pages": 1}}
    monkeypatch.setattr(paper, "render_profiles", render)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source changed during compilation"):
        paper.build_from_statistics({"verification": {"scientific_budgets_checked": True}},
                                    {}, output, ["full"])
    assert not output.exists()


@pytest.mark.parametrize("current", [
    {"mean": 2., "resolved": True},
    {"mean": 1., "resolved": 1},
    {"mean": True, "resolved": True},
    {"mean": 1.},
    {"mean": float("nan"), "resolved": True},
])
def test_claim_comparison_rejects_changed_values_keys_and_boolean_types(current):
    with pytest.raises(ValueError, match="claim"):
        paper.compare_claims(current, {"mean": 1., "resolved": True})


def test_claim_comparison_retains_null_and_tolerates_only_roundoff():
    paper.compare_claims({"x": [None, 1. + 1e-10]}, {"x": [None, 1.]})
    with pytest.raises(ValueError, match="numerical claim"):
        paper.compare_claims({"x": [None, 1. + 1e-8]}, {"x": [None, 1.]})
    assert assets.number(None) == assets.interval(None) == "undefined"
    assert "-0.000" not in assets.number(-1e-8)
    with pytest.raises(ValueError, match="null"):
        assets.number(float("inf"))
    with pytest.raises(ValueError, match="invalid manuscript interval"):
        assets.interval([3., 2.])


def test_unresolved_legend_and_intervals_are_drawn_without_changing_their_bounds(tmp_path, monkeypatch):
    calibration = [{"signal_strength": strength, "rejection_rate_resolved": None,
                    "rejection_rate_ci95": None,
                    "rank_comparators": {kind: {
                        "rejection_rate": float(strength > 0),
                        "rejection_rate_ci95": ([0., .2] if strength == 0 else
                                                 [.8, np.nextafter(1., 0.)])}
                        for kind in ("greater", "two-sided")}}
                   for strength in (0., .2, 1.)]
    # A percentile interval need not contain the point estimate.
    intervention = {"mean_span": 1., "ci95": [2., 3.]}
    steering = [{"penalty": 0., "target_share_mean": .3, "target_shares": [.2, .4]},
                {"penalty": 20., "target_share_mean": .1, "target_shares": [.05, .15],
                 "return_difference": 1., "return_diff_ci95": [2., 3.]}]
    statistics = {"conditional_calibration": {"cases": calibration},
                  "state_interventions": {"cases": [{"case": "market", "interventions": {
                      mode: intervention for mode in
                      ("full_observation", "market_rows", "market_trajectories")}}]},
                  "steering": {"populations": {name: steering for name in
                      ("real_market", "learnable_synthetic")}}}
    figures = {}
    monkeypatch.setattr(assets, "save", lambda fig, path: figures.setdefault(path.stem, fig))
    try:
        with plt.rc_context():
            assets.scientific_figures(statistics, tmp_path)
        labels = [text.get_text() for text in figures["calibration"].axes[0].get_legend().get_texts()]
        assert labels == ["Legacy rule", "Upper rank", "Symmetric rank"]
        state = figures["state_interventions"].axes[0]
        np.testing.assert_array_equal(state.lines[0].get_xdata(), [2., 3.])
        np.testing.assert_array_equal(state.lines[2].get_xdata(), [1.])
        changes = figures["steering"].axes[1]
        assert any(np.array_equal(line.get_ydata(), [2., 3.]) for line in changes.lines)
        assert any(np.array_equal(line.get_ydata(), [1.]) for line in changes.lines)
    finally:
        for fig in figures.values():
            plt.close(fig)
