"""Publication views must retain diagnostic states and task-specific values."""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("publication_figures", Path(__file__).resolve().parents[1] / "scripts/paper_figures.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def capture(monkeypatch):
    saved = []
    monkeypatch.setattr(figures, "save", lambda fig, path: saved.append(fig))
    return saved


def test_six_task_width_view_marks_zero_denominators(monkeypatch, tmp_path):
    saved = capture(monkeypatch)
    data = {"random_return_sd": [1., 0., 2., 3., 4., 0.],
            "weight_null_sd": [2., 0., 3., 1., 8., 1.],
            "env_ids": [f"Task{i}-v1" for i in range(6)]}
    figures.fig_conjecture(data, tmp_path)
    ax = saved[0].axes[0]
    assert len(ax.get_yticklabels()) == 6
    assert len([text for text in ax.texts if "undefined" in text.get_text()]) == 2
    np.testing.assert_allclose([patch.get_width() for patch in ax.patches], [2., 1.5, 1/3, 2.])
    plt.close(saved[0])


def test_signal_sweep_preserves_three_states_without_historical_market_constant(monkeypatch, tmp_path):
    saved = capture(monkeypatch)
    rows = [{"agent_edge": edge, "z": z, "detected": rejected, "verdict": verdict}
            for edge, z, rejected, verdict in ((0., .2, False, "not distinguishable from null"),
                                               (4., 4., True, "informative"),
                                               (8., 8., False, "unresolved"))]
    figures.fig_power({"power_curve": rows}, tmp_path)
    ax = saved[0].axes[0]
    assert [item.get_label() for item in ax.collections] == ["outside reference", "not rejected", "unresolved"]
    assert all(len(item.get_offsets()) == 1 for item in ax.collections)
    assert len(ax.lines) == 1
    assert not any("real agent" in text.get_text() or "nothing" in text.get_text() for text in ax.texts)
    plt.close(saved[0])


def test_constructions_use_each_observation_and_preserve_unresolved(monkeypatch, tmp_path):
    saved = capture(monkeypatch)
    def data(observed):
        return {"cases": [{"case": name, "span": observed, "null_spans": [0., 1., 2.],
                           "result": {"verdict": "UNRESOLVED", "passes": False}}
                          for name in ("planted signal", "null corpus", "real market")]}
    figures.fig_constructions(data(10.), data(20.), tmp_path)
    for ax in saved[0].axes:
        observations = [item.get_offsets()[0, 0] for item in ax.collections if len(item.get_offsets()) == 1]
        assert observations == [20., 10.]
        assert [text.get_text() for text in ax.texts] == ["unresolved", "unresolved"]
    plt.close(saved[0])


@pytest.mark.parametrize("z", [float("inf"), 0.])
def test_zero_width_reference_is_labeled_undefined_not_infinite_evidence(monkeypatch, tmp_path, z):
    saved = capture(monkeypatch)
    block = {"spans": [0., 0.], "result": {"z_score": z,
             "verdict": "UNRESOLVED", "passes": False}}
    figures.fig_nullchoice({"observed_span": 4., "null_synthetic": block,
                            "null_blinded_real": block}, tmp_path)
    labels = [text.get_text() for text in saved[0].axes[0].texts]
    assert labels.count("z undefined   unresolved") == 2
    assert not any("inf" in label for label in labels)
    plt.close(saved[0])
