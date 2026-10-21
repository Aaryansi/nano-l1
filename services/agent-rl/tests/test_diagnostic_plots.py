"""Plot geometry must not create diagnostic decisions or invent a scale."""
import matplotlib.pyplot as plt
import numpy as np

from nano_rl import plots
from nano_rl.explain.sanity import test_span_against_null as compare


def test_small_reference_stays_unresolved_even_outside_sample_range(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(plots, "_save", lambda fig, path: saved.append(fig))
    result = compare(100., [0., 1.]).as_dict()
    plots.null_test({"example": result}, np.array([0., 1.]), tmp_path / "plot.png")
    labels = [text.get_text() for text in saved[0].axes[0].texts]
    assert any("unresolved" in label for label in labels)
    assert not any("informative" in label for label in labels)
    plt.close(saved[0])


def test_scheme_view_does_not_replace_zero_sd_with_one(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(plots, "_save", lambda fig, path: saved.append(fig))
    plots.scheme_comparison({"example": (np.zeros(3), 100., 1.)}, tmp_path / "plot.png")
    ax = saved[0].axes[0]
    assert any("standardized span undefined" in text.get_text() for text in ax.texts)
    assert len(ax.collections) == 0
    plt.close(saved[0])


def test_scheme_legend_survives_a_degenerate_first_row(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(plots, "_save", lambda fig, path: saved.append(fig))
    plots.scheme_comparison({"zero": (np.zeros(3), 100., 1.),
                             "variable": (np.arange(3.), 4., 2.)}, tmp_path / "plot.png")
    assert saved[0].axes[0].get_legend_handles_labels()[1] == ["planted signal", "real market"]
    plt.close(saved[0])
