"""figures for docs/REPORT.md.

design constraints applied throughout, rather than left to per-plot taste:

  - categorical hues are assigned in fixed order and never cycled. the three
    used here validated all-pairs in light mode (worst CVD deltaE 9.2, worst
    normal-vision deltaE 24.0), so series stay distinguishable under deuteran
    and tritan vision.
  - one y-axis per panel, never two. two measures of different scale get two
    panels.
  - lines are direct-labelled as well as legended, so identity is never carried
    by colour alone. this also discharges the contrast relief rule for the aqua
    slot, which sits below 3:1 on a light surface.
  - grid and axes are recessive; text wears ink colours, never the series
    colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# validated categorical slots (light mode, all-pairs)
C1 = "#2a78d6"  # blue
C2 = "#eb6834"  # orange
C3 = "#1baf7a"  # aqua

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

# a fourth, deliberately neutral, for reference lines that are not a series
REFERENCE = "#9a9894"


def _style_axes(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    pad: float = 10,
) -> None:
    """recessive chrome: the data should be the only assertive thing."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
        ax.spines[spine].set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=pad)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)


def _new_fig(w: float = 8.0, h: float = 4.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    return fig, ax


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def equity_curves(
    series: dict[str, np.ndarray],
    path: Path,
    title: str = "cumulative pnl on the held-out test split",
    band: tuple[np.ndarray, np.ndarray] | None = None,
    band_label: str = "",
) -> None:
    """cumulative pnl per episode, at most three series.

    args:
        series: name -> per-episode pnl array. cumulative sums are taken here
            so callers cannot accidentally pass one already cumulated.
        band: optional (lower, upper) cumulative band, for across-seed spread.
    """
    fig, ax = _new_fig()
    colors = [C1, C2, C3]

    if band is not None:
        ax.fill_between(
            np.arange(len(band[0])),
            band[0],
            band[1],
            color=C1,
            alpha=0.15,
            linewidth=0,
            label=band_label or None,
        )

    for i, (name, pnl) in enumerate(series.items()):
        cum = np.cumsum(pnl)
        x = np.arange(len(cum))
        ax.plot(x, cum, color=colors[i % 3], linewidth=2.0, label=name, zorder=3)
        # direct label at the line end, so identity is not colour-alone
        ax.annotate(
            f"{name}  {cum[-1]:+,.0f}",
            xy=(x[-1], cum[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            va="center",
            fontweight="medium",
        )

    ax.axhline(0.0, color=REFERENCE, linewidth=1.2, linestyle="--", zorder=2)
    _style_axes(ax, title, "test episode", "cumulative pnl (dollars)")
    # legend below the axes: an in-plot legend collided with the series here,
    # and the lines are already direct-labelled, so the box only needs to
    # exist, not to compete for space.
    ax.legend(
        frameon=False, fontsize=9, labelcolor=INK_MUTED,
        loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3,
    )
    # headroom for the end labels
    ax.set_xlim(0, len(next(iter(series.values()))) * 1.30)
    _save(fig, path)


def policy_comparison(
    names: list[str],
    means: list[float],
    errs: list[float],
    path: Path,
    title: str = "mean pnl per episode, test split",
) -> None:
    """horizontal bars with error bars. one measure, so one axis.

    bars are coloured by sign rather than by identity: this is a magnitude
    chart with a meaningful zero, not a categorical one, so a status-style
    encoding is the honest choice and the names carry identity.
    """
    fig, ax = _new_fig(8.0, 0.55 * len(names) + 1.8)

    y = np.arange(len(names))
    colors = [C1 if m >= 0 else C2 for m in means]

    ax.barh(y, means, height=0.62, color=colors, zorder=3)
    ax.errorbar(
        means, y, xerr=errs, fmt="none", ecolor=INK_MUTED,
        elinewidth=1.2, capsize=3, zorder=4,
    )

    for yi, (m, e) in enumerate(zip(means, errs)):
        offset = 6 if m >= 0 else -6
        ax.annotate(
            f"{m:+.3f}" + (f" ± {e:.3f}" if e > 0 else ""),
            xy=(m, yi),
            xytext=(offset, 0),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            va="center",
            ha="left" if m >= 0 else "right",
        )

    ax.axvline(0.0, color=REFERENCE, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK, fontsize=9)
    ax.invert_yaxis()
    _style_axes(ax, title, "mean pnl per episode (dollars)", "")
    ax.grid(axis="y", visible=False)
    span = max(max(np.abs(means)) if len(means) else 1.0, 1e-6)
    ax.set_xlim(-span * 1.45, span * 1.45)
    _save(fig, path)


def learning_curves(logs: list[dict], path: Path) -> None:
    """three panels: return, entropy, critic explained variance.

    three separate panels rather than three lines on shared axes, because the
    quantities have unrelated scales and a dual axis would be a lie.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), facecolor=SURFACE)

    panels = [
        ("mean_return", "episode return during training", "dollars", C1),
        ("entropy", "policy entropy", "nats", C2),
        ("explained_var", "critic explained variance", "fraction", C3),
    ]

    for ax, (key, title, ylabel, color) in zip(axes, panels):
        stacked = np.array([lg[key] for lg in logs], dtype=float)
        x = np.arange(stacked.shape[1])
        # individual seeds, recessive; the mean carries the message
        for row in stacked:
            ax.plot(x, row, color=color, linewidth=0.8, alpha=0.28, zorder=2)
        ax.plot(x, stacked.mean(axis=0), color=color, linewidth=2.0, zorder=3)

        if key == "entropy":
            ax.axhline(
                np.log(3), color=REFERENCE, linewidth=1.2, linestyle="--", zorder=2
            )
            ax.annotate(
                "uniform (ln 3)",
                xy=(x[-1], np.log(3)),
                xytext=(-4, 5),
                textcoords="offset points",
                color=INK_MUTED,
                fontsize=8,
                ha="right",
            )
        if key in ("mean_return", "explained_var"):
            ax.axhline(0.0, color=REFERENCE, linewidth=1.2, linestyle="--", zorder=2)

        _style_axes(ax, title, "ppo update", ylabel)

    fig.suptitle(
        f"training diagnostics, {len(logs)} seeds (thin lines) and their mean (thick)",
        color=INK,
        fontsize=11,
        x=0.008,
        ha="left",
    )
    _save(fig, path)


def value_calibration(
    bins: list[tuple[float, float, int]],
    path: Path,
    title: str = "critic calibration against realised settlement",
    subtitle: str = "",
) -> None:
    """reliability diagram: predicted value against realised frequency.

    this figure is only possible because every episode resolves to a known 0/1
    (docs/MDP.md section 1.2). the diagonal is perfect calibration.
    """
    fig, ax = _new_fig(6.2, 5.4)

    ax.plot(
        [0, 1], [0, 1], color=REFERENCE, linewidth=1.4, linestyle="--", zorder=2,
    )
    ax.annotate(
        "perfect calibration",
        xy=(0.62, 0.62),
        xytext=(4, -14),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=8.5,
        rotation=38,
    )

    if bins:
        pred = [b[0] for b in bins]
        real = [b[1] for b in bins]
        n = np.array([b[2] for b in bins], dtype=float)
        # marker area carries sample count; minimum 8px per the mark spec
        sizes = 40 + 320 * (n / n.max())
        ax.plot(pred, real, color=C1, linewidth=2.0, zorder=3)
        ax.scatter(
            pred, real, s=sizes, color=C1, zorder=4,
            edgecolors=SURFACE, linewidths=2.0,
        )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    _style_axes(
        ax,
        title,
        "predicted probability of yes",
        "realised frequency of yes",
        pad=26 if subtitle else 10,
    )
    if subtitle:
        # sits under the title, not on top of it
        ax.annotate(
            subtitle,
            xy=(0, 1.035),
            xycoords="axes fraction",
            color=INK_MUTED,
            fontsize=9,
            va="bottom",
        )
    _save(fig, path)


def cost_ablation(
    with_costs: np.ndarray,
    without_costs: np.ndarray,
    path: Path,
    title: str = "what frictions cost each policy",
    labels: list[str] | None = None,
) -> None:
    """paired bars: the same policies with frictions on and off.

    this is the figure that separates "cannot predict" from "predicts but
    cannot cover costs" (docs/MDP.md section 9.4).
    """
    fig, ax = _new_fig(8.0, 0.62 * len(with_costs) + 1.8)
    y = np.arange(len(with_costs))
    h = 0.36
    # 2px surface gap between adjacent bars, per the mark spec
    ax.barh(y - h / 2 - 0.012, without_costs, height=h, color=C3,
            label="frictionless", zorder=3)
    ax.barh(y + h / 2 + 0.012, with_costs, height=h, color=C1,
            label="with fees and spread", zorder=3)

    ax.axvline(0.0, color=REFERENCE, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels or [f"policy {i}" for i in y], color=INK, fontsize=9)
    ax.invert_yaxis()
    _style_axes(ax, title, "mean pnl per episode (dollars)", "")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="lower right")
    _save(fig, path)


def attribution_bar(
    names: list[str],
    values: np.ndarray,
    path: Path,
    title: str,
    stderr: np.ndarray | None = None,
    subtitle: str = "",
    top_k: int = 12,
) -> None:
    """shapley attributions, largest absolute contribution first.

    signed, so a diverging encoding is correct here: one hue for positive
    contributions, one for negative, and the zero line carries meaning.
    """
    values = np.asarray(values)
    order = np.argsort(-np.abs(values))[:top_k][::-1]
    v = values[order]
    labels = [names[i] for i in order]
    err = np.asarray(stderr)[order] if stderr is not None else None

    fig, ax = _new_fig(8.0, 0.42 * len(v) + 1.9)
    y = np.arange(len(v))
    ax.barh(y, v, height=0.66, color=[C1 if x >= 0 else C2 for x in v], zorder=3)
    if err is not None and np.any(err > 0):
        ax.errorbar(v, y, xerr=err, fmt="none", ecolor=INK_MUTED,
                    elinewidth=1.1, capsize=2.5, zorder=4)

    ax.axvline(0.0, color=REFERENCE, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    _style_axes(ax, title, "shapley value", "", pad=26 if subtitle else 10)
    ax.grid(axis="y", visible=False)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.035), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=9, va="bottom")
    span = max(np.abs(v).max(), 1e-9)
    ax.set_xlim(-span * 1.35, span * 1.35)
    _save(fig, path)


def attribution_comparison(
    names: list[str],
    naive: np.ndarray,
    trajectory: np.ndarray,
    path: Path,
    title: str = "per-decision attribution versus trajectory-aware attribution",
    subtitle: str = "",
    top_k: int = 10,
) -> None:
    """paired bars contrasting the two characteristic functions.

    both series are normalised to their own total absolute mass, because they
    are measured in different units (action probability against dollars of
    return). the comparison of interest is which features are credited and in
    what proportion, not the raw magnitudes, and putting two unit systems on
    one axis without normalising would be the dual-axis mistake in disguise.
    """
    naive = np.asarray(naive, dtype=float)
    trajectory = np.asarray(trajectory, dtype=float)

    def share(x):
        total = np.abs(x).sum()
        return x / total if total > 1e-12 else x

    a, b = share(naive), share(trajectory)
    order = np.argsort(-(np.abs(a) + np.abs(b)))[:top_k][::-1]

    fig, ax = _new_fig(8.6, 0.52 * len(order) + 2.1)
    y = np.arange(len(order))
    h = 0.36
    ax.barh(y - h / 2 - 0.012, a[order], height=h, color=C2,
            label="per-decision, pi(a|s)", zorder=3)
    ax.barh(y + h / 2 + 0.012, b[order], height=h, color=C1,
            label="trajectory-aware, episode return", zorder=3)

    ax.axvline(0.0, color=REFERENCE, linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], color=INK, fontsize=9)
    _style_axes(ax, title, "share of total attribution mass", "",
                pad=26 if subtitle else 10)
    ax.grid(axis="y", visible=False)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.035), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=9, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED,
              loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=2)
    _save(fig, path)


def deletion_curves(
    curves: dict[str, np.ndarray],
    path: Path,
    title: str = "faithfulness: return after removing the top-k ranked features",
    subtitle: str = "",
    ylabel: str = "mean episode return (dollars)",
) -> None:
    """deletion curve, one line per feature ranking.

    features are removed in the order a ranking considers most important. a
    ranking that identifies genuinely load-bearing features degrades
    performance FASTER, so the lower curve is the better ranking. a random
    ranking is included as the control that makes the comparison meaningful.
    """
    fig, ax = _new_fig(8.2, 4.8)
    colors = [C1, C2, C3]

    # curves that converge would overprint their end labels, so stagger any
    # labels whose endpoints sit within a few percent of the axis range.
    ends = [ys[-1] for ys in curves.values()]
    span = (max(ends) - min(ends)) or 1.0
    offsets = []
    for i, e in enumerate(ends):
        clash = sum(1 for j, o in enumerate(ends[:i]) if abs(o - e) < 0.06 * span)
        offsets.append(11 * clash - (5 if clash else 0))

    for i, (name, ys) in enumerate(curves.items()):
        x = np.arange(len(ys))
        ax.plot(x, ys, color=colors[i % 3], linewidth=2.0, marker="o",
                markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=name, zorder=3)
        ax.annotate(
            name,
            xy=(x[-1], ys[-1]),
            xytext=(8, offsets[i]),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            va="center",
        )

    ax.axhline(0.0, color=REFERENCE, linewidth=1.2, linestyle="--", zorder=2)
    _style_axes(ax, title, "features removed (most important first)", ylabel,
                pad=26 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.035), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=9, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED,
              loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3)
    n = len(next(iter(curves.values())))
    ax.set_xlim(-0.4, n * 1.28)
    _save(fig, path)


def attribution_stability(
    names: list[str],
    behaviour: np.ndarray,
    outcomes: np.ndarray,
    path: Path,
    title: str = "do different seeds explain themselves the same way?",
    subtitle: str = "",
    top_k: int = 10,
) -> None:
    """per-feature spread of attribution across seeds, for two targets.

    each row shows the range across seeds as a bar and the individual seeds as
    dots. a tight cluster means the explanation is a property of the task; a
    wide spread means it is a property of the seed.

    both blocks are normalised to their own total mass per seed, since the two
    targets are measured in different units.
    """
    def norm(mat: np.ndarray) -> np.ndarray:
        totals = np.abs(mat).sum(axis=1, keepdims=True)
        totals[totals < 1e-12] = 1.0
        return mat / totals

    b, o = norm(np.asarray(behaviour)), norm(np.asarray(outcomes))
    order = np.argsort(-(np.abs(b).mean(0) + np.abs(o).mean(0)))[:top_k][::-1]

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 0.42 * len(order) + 2.2),
        facecolor=SURFACE, sharey=True,
    )

    for ax, mat, color, label in (
        (axes[0], b, C2, "behaviour, pi(a|s)"),
        (axes[1], o, C1, "outcomes, episode return"),
    ):
        y = np.arange(len(order))
        for row, feat in enumerate(order):
            vals = mat[:, feat]
            ax.plot([vals.min(), vals.max()], [row, row],
                    color=color, linewidth=6, alpha=0.30, solid_capstyle="round",
                    zorder=2)
            ax.scatter(vals, np.full_like(vals, row), s=26, color=color,
                       edgecolors=SURFACE, linewidths=1.2, zorder=4)
        ax.axvline(0.0, color=REFERENCE, linewidth=1.2, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([names[i] for i in order], color=INK, fontsize=9)
        _style_axes(ax, label, "share of attribution mass", "")
        ax.grid(axis="y", visible=False)

    fig.suptitle(title, color=INK, fontsize=11, x=0.008, ha="left", y=1.02)
    if subtitle:
        fig.text(0.008, 0.985, subtitle, color=INK_MUTED, fontsize=9, ha="left")
    _save(fig, path)


def null_test(
    cases: dict[str, tuple[float, float]],
    null_samples: np.ndarray,
    path: Path,
    title: str = "is this explanation distinguishable from an explanation of nothing?",
    subtitle: str = "",
) -> None:
    """observed attribution spans against the null distribution.

    the null band is the reference; a case inside it is an explanation that
    could have been produced by an agent with nothing to learn.
    """
    fig, ax = _new_fig(8.6, 0.62 * len(cases) + 2.6)

    lo, hi = float(np.min(null_samples)), float(np.max(null_samples))
    mean = float(np.mean(null_samples))

    ax.axvspan(lo, hi, color=REFERENCE, alpha=0.16, zorder=1,
               label="null range (agents with nothing to learn)")
    ax.axvline(mean, color=REFERENCE, linewidth=1.4, linestyle="--", zorder=2)

    # the null draws themselves, so the band is not just an assertion
    ax.scatter(null_samples, np.full_like(null_samples, -0.85), s=26,
               color=REFERENCE, edgecolors=SURFACE, linewidths=1.0, zorder=3)
    ax.annotate("null draws", xy=(mean, -0.85), xytext=(0, -16),
                textcoords="offset points", color=INK_MUTED, fontsize=8.5,
                ha="center")

    for i, (name, (stat, p)) in enumerate(cases.items()):
        inside = lo <= stat <= hi
        color = C2 if inside else C1
        ax.scatter([stat], [i], s=150, color=color, zorder=5,
                   edgecolors=SURFACE, linewidths=2.0)
        ax.annotate(
            f"p = {p:.3f}   {'not distinguishable' if inside else 'informative'}",
            xy=(stat, i), xytext=(12, 0), textcoords="offset points",
            color=INK, fontsize=9, va="center",
        )

    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(list(cases), color=INK, fontsize=9.5)
    ax.set_ylim(-1.6, len(cases) - 0.4)
    _style_axes(ax, title, "attribution span  v(all features) - v(none)", "",
                pad=26 if subtitle else 10)
    ax.grid(axis="y", visible=False)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.035), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=9, va="bottom")
    span = max(hi - lo, 1e-6)
    xs = [s for s, _ in cases.values()] + [lo, hi]
    ax.set_xlim(min(xs) - span * 0.4, max(xs) + span * 2.4)
    _save(fig, path)


def power_curve(
    edges: list[float],
    z_scores: list[float],
    detected: list[bool],
    path: Path,
    real_edge: float | None = None,
    title: str = "how much edge before an explanation is detectably informative?",
    subtitle: str = "",
) -> None:
    """detection z-score against the agent's measured edge.

    the x axis is measured edge rather than latent signal strength, because
    measured edge is what a practitioner has. points are coloured by whether
    the null test flagged them, and the detection band is drawn so the
    threshold can be read off directly.
    """
    fig, ax = _new_fig(8.6, 5.0)
    e = np.asarray(edges, dtype=float)
    z = np.asarray(z_scores, dtype=float)
    det = np.asarray(detected, dtype=bool)

    ax.axhspan(-1.96, 1.96, color=REFERENCE, alpha=0.16, zorder=1)
    ax.annotate("not distinguishable from an explanation of nothing",
                xy=(e.min(), 0), xytext=(4, 4), textcoords="offset points",
                color=INK_MUTED, fontsize=8.5, va="bottom")

    ax.plot(e, z, color=C1, linewidth=2.0, zorder=3)
    ax.scatter(e[~det], z[~det], s=90, color=C2, zorder=4,
               edgecolors=SURFACE, linewidths=2.0, label="not detected")
    if det.any():
        ax.scatter(e[det], z[det], s=90, color=C1, zorder=4,
                   edgecolors=SURFACE, linewidths=2.0, label="detected")

    if real_edge is not None:
        ax.axvline(real_edge, color=C2, linewidth=1.6, linestyle=":", zorder=2)
        ax.annotate(f"the real agent\n({real_edge:+.2f}/ep)",
                    xy=(real_edge, ax.get_ylim()[1]), xytext=(6, -26),
                    textcoords="offset points", color=INK, fontsize=8.5)

    _style_axes(ax, title, "agent's measured edge (dollars per episode)",
                "detection z-score", pad=26 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.035), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=9, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED,
              loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    _save(fig, path)


def steering(
    results: dict[str, list[dict]],
    path: Path,
    target_names: dict[str, str] | None = None,
    title: str = "can an explanation be changed while performance is held fixed?",
) -> None:
    """attribution share and return against penalty strength, per corpus.

    two panels rather than two axes on one plot: attribution share is a
    fraction and return is in dollars, and putting them on a shared axis would
    be the dual-axis mistake. the comparison the reader needs is between the
    two corpora within each panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), facecolor=SURFACE)
    colors = [C1, C2]

    for i, (name, rows) in enumerate(results.items()):
        coefs = [r["coef"] for r in rows]
        x = np.arange(len(coefs))
        share = np.array([r["target_share_mean"] for r in rows])
        share_e = np.array([r["target_share_std"] for r in rows])
        ret = np.array([r["return_mean"] for r in rows])
        ret_e = np.array([r["return_std"] for r in rows])
        label = name
        if target_names and name in target_names:
            label = f"{name}\n({target_names[name]})"

        axes[0].plot(x, share, color=colors[i], linewidth=2.0, marker="o",
                     markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.4,
                     label=label, zorder=3)
        axes[0].fill_between(x, share - share_e, share + share_e,
                             color=colors[i], alpha=0.16, linewidth=0, zorder=2)

        axes[1].plot(x, ret, color=colors[i], linewidth=2.0, marker="o",
                     markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.4,
                     label=label, zorder=3)
        axes[1].fill_between(x, ret - ret_e, ret + ret_e,
                             color=colors[i], alpha=0.16, linewidth=0, zorder=2)

    for ax, lab, ylab in (
        (axes[0], "attribution to the target feature", "share of attribution mass"),
        (axes[1], "task performance", "return per episode (dollars)"),
    ):
        ax.set_xticks(np.arange(len(next(iter(results.values())))))
        ax.set_xticklabels([str(r["coef"]) for r in next(iter(results.values()))])
        ax.axhline(0.0, color=REFERENCE, linewidth=1.2, linestyle="--", zorder=1)
        _style_axes(ax, lab, "steering penalty strength", ylab)

    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED,
                   loc="upper right")
    fig.suptitle(title, color=INK, fontsize=11, x=0.008, ha="left", y=1.03)
    _save(fig, path)


def competence_curve(
    returns: list[float],
    z_scores: list[float],
    detected: list[bool],
    fractions: list[float],
    path: Path,
    title: str = "does the test track agent competence?",
    subtitle: str = "",
) -> None:
    """detection z-score against agent return, along a training trajectory.

    each point is a checkpoint of the same agent on the same task, so the only
    thing varying is how good it is. the shaded band is where an explanation is
    not distinguishable from an explanation of an agent that learned nothing.
    """
    fig, ax = _new_fig(8.6, 5.0)
    r = np.asarray(returns, dtype=float)
    z = np.asarray(z_scores, dtype=float)
    det = np.asarray(detected, dtype=bool)

    ax.axhspan(-1.96, 1.96, color=REFERENCE, alpha=0.16, zorder=1)
    ax.annotate("not distinguishable from an explanation of nothing",
                xy=(r.min(), 0), xytext=(4, 4), textcoords="offset points",
                color=INK_MUTED, fontsize=8.5, va="bottom")

    order = np.argsort(r)
    ax.plot(r[order], z[order], color=C1, linewidth=2.0, zorder=3)
    ax.scatter(r[~det], z[~det], s=95, color=C2, zorder=4,
               edgecolors=SURFACE, linewidths=2.0, label="not detected")
    if det.any():
        ax.scatter(r[det], z[det], s=95, color=C1, zorder=4,
                   edgecolors=SURFACE, linewidths=2.0, label="detected")

    for ri, zi, f in zip(r, z, fractions):
        ax.annotate(f"{f:.0%}", xy=(ri, zi), xytext=(0, 11),
                    textcoords="offset points", color=INK_MUTED,
                    fontsize=8, ha="center")

    _style_axes(ax, title, "agent return (CartPole-v1, max 500)",
                "detection z-score", pad=26 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.035), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=9, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED,
              loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    _save(fig, path)
