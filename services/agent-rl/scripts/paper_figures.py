"""regenerate the paper's figures at publication size.

the repository figures are drawn at 8-9 inches for screen reading. dropped into
a 3.3-inch journal column they are legible only under magnification, which is
the single most common way a good result is presented badly.

this redraws the five figures the paper uses, at the width they will actually
occupy, with type sized relative to that width rather than to the screen. two
are full-width (spanning both columns) because they carry two panels or a wide
axis; three are single-column.

reads the same json the analysis wrote, so these cannot drift from the numbers
verify_paper_numbers.py checks.

usage:
    python scripts/paper_figures.py --out ../../docs/paper/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# column widths in inches for a standard two-column letter layout
COL = 3.35
FULL = 7.0

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, REF = "#111111", "#555555", "#dddddd", "#999999"


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bbbbbb")
        ax.spines[sp].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=7, length=2, width=0.6)
    if title:
        ax.set_title(title, color=INK, fontsize=8, loc="left", pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=7.5)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {path.name}")


def _box(ax, x, y, w, h, label, *, edge=INK, face="white", lw=0.9, fs=7.5, tc=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.025",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4)


def _arrow(ax, p0, p1, *, color=INK, lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, linewidth=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2,
        shrinkA=0, shrinkB=0))


def fig_method(out: Path):
    """full width: the two nulls as interventions on the agent-environment loop.

    the distinction this paper turns on is *where* the corruption is applied,
    and that is a fact about a diagram, not about a number. carrying it in prose
    alone makes a one-glance idea take a paragraph to reconstruct.

    the only figure here not driven by reports/ json: it is a schematic, so it
    asserts nothing that could go stale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.85), facecolor="white")

    panels = (
        ("(a) parameter randomisation", True,
         "weights resampled, so the policy never learned anything.\n"
         "masking its inputs moves return arbitrarily."),
        ("(b) observation corruption (ours)", False,
         "the policy is trained normally on a real reward.\n"
         "the only thing withheld is information."),
    )

    for ax, (title, hot, note) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, color=INK, fontsize=8, loc="left", pad=3)

        # the corrupted component is the policy in (a), the channel in (b)
        _box(ax, 0.04, 0.52, 0.30, 0.26, "environment")
        _box(ax, 0.62, 0.52, 0.34, 0.26, r"policy $\pi_\theta$",
             edge=C2 if hot else INK, lw=1.7 if hot else 0.9,
             face="#fdeee7" if hot else "white")

        # observation channel, left to right
        _arrow(ax, (0.34, 0.65), (0.62, 0.65),
               color=INK if hot else C1, lw=0.9 if hot else 1.6,
               ls="-" if hot else (0, (2.2, 1.5)))
        ax.text(0.48, 0.83, "observation" if hot else "observation\n(resampled)",
                ha="center", va="center", fontsize=7,
                color=INK if hot else C1, linespacing=1.25)

        # action closes the loop and is untouched by either intervention. drawn
        # as an explicit rectangular path: an arc3 of this span bows up through
        # the boxes rather than around them.
        ax.plot([0.79, 0.79], [0.52, 0.28], color=MUTED, lw=0.9, zorder=2)
        ax.plot([0.79, 0.19], [0.28, 0.28], color=MUTED, lw=0.9, zorder=2)
        _arrow(ax, (0.19, 0.28), (0.19, 0.52), color=MUTED)
        ax.text(0.49, 0.20, "action", ha="center", va="top", fontsize=7,
                color=MUTED)

        ax.text(0.0, 0.06, note, ha="left", va="top", fontsize=6.8,
                color=MUTED, linespacing=1.45, transform=ax.transAxes)

    fig.subplots_adjust(wspace=0.10)
    save(fig, out / "fig_method.png")


def fig_null_test(d, out: Path):
    """single column: three cases against the null band."""
    nulls = np.array(d["null_spans"])
    lo, hi, mu = nulls.min(), nulls.max(), nulls.mean()
    cases = [
        ("planted signal", d["results"]["planted_signal"]),
        ("null corpus", d["results"]["held_out_null"]),
        ("real market", d["results"]["real_market"]),
    ]

    fig, ax = plt.subplots(figsize=(COL, 1.9), facecolor="white")
    ax.axvspan(lo, hi, color=REF, alpha=0.18, zorder=1)
    ax.axvline(mu, color=REF, lw=0.9, ls="--", zorder=2)
    ax.scatter(nulls, np.full_like(nulls, -0.85), s=8, color=REF,
               edgecolors="white", linewidths=0.4, zorder=3)
    ax.text(mu, -1.35, "null draws", color=MUTED, fontsize=6.5, ha="center")

    for i, (name, r) in enumerate(cases):
        inside = lo <= r["statistic"] <= hi
        ax.scatter([r["statistic"]], [i], s=55, color=C2 if inside else C1,
                   zorder=5, edgecolors="white", linewidths=1.1)
        ax.annotate(f"$z={r['z_score']:+.2f}$", xy=(r["statistic"], i),
                    xytext=(7, 0), textcoords="offset points",
                    color=INK, fontsize=7, va="center")

    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([c[0] for c in cases], color=INK, fontsize=7.5)
    ax.set_ylim(-1.7, len(cases) - 0.4)
    style(ax, xlabel=r"attribution span  $v(N)-v(\emptyset)$")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(lo - 12, max(r["statistic"] for _, r in cases) * 1.28)
    save(fig, out / "fig_null_test.png")


def fig_power(d, out: Path):
    """single column: detection z against measured edge."""
    rows = d["power_curve"]
    e = np.array([r["agent_edge"] for r in rows])
    z = np.array([r["z"] for r in rows])
    det = np.array([r["detected"] for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(COL, 2.1), facecolor="white")
    ax.axhspan(-1.96, 1.96, color=REF, alpha=0.18, zorder=1)
    o = np.argsort(e)
    ax.plot(e[o], z[o], color=C1, lw=1.4, zorder=3)
    ax.scatter(e[~det], z[~det], s=34, color=C2, zorder=4,
               edgecolors="white", linewidths=1.0, label="not detected")
    ax.scatter(e[det], z[det], s=34, color=C1, zorder=4,
               edgecolors="white", linewidths=1.0, label="detected")
    ax.axvline(-0.595, color=C2, lw=1.0, ls=":", zorder=2)
    ax.annotate("real agent", xy=(-0.595, z.max() * 0.82), xytext=(4, 0),
                textcoords="offset points", color=INK, fontsize=6.5)
    ax.text(e.min(), 0.2, "not distinguishable from nothing",
            color=MUTED, fontsize=6.5, va="bottom")
    style(ax, xlabel="agent's measured edge (\\$ / episode)",
          ylabel="detection $z$")
    ax.legend(frameon=False, fontsize=6.5, labelcolor=MUTED, loc="lower right",
              handletextpad=0.4, borderpad=0.2)
    save(fig, out / "fig_power.png")


def fig_null_comparison(d, out: Path):
    """full width: the two null constructions, one panel per environment.

    laid out on a grid rather than a single row. four panels across seven
    inches leaves each one 1.75in wide, which is narrower than its own x-axis
    label.
    """
    ncols = 2 if len(d) > 2 else len(d)
    nrows = int(np.ceil(len(d) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL, 1.9 * nrows),
                             facecolor="white", squeeze=False)
    flat = axes.ravel()
    for spare in flat[len(d):]:
        spare.axis("off")

    for i, (ax, row) in enumerate(zip(flat, d)):
        w = np.array(row["weight_null"]["spans"])
        v = np.array(row["env_null"]["spans"])
        for y, vals, color, lab in ((1, w, C2, "parameter"), (0, v, C1, "environment")):
            ax.plot([vals.min(), vals.max()], [y, y], color=color, lw=5,
                    alpha=0.30, solid_capstyle="round", zorder=2)
            ax.scatter(vals, np.full_like(vals, y), s=10, color=color,
                       edgecolors="white", linewidths=0.4, zorder=4)
            ax.annotate(f"sd {vals.std(ddof=1):.2f}", xy=(vals.max(), y),
                        xytext=(5, 0), textcoords="offset points",
                        color=INK, fontsize=6.5, va="center")
        ax.axvline(0, color=REF, lw=0.9, ls="--", zorder=1)
        ax.set_yticks([0, 1])
        # only the leftmost panel is labelled. repeating identical row labels on
        # every panel puts them in the gap between subplots, where they collide
        # with the sd annotation the panel to their left writes past its axes.
        ax.set_yticklabels(
            ["environment", "parameter"] if i % ncols == 0 else ["", ""],
            color=INK, fontsize=7)
        ax.set_ylim(-0.6, 1.6)
        style(ax, xlabel=r"span  $v(N)-v(\emptyset)$", title=row["env_id"])
        ax.grid(axis="y", visible=False)
        # leave room on the right for the widest sd annotation
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.22 * (hi - lo))
    fig.subplots_adjust(wspace=0.12, hspace=0.85)
    save(fig, out / "fig_null_comparison.png")


def fig_steering(d, out: Path):
    """full width: attribution and return against penalty strength."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1), facecolor="white")
    series = (("real market", d["real_market"], C1),
              ("planted signal", d["learnable_synthetic"], C2))

    for name, rows, color in series:
        x = np.arange(len(rows))
        sh = np.array([r["target_share_mean"] for r in rows])
        se = np.array([r["target_share_std"] for r in rows])
        rt = np.array([r["return_mean"] for r in rows])
        re_ = np.array([r["return_std"] for r in rows])
        axes[0].plot(x, sh, color=color, lw=1.4, marker="o", ms=3.4,
                     markeredgecolor="white", markeredgewidth=0.8, label=name)
        axes[0].fill_between(x, sh - se, sh + se, color=color, alpha=0.15, lw=0)
        axes[1].plot(x, rt, color=color, lw=1.4, marker="o", ms=3.4,
                     markeredgecolor="white", markeredgewidth=0.8, label=name)
        axes[1].fill_between(x, rt - re_, rt + re_, color=color, alpha=0.15, lw=0)

    labels = [str(r["coef"]) for r in d["real_market"]]
    for ax, ttl, yl in ((axes[0], "attribution to the target feature",
                         "share of attribution mass"),
                        (axes[1], "task performance", "return (\\$ / episode)")):
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.axhline(0, color=REF, lw=0.9, ls="--", zorder=1)
        style(ax, xlabel="steering penalty strength", ylabel=yl, title=ttl)
    axes[0].legend(frameon=False, fontsize=6.5, labelcolor=MUTED, loc="upper right",
                   handletextpad=0.4, borderpad=0.2)
    save(fig, out / "fig_steering.png")


def fig_nullchoice(d, out: Path):
    """single column: one observed span against two null constructions.

    the whole point is that the vertical line does not move. everything that
    changes between the two rows is the reference distribution, and the verdict
    changes with it.
    """
    obs = d["observed_span"]
    rows = (
        ("synthetic\ncorpora", d["null_synthetic"], C2),
        ("blinded real\nepisodes", d["null_blinded_real"], C1),
    )

    fig, ax = plt.subplots(figsize=(COL, 1.95), facecolor="white")
    ax.axvline(obs, color=INK, lw=1.2, zorder=5)
    ax.annotate(f"observed  {obs:+.2f}", xy=(obs, 1.62), xytext=(4, 0),
                textcoords="offset points", fontsize=6.6, color=INK, va="top")

    for y, (label, block_, color) in enumerate(rows):
        v = np.array(block_["spans"])
        ax.plot([v.min(), v.max()], [y, y], color=color, lw=6, alpha=0.28,
                solid_capstyle="round", zorder=2)
        ax.scatter(v, np.full_like(v, y), s=9, color=color, zorder=4,
                   edgecolors="white", linewidths=0.4)
        z = block_["result"]["z_score"]
        verdict = "informative" if block_["result"]["passes"] else "not distinguishable"
        # anchored to the axes edge, not to the data, so the two rows'
        # annotations line up instead of tracking their own maxima
        ax.annotate(f"$z={z:+.2f}$   {verdict}", xy=(1.0, y),
                    xycoords=("axes fraction", "data"),
                    xytext=(-2, -11), textcoords="offset points",
                    fontsize=6.4, color=MUTED, ha="right", va="top")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=7,
                       linespacing=1.15)
    ax.set_ylim(-0.75, 1.75)
    style(ax, xlabel=r"span  $v(N)-v(\emptyset)$")
    ax.grid(axis="y", visible=False)
    save(fig, out / "fig_nullchoice.png")


def fig_conjecture(d, out: Path):
    """single column: the parameter null's width against random-init return sd.

    the points sit on the identity line, which is the explanation: the
    parameter null does not measure anything about explanations, it measures
    the spread of random initialisation.
    """
    x = np.array(d["random_return_sd"])
    y = np.array(d["weight_null_sd"])
    names = [n.split("-")[0] for n in d["env_ids"]]

    fig, ax = plt.subplots(figsize=(COL, 2.2), facecolor="white")
    hi = max(x.max(), y.max()) * 1.18
    ax.plot([0, hi], [0, hi], color=REF, lw=0.9, ls="--", zorder=1)
    # label the line in the empty middle stretch: at the top right it lands on
    # the cluster the figure is about
    ax.text(hi * 0.52, hi * 0.46, "$y=x$", fontsize=6.5, color=MUTED,
            ha="left", va="top")
    ax.scatter(x, y, s=42, color=C1, zorder=4, edgecolors="white",
               linewidths=1.1)
    # three of the four environments land within ~15 units of each other, so
    # labels placed at a uniform offset overprint each other and the markers.
    # they are staggered around the cluster and tied back with leader lines.
    offsets = {0: (-2, -30), 1: (-24, 30), 2: (-34, -12), 3: (34, 15)}
    order = np.argsort(-x)
    for rank, i in enumerate(order):
        ax.annotate(
            names[i], xy=(x[i], y[i]), xytext=offsets[rank],
            textcoords="offset points", fontsize=6.6, color=INK,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=REF, lw=0.6,
                            shrinkA=1, shrinkB=4))

    ax.set_xlim(-hi * 0.06, hi)
    ax.set_ylim(-hi * 0.06, hi)
    style(ax, xlabel="sd of random-init return",
          ylabel="sd of parameter null")
    save(fig, out / "fig_conjecture.png")


def fig_constructions(perm, matched, out: Path):
    """full width: three null constructions on the same three cases.

    the figure the null sections build to. each panel is one case; each row is
    one construction; the vertical line is the observed span, which never moves.
    what a reader should see is that the middle construction's reference has no
    width at all on the two synthetic cases, which is why it fires on both.
    """
    by_case = {}
    for label, d in (("permuted", perm), ("blinded", matched)):
        for c in d["cases"]:
            by_case.setdefault(c["case"], {})[label] = c

    cases = ["planted signal", "null corpus", "real market"]
    fig, axes = plt.subplots(1, len(cases), figsize=(FULL, 1.95),
                             facecolor="white")

    for i, (ax, case) in enumerate(zip(np.atleast_1d(axes), cases)):
        rows = (("blinded", by_case[case]["blinded"], C2),
                ("permuted", by_case[case]["permuted"], C1))
        obs = by_case[case]["permuted"]["span"]

        for y, (label, c, color) in enumerate(rows):
            v = np.array(c["null_spans"])
            lo, hi = v.min(), v.max()
            if hi - lo < 1e-9:          # a point mass would draw as nothing
                ax.scatter([lo], [y], s=46, color=color, marker="D", zorder=5,
                           edgecolors="white", linewidths=1.0)
                ax.annotate("point mass", xy=(lo, y), xytext=(6, 0),
                            textcoords="offset points", fontsize=6.2,
                            color=color, va="center")
            else:
                ax.plot([lo, hi], [y, y], color=color, lw=5, alpha=0.28,
                        solid_capstyle="round", zorder=2)
                ax.scatter(v, np.full_like(v, y), s=7, color=color, zorder=4,
                           edgecolors="white", linewidths=0.3)
            # the permuted artifact stores a three-way verdict, since a
            # rejection BELOW the null is not the same as declining
            mark = c.get("verdict") or ("fires" if c["fires"] else "declines")
            ax.annotate(mark, xy=(1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(-2, -10), textcoords="offset points",
                        fontsize=6.2, color=MUTED, ha="right", va="top")

        ax.axvline(obs, color=INK, lw=1.1, zorder=6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["blinded", "permuted"] if i == 0 else ["", ""],
                           color=INK, fontsize=7)
        ax.set_ylim(-0.75, 1.6)
        style(ax, xlabel=r"span  $v(N)-v(\emptyset)$", title=case)
        ax.grid(axis="y", visible=False)

    fig.subplots_adjust(wspace=0.14)
    save(fig, out / "fig_constructions.png")


def fig_budget(d, out: Path):
    """single column: the blinded null collapses as its agents converge.

    two axes on one plot would be a dual-axis chart, which is the one chart
    type worth refusing outright. the null width and the verdict statistic are
    the same quantity's numerator and denominator, so the panel shows the null
    band shrinking and marks where the verdict crosses.
    """
    rows = sorted(d["rows"], key=lambda r: r["updates"])
    x = np.arange(len(rows))
    mean = np.array([r["null_mean"] for r in rows])
    sd = np.array([r["null_std"] for r in rows])
    obs = d["observed_span"]
    fires = np.array([r["fires"] for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(COL, 2.05), facecolor="white")
    ax.axhline(obs, color=INK, lw=1.2, zorder=4)
    ax.annotate("observed span", xy=(x[-1], obs), xytext=(0, 4),
                textcoords="offset points", fontsize=6.5, color=INK,
                ha="right")
    ax.fill_between(x, mean - sd, mean + sd, color=C1, alpha=0.18, lw=0)
    ax.plot(x, mean, color=C1, lw=1.5, marker="o", ms=3.6,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(x[0], mean[0] + sd[0], "  null, $\\pm$1 sd", fontsize=6.5,
            color=C1, va="bottom")

    # mark where the verdict changes rather than colouring every point
    flip = np.flatnonzero(fires[1:] != fires[:-1])
    for i in flip:
        ax.axvline(i + 0.5, color=C2, lw=0.9, ls=":", zorder=2)
        ax.text(i + 0.55, ax.get_ylim()[1], " verdict flips", fontsize=6.4,
                color=C2, va="top")

    ax.set_xticks(x)
    ax.set_xticklabels([str(r["updates"]) for r in rows])
    style(ax, xlabel="training updates for the null agents",
          ylabel=r"span  $v(N)-v(\emptyset)$")
    save(fig, out / "fig_budget.png")


def fig_manifold(d, out: Path):
    """single column: how far each masking mode strays from the data manifold.

    the shape is the argument. both curves start and end on the manifold, and
    only the interior is off it, which is exactly the claim that the span
    (built from the two endpoints) cannot be an off-manifold artefact.
    """
    rows = sorted(d["offmanifold_distance"], key=lambda r: r["n_kept"],
                  reverse=True)
    n_feat = len(d["feature_names"])

    # the fully-masked point is dropped: those states are drawn FROM the
    # reference sample, so their distance to it is trivially zero and plotting
    # it would read as evidence when it is a tautology. the v(N) row is the
    # honest baseline, drawn as a floor instead.
    floor = next(r["marginal"] for r in rows if r["n_kept"] == n_feat)
    body = [r for r in rows if 0 < r["n_kept"] < n_feat]
    x = np.array([n_feat - r["n_kept"] for r in body])
    m = np.array([r["marginal"] for r in body])
    c = np.array([r["conditional"] for r in body])

    fig, ax = plt.subplots(figsize=(COL, 2.0), facecolor="white")
    ax.axhline(floor, color=REF, lw=0.9, ls="--", zorder=1)
    ax.text(x.max(), floor, "real states", fontsize=6.4, color=MUTED,
            ha="right", va="bottom")
    for y, color, lab in ((m, C2, "marginal"), (c, C1, "conditional")):
        ax.plot(x, y, color=color, lw=1.5, marker="o", ms=3.6,
                markeredgecolor="white", markeredgewidth=0.8, label=lab)

    style(ax, xlabel="features replaced",
          ylabel="distance to nearest real state")
    ax.legend(frameon=False, fontsize=6.5, labelcolor=MUTED, loc="lower left",
              handletextpad=0.4, borderpad=0.2)
    save(fig, out / "fig_manifold.png")


def fig_decoy(d, out: Path):
    """single column: the decoy's share under each attribution target."""
    fig, ax = plt.subplots(figsize=(COL, 1.75), facecolor="white")
    labels = ["per-decision\n$\\pi(a\\mid s)$", "outcome-based\nepisode return"]
    shares = [d["decoy"]["decoy_naive_share"], d["decoy"]["decoy_trajectory_share"]]
    ranks = [d["decoy"]["decoy_naive_rank"], d["decoy"]["decoy_trajectory_rank"]]

    y = np.arange(len(labels))
    ax.barh(y, shares, height=0.5, color=[C2, C1], zorder=3)
    for i, (s_, r_) in enumerate(zip(shares, ranks)):
        ax.annotate(f"{s_:.1%}   rank {r_} of 18", xy=(s_, i), xytext=(5, 0),
                    textcoords="offset points", color=INK, fontsize=7, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=7)
    ax.set_xlim(0, max(shares) * 2.5)
    style(ax, xlabel="share of attribution mass given to the decoy")
    ax.grid(axis="y", visible=False)
    save(fig, out / "fig_decoy.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="../../reports")
    ap.add_argument("--out", default="../../docs/paper/figures")
    args = ap.parse_args()

    R, out = Path(args.reports), Path(args.out)
    L = lambda n: json.loads((R / n).read_text())  # noqa: E731

    print("regenerating paper figures at publication size")
    fig_method(out)
    fig_null_test(L("sanity_test.json"), out)
    fig_power(L("power_curve.json"), out)
    fig_null_comparison(L("generalize_gym.json"), out)
    fig_steering(L("steering.json"), out)
    fig_conjecture(L("null_width_conjecture.json"), out)
    fig_nullchoice(L("null_corpus_check.json"), out)
    fig_constructions(L("permuted_null_test.json"),
                      L("matched_null_test.json"), out)
    fig_budget(L("null_budget_check.json"), out)
    fig_manifold(L("manifold_masking.json"), out)
    fig_decoy(L("faithfulness.json"), out)
    print("done")


if __name__ == "__main__":
    main()
