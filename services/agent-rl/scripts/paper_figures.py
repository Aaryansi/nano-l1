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
    """full width: the two null constructions, one row per environment."""
    fig, axes = plt.subplots(1, len(d), figsize=(FULL, 1.9), facecolor="white")
    axes = np.atleast_1d(axes)

    for i, (ax, row) in enumerate(zip(axes, d)):
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
        ax.set_yticklabels(["environment", "parameter"] if i == 0 else ["", ""],
                           color=INK, fontsize=7)
        ax.set_ylim(-0.6, 1.6)
        style(ax, xlabel=r"span  $v(N)-v(\emptyset)$", title=row["env_id"])
        ax.grid(axis="y", visible=False)
        # leave room on the right for the widest sd annotation
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.22 * (hi - lo))
    fig.subplots_adjust(wspace=0.12)
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
    fig_null_test(L("sanity_test.json"), out)
    fig_power(L("power_curve.json"), out)
    fig_null_comparison(L("generalize_gym.json"), out)
    fig_steering(L("steering.json"), out)
    fig_decoy(L("faithfulness.json"), out)
    print("done")


if __name__ == "__main__":
    main()
