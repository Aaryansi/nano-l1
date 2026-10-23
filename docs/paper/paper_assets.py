"""Generate numerical manuscript inputs and scientific plots from one artifact."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def tex(text):
    substitutions = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%",
                     "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(substitutions.get(character, character) for character in str(text))


def number(value, digits=3):
    if value is None:
        return "undefined"
    if not math.isfinite(value):
        raise ValueError("undefined manuscript numbers must be null")
    if round(value, digits) == 0:
        value = 0.
    return rf"\ensuremath{{{value:.{digits}f}}}"


def interval(values, digits=3):
    if values is None:
        return "undefined"
    if (len(values) != 2 or not all(math.isfinite(value) for value in values)
            or values[0] > values[1]):
        raise ValueError("invalid manuscript interval")
    return rf"\ensuremath{{[{values[0]:.{digits}f},\,{values[1]:.{digits}f}]}}"


def percent(value):
    if value is None:
        return "undefined"
    if not math.isfinite(value):
        raise ValueError("invalid manuscript percentage")
    return f"{100 * value:.1f}\\%"


def decision(value):
    return {"informative": "outside", "not distinguishable from null": "within",
            "unresolved": "unresolved"}[value]


def classification_counts(rows):
    labels = [row["legacy_verdict"] for row in rows]
    return (labels.count("informative"), labels.count("not distinguishable from null"),
            labels.count("unresolved"))


def table(columns, headings, rows):
    return (rf"\begin{{tabular}}{{{columns}}}\toprule" + "\n"
            + " & ".join(headings) + r" \\ \midrule" + "\n"
            + "\n".join(" & ".join(row) + r" \\" for row in rows)
            + "\n" + r"\bottomrule\end{tabular}")


def macro(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}" + "\n"


def numerical_text(statistics, context):
    market = statistics["market"]
    manifest, parity = context["manifest"], context["parity"]
    macros = {
        "MarketMean": number(market["costs"]["mean"]),
        "MarketInterval": interval(market["primary_uncertainty"]["ci95"]),
        "MarketProbability": number(market["primary_uncertainty"]["p_value"], 4),
        "TrainingCommit": tex(manifest["commit"][:12]),
        "EpisodeCount": f"{manifest['episodes']:,}",
        "TrainCount": f"{manifest['splits']['train']['episodes']:,}",
        "ValidationCount": f"{manifest['splits']['validation']['episodes']:,}",
        "TestCount": f"{manifest['splits']['test']['episodes']:,}",
        "DayCount": str(market["primary_uncertainty"]["n_utc_days"]),
    }
    error = max(value for row in parity["checkpoints"]
                for value in row["maximum_accounting_errors"].values())
    if error == 0:
        macros["ParityError"] = "zero"
    else:
        exponent = math.floor(math.log10(error))
        mantissa = math.ceil(error / 10 ** exponent)
        macros["ParityError"] = rf"\ensuremath{{\leq {mantissa}\times10^{{{exponent}}}}}"
    primary = statistics["headline"]["market_per_seed"]
    outside, within, unresolved = classification_counts(primary)
    macros["MarketSeedParagraph"] = (
        f"Across {len(primary)} market seeds, the primary bank gives {outside} outside, "
        f"{within} within and {unresolved} unresolved classifications. "
        f"Their spans range from {number(min(row['span'] for row in primary))} to "
        f"{number(max(row['span'] for row in primary))}.")
    larger = statistics["headline"]["larger_bank_market"]
    if larger is None:
        raise ValueError("manuscript requires the larger-bank sensitivity")
    outside, within, unresolved = classification_counts(larger)
    macros["SensitivityParagraph"] = (
        f"The larger bank gives {outside} outside, {within} within and {unresolved} "
        f"unresolved classifications across the same {len(larger)} market seeds.")
    null_case = next(row for row in statistics["conditional_calibration"]["cases"]
                     if row["signal_strength"] == 0.)
    if null_case["n_resolved"]:
        legacy = (f"flags {null_case['n_rejected']}/{null_case['n_resolved']} resolved trials "
                  f"(conditional interval {interval(null_case['rejection_rate_ci95'])})")
    else:
        legacy = "has no resolved trial rate"
    ranks = null_case["rank_comparators"]
    macros["CalibrationParagraph"] = (
        f"At zero strength, the legacy rule {legacy}, with "
        f"{null_case['n_unresolved']} unresolved trials. The upper-tail and symmetric ranks "
        f"flag {ranks['greater']['n_rejected']}/{null_case['n_trials']} and "
        f"{ranks['two-sided']['n_rejected']}/{null_case['n_trials']}, respectively. "
        f"Their conditional intervals are {interval(ranks['greater']['rejection_rate_ci95'])} and "
        f"{interval(ranks['two-sided']['rejection_rate_ci95'])}.")
    controls = statistics["controls"]["seeds"]
    units = [unit for task in controls for unit in task["units"]]
    eligible = [unit for unit in units if unit["eligible"]]
    resolved = [unit for unit in eligible if "unresolved" not in
                (unit["environment_verdict"], unit["weight_verdict"])]
    differing = sum(unit["environment_verdict"] != unit["weight_verdict"] for unit in resolved)
    macros["ControlParagraph"] = (
        f"Of {len(units)} independently trained control agents, {len(eligible)} meet both "
        f"eligibility gates. Of those, {len(resolved)} have two resolved references; "
        f"their classifications differ in {differing} cases. No unresolved comparison "
        "is counted as reference agreement.")
    market_state = next(row for row in statistics["state_interventions"]["cases"]
                        if row["case"] == "market")["interventions"]
    full, trajectory = market_state["full_observation"], market_state["market_trajectories"]
    macros["StateParagraph"] = (
        f"The full-input replacement span is {number(full['mean_span'])}, with interval "
        f"{interval(full['ci95'])}; the donor-trajectory span retaining inventory and clock "
        f"is {number(trajectory['mean_span'])}, with interval {interval(trajectory['ci95'])}.")
    sentences = []
    for key, label in (("real_market", "market clock"), ("learnable_synthetic", "synthetic signal")):
        rows = statistics["steering"]["populations"][key]
        baseline = next(row for row in rows if row["penalty"] == 0.)
        selected = next(row for row in rows if row["penalty"] == 20.)
        sentences.append(
            f"For the {label}, the mean target share changes from "
            f"{percent(baseline['target_share_mean'])} to {percent(selected['target_share_mean'])}; "
            f"the paired return change is {number(selected['return_difference'])} dollars, "
            f"with interval {interval(selected['return_diff_ci95'])}.")
    macros["SteeringParagraph"] = " ".join(sentences)
    return "".join(macro(name, value) for name, value in macros.items())


def numerical_tables(statistics):
    market = statistics["market"]
    result = macro("MarketTable", table("lrrr", ["Training objective", "Seeds", "Mean", "Seed SD"], [
        [label, str(market[key]["n_seeds"]), number(market[key]["mean"]), number(market[key]["seed_sd"])]
        for key, label in (("costs", "Fees and spread"), ("zero_cost", "Zero cost"))]))
    labels = {"planted_signal": "Planted signal", "held_out_null": "Zero strength", "real_market": "Market seed 0"}
    result += macro("HeadlineTable", table("lrrrl", ["Case", "Span", "Ref. SD", "$z$", "Decision"], [
        [labels[name], number(row["span"]), number(row["reference_std"]),
         number(row["standardized_distance"], 2), decision(row["legacy_verdict"])]
        for name, row in statistics["headline"]["cases"].items()]))
    result += macro("MarketSeedTable", table("rrrl", ["Seed", "Span", "$z$", "Decision"], [
        [str(index), number(row["span"]), number(row["standardized_distance"], 2), decision(row["legacy_verdict"])]
        for index, row in enumerate(statistics["headline"]["market_per_seed"])]))
    controls = statistics["controls"]["seeds"]
    result += macro("ControlTable", table("lrrrr", ["Task", "Return", "Elig.", "Env. O/U", "Param. O/U"], [
        [tex(task["task"].split("-")[0]), number(task["mean_return"], 1),
         f"{task['eligible']}/{task['n_units']}",
         f"{task['reference_counts']['env']['outside']}/{task['reference_counts']['env']['unresolved']}",
         f"{task['reference_counts']['weight']['outside']}/{task['reference_counts']['weight']['unresolved']}"]
        for task in controls]))
    result += macro("BaselineTable", table("lr", ["Policy", "Mean return"], [
        [tex(name), number(value)] for name, value in market["baseline_mean_returns"].items()]))
    rows = []
    for key, label in (("real_market", "Market"), ("learnable_synthetic", "Signal")):
        for row in statistics["steering"]["populations"][key]:
            rows.append([label, number(row["penalty"], 0), percent(row["target_share_mean"]),
                         number(row["mean"], 2), interval(row.get("return_diff_ci95"), 2)
                         if row["penalty"] else "baseline"])
    result += macro("SteeringTable", table("lrrrr", ["Population", "$\\lambda$", "Share", "Return", "Change CI"], rows))
    rows = []
    mode_labels = {"full_observation": "Full inputs", "market_rows": "Market rows",
                   "market_trajectories": "Donor trajectory"}
    for case in statistics["state_interventions"]["cases"]:
        for mode, row in case["interventions"].items():
            rows.append([tex(case["case"]), mode_labels[mode], number(row["mean_span"], 2), interval(row["ci95"], 2)])
    result += macro("StateTable", table("llrr", ["Case", "Intervention", "Span", "95\\% CI"], rows))
    return result


def figure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "savefig.dpi": 450,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})


def save(fig, path):
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=.04, facecolor="white")
    plt.close(fig)


def scientific_figures(statistics, output):
    output.mkdir(parents=True, exist_ok=True)
    figure_style()
    blue, orange, gray = "#246898", "#bd622f", "#62676b"
    cases = statistics["conditional_calibration"]["cases"]
    fig, ax = plt.subplots(figsize=(3.2, 2.2), layout="constrained")
    for offset, kind, label, color, marker in ((-.16, "legacy", "Legacy rule", gray, "s"),
                                              (0., "greater", "Upper rank", blue, "o"),
                                              (.16, "two-sided", "Symmetric rank", orange, "^")):
        for index, case in enumerate(cases):
            data = case if kind == "legacy" else case["rank_comparators"][kind]
            mean = data["rejection_rate_resolved"] if kind == "legacy" else data["rejection_rate"]
            bounds = data["rejection_rate_ci95"]
            if mean is None:
                ax.text(index + offset, .05, "unresolved", rotation=90, fontsize=6,
                        color=color, ha="center", va="bottom")
                continue
            position = index + offset
            ax.plot([position] * 2, bounds, color=color, linewidth=.8)
            ax.plot([position] * 2, bounds, color=color, marker="_",
                    linestyle="none", markersize=4)
            ax.plot(position, mean, color=color, marker=marker, markersize=3.5)
    ax.axhline(.05, color=gray, linestyle=":", linewidth=.7)
    ax.set(xticks=range(len(cases)), xticklabels=[f"{case['signal_strength']:g}" for case in cases],
           xlabel="planted signal strength", ylabel="conditional flag rate", ylim=(-.03, 1.04))
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=3.5, label=label)
               for label, color, marker in (("Legacy rule", gray, "s"), ("Upper rank", blue, "o"),
                                             ("Symmetric rank", orange, "^"))]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0., .6))
    save(fig, output / "calibration.png")
    case = next(row for row in statistics["state_interventions"]["cases"] if row["case"] == "market")
    names = ("full_observation", "market_rows", "market_trajectories")
    fig, ax = plt.subplots(figsize=(3.2, 1.9), layout="constrained")
    for index, name in enumerate(names):
        row = case["interventions"][name]
        mean, bounds = row["mean_span"], row["ci95"]
        ax.plot(bounds, [index, index], color=blue, linewidth=1.)
        ax.plot(bounds, [index, index], color=blue, marker="|", linestyle="none", markersize=5)
        ax.plot(mean, index, color=blue, marker="o", markersize=3.5)
    ax.axvline(0., color=gray, linewidth=.6)
    ax.set(yticks=range(3), yticklabels=["Full inputs", "Market rows", "Donor trajectory"],
           xlabel="episode return span (USD)", ylim=(2.5, -.5))
    save(fig, output / "state_interventions.png")
    fig, axes = plt.subplots(2, 1, figsize=(3.2, 3.2), sharex=True, layout="constrained")
    for key, label, color, marker in (("real_market", "Market clock", blue, "o"),
                                      ("learnable_synthetic", "Synthetic signal", orange, "s")):
        rows = statistics["steering"]["populations"][key]
        x = np.arange(len(rows))
        axes[0].plot(x, [row["target_share_mean"] for row in rows], color=color,
                     marker=marker, markersize=3, linewidth=1., label=label)
        for index, row in enumerate(rows):
            shares = row["target_shares"]
            axes[0].scatter(index + np.linspace(-.06, .06, len(shares)), shares,
                            s=4, color=color, alpha=.4)
            if row["penalty"]:
                mean, bounds = row["return_difference"], row["return_diff_ci95"]
                offset = -.04 if key == "real_market" else .04
                axes[1].plot([index + offset] * 2, bounds, color=color, linewidth=.9)
                axes[1].plot([index + offset] * 2, bounds, color=color,
                             marker="_", linestyle="none", markersize=4)
                axes[1].plot(index + offset, mean, color=color, marker=marker, markersize=3)
        axes[1].set_xticks(x, [f"{row['penalty']:g}" for row in rows])
    axes[0].set(ylabel="target attribution share", ylim=(-.03, 1.03))
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].axhline(0., color=gray, linewidth=.6)
    axes[1].set(xlabel="training penalty", ylabel="paired return change (USD)")
    save(fig, output / "steering.png")


def write_assets(statistics, context, output: Path, *, fixture=False):
    if statistics["verification"]["scientific_budgets_checked"] is not True and not fixture:
        raise ValueError("manuscript assets require verified scientific budgets")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.tex").write_text(numerical_text(statistics, context))
    (output / "tables.tex").write_text(numerical_tables(statistics))
    scientific_figures(statistics, output / "figures")
