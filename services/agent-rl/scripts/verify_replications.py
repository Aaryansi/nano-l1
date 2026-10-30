"""Verify supplemental studies and derive their publication inputs explicitly."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import combinations
import json
from pathlib import Path

import torch

import calibrate_reference_banks as banks
import objective_study
from study_archive import materialized
from run_manifest import digest
from publication_statistics import objective_statistics

ROOT = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def bank_input(path):
    if (path / "archive.json").exists():
        with materialized(path) as study:
            yield study
    else:
        yield path


def bank_statistics(path, proof_path, *, replay=False, allow_quick=False):
    if replay:
        require(not proof_path.resolve().is_relative_to(path.resolve())
                and not proof_path.exists(), "replay proof requires a fresh path outside immutable evidence")
    with bank_input(path) as study:
        if replay:
            require(not proof_path.resolve().is_relative_to(study.resolve())
                    and not proof_path.exists(), "replay proof requires a fresh path outside immutable evidence")
        result = banks.verify(study, replay=replay, allow_quick=allow_quick)
        expected_proof = {"study_manifest_sha256": digest(study / "study_manifest.json"),
                          "verification_source_sha256": digest(Path(banks.__file__)),
                          "verification": result | {"endpoints_replayed": True}}
        if replay:
            proof_path.parent.mkdir(parents=True, exist_ok=True)
            proof_path.write_text(json.dumps(expected_proof, indent=2, allow_nan=False) + "\n")
        else:
            require(json.loads(proof_path.read_text()) == expected_proof,
                    "bank replay proof differs from verified study/source")
        summary = banks.read(study / "summary.json")
        manifest = banks.read(study / "study_manifest.json")
        disagreements = []
        for index, strength in enumerate(manifest["config"]["strengths"]):
            counts = {f"{a}_versus_{b}": 0 for a, b in combinations(banks.METHODS, 2)}
            for bank in range(manifest["config"]["banks"]):
                case = banks.read(study / f"bank_{bank:03d}/bank.json")["cases"][index]
                for trial in case["trials"]:
                    for a, b in combinations(banks.METHODS, 2):
                        counts[f"{a}_versus_{b}"] += trial["methods"][a] != trial["methods"][b]
            disagreements.append({"signal_strength": strength, "trial_disagreements": counts})
        return {"config": manifest["config"], "procedure": manifest["procedure"], **summary,
                "trial_agreement": disagreements, "verification": result,
                "recorded_checkpoint_replay": expected_proof["verification"],
                "study_manifest_sha256": expected_proof["study_manifest_sha256"],
                "replay_proof_sha256": digest(proof_path)}


def derive(bank_path, proof_path, objective_path, corpus, original, *, allow_quick=False):
    bank_data = bank_statistics(bank_path, proof_path, allow_quick=allow_quick)
    objective = objective_study.verify(objective_path, corpus, original, allow_quick=allow_quick)
    statistics = objective_statistics(objective_path / "reports")
    statistics["scope"] = ("both endpoints reconstructed for every observed and reference policy; "
                           "original returns/spans compared; different reward units are not directly comparable; "
                           "no original checkpoint replay or objective-effect significance test")
    statistics["verification"] = objective
    statistics["study_manifest_sha256"] = digest(objective_path / "objective_study_manifest.json")
    return {"schema_version": 1, "reference_banks": bank_data, "objective_replication": statistics,
            "scientific_budgets_checked": bank_data["verification"]["scientific_budgets_checked"]
                and objective["scientific_budgets_checked"],
            "analysis_source_sha256": {str(Path(p).relative_to(ROOT)): digest(Path(p)) for p in
                (__file__, objective_study.__file__, str(ROOT / "services/agent-rl/scripts/study_archive.py"),
                 str(ROOT / "services/agent-rl/scripts/verify_objective_control.py"),
                 str(ROOT / "services/agent-rl/scripts/publication_statistics.py"))}}


def check_output(path, banks_path, objectives, original, corpus, proof):
    require(not path.exists() and not any(path.resolve().is_relative_to(source.resolve())
                for source in (banks_path, objectives, original) if source is not None)
            and path.resolve() not in (corpus.resolve(), proof.resolve()),
            "derived statistics require a fresh path outside immutable inputs and replay proof")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--replay-proof", type=Path, required=True)
    parser.add_argument("--objectives", type=Path)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/releases/market_snapshot_v1/corpus.npz")
    parser.add_argument("--original", type=Path, default=ROOT / "data/releases/research_v1/reports")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay-models", action="store_true")
    parser.add_argument("--allow-quick", action="store_true")
    args = parser.parse_args()
    if args.out:
        check_output(args.out, args.banks, args.objectives, args.original, args.corpus, args.replay_proof)
    torch.set_num_threads(1)
    if args.replay_models:
        bank_statistics(args.banks, args.replay_proof, replay=True, allow_quick=args.allow_quick)
    if args.objectives is not None:
        result = derive(args.banks, args.replay_proof, args.objectives, args.corpus,
                        args.original, allow_quick=args.allow_quick)
    else:
        result = bank_statistics(args.banks, args.replay_proof, allow_quick=args.allow_quick)
    if args.out:
        check_output(args.out, args.banks, args.objectives, args.original, args.corpus, args.replay_proof)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"scientific_budgets_checked": result.get("scientific_budgets_checked",
                      result.get("verification", {}).get("scientific_budgets_checked")),
                      "scope": "verified replication evidence and raw-derived summaries"}, indent=2))


if __name__ == "__main__":
    main()
