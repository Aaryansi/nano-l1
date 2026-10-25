"""Replicate synthetic calibration with fresh reference data in every bank.

Outputs are isolated, replayable and separate from the selected manuscript
release. Full defaults implement docs/REFERENCE_BANK_PROTOCOL.md. --quick is
an execution check and cannot establish the declared scientific budgets.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_rl.agents.ppo import PPOAgent, PPOConfig
from nano_rl.env.binary_market import BinaryMarketEnv
from nano_rl.env.features import N_FEATURES, fit_normalizer
from nano_rl.env.synthetic import make_learnable_corpus
from nano_rl.explain.rank import rank_comparisons
from nano_rl.explain.sanity import UNRESOLVED, check_resolving_power, test_span_against_null
from bank_calibration_statistics import METHODS, summarize_banks
from calibrate_detection import summarize
from portable_validation import reconstructed
from run_manifest import digest
from sanity_check_explanations import endpoint_span

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "docs/REFERENCE_BANK_PROTOCOL.md"
STANDARD = {"banks": 20, "n_null": 24, "trials": 30, "updates": 40,
            "episodes": 1200, "strengths": [0., .2, 1.], "seed": 20261024}
SUFFIXES = {".py", ".sh", ".txt", ".yml"}


def procedure():
    ppo = asdict(PPOConfig())
    ppo.pop("seed")
    return {"steps": 14, "noise": .3, "price": .5, "spread": .02, "max_position": 100.,
            "background_samples": 256, "masking_seed_equals_policy_seed": False, "ppo": ppo,
            "evaluation": "greedy full versus repeated whole-row replacement",
            "diagnostic_nulls": "undefined training log metrics and zero-scale legacy scores"}


def validate_config(config):
    require(set(config) == set(STANDARD) | {"jobs"}, "study configuration membership differs")
    require(all(type(config[key]) is int for key in (set(STANDARD) - {"strengths"}) | {"jobs"}),
            "study counts and seeds must be integers")
    require(config["banks"] >= 2 and config["n_null"] >= 2
            and min(config["trials"], config["updates"], config["jobs"]) >= 1
            and config["episodes"] >= 2 and config["seed"] >= 0, "invalid bank study budgets")
    require(isinstance(config["strengths"], list) and config["strengths"]
            and len(set(config["strengths"])) == len(config["strengths"])
            and all(type(x) in (float, int) and np.isfinite(x) and x >= 0 for x in config["strengths"]),
            "invalid signal strengths")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(token):
        raise ValueError(f"non-finite JSON value in {path}: {token}")
    return json.loads(path.read_text(), parse_constant=invalid)


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def diagnostic_values(value):
    if isinstance(value, dict):
        return {key: diagnostic_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_values(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def source_state():
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                    cwd=ROOT, text=True).splitlines()
    names = sorted({name for name in names if (Path(name).suffix in SUFFIXES or name == PROTOCOL)
                    and (ROOT / name).is_file()})
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_source_dirty": subprocess.run(["git", "diff", "HEAD", "--quiet"], cwd=ROOT).returncode != 0,
            "untracked_source": sorted(set(names) - tracked),
            "source_sha256": {name: digest(ROOT / name) for name in names},
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "torch", "gymnasium", "scikit-learn", "stable-baselines3")},
            "python": platform.python_version(), "platform": platform.platform(),
            "runtime": {"torch_threads": torch.get_num_threads(), "environment": {
                key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")}}}


def seed_plan(config):
    plans, used = [], set()
    for index, bank_seed in enumerate(np.random.SeedSequence(config["seed"]).spawn(config["banks"])):
        count = config["n_null"] + len(config["strengths"]) * config["trials"]
        records = []
        for child in bank_seed.spawn(count):
            seeds = [int(value) for value in child.generate_state(4)]
            require(len(set(seeds)) == 4 and not (set(seeds) & used),
                    "seed collision; choose another master seed before starting the study")
            used.update(seeds)
            records.append(dict(zip(("train_seed", "evaluation_seed", "policy_seed", "masking_seed"), seeds)))
        plans.append({"bank": index, "agents": records})
    return plans


def population(config, strength, seed):
    return make_learnable_corpus(n_episodes=config["episodes"], n_steps=14,
                                 signal_strength=strength, seed=seed)


def corpus_digest(batch):
    h = hashlib.sha256()
    for field in fields(batch):
        if field.name.startswith("_"):
            continue
        value = getattr(batch, field.name)
        h.update(field.name.encode())
        if value is None:
            h.update(b"None")
        else:
            a = np.ascontiguousarray(value)
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def decisions(span, nulls):
    result = test_span_against_null(span, nulls).as_dict()
    if result["null_std"] <= 1e-12:
        result["z_score"] = result["p_normal"] = None
    ranks = rank_comparisons(span, nulls)
    methods = {"legacy": {"flagged": result["passes"], "resolved": result["verdict"] != UNRESOLVED}}
    methods.update({kind: {"flagged": row["rejected"], "resolved": True} for kind, row in ranks.items()})
    return {"result": result, "rank_comparators": ranks, "methods": methods}


def train_record(config, seeds, strength, key, directory, raw):
    training = population(config, strength, seeds["train_seed"])
    evaluation = population(config, strength, seeds["evaluation_seed"])
    normalizer = fit_normalizer(training)
    agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seeds["policy_seed"]))
    env = BinaryMarketEnv(training, max_position=100., normalizer=normalizer)
    env.reset(seed=seeds["policy_seed"])
    log = agent.train(env, n_updates=config["updates"], verbose=False)
    require(all(torch.isfinite(value).all().item() for value in agent.net.state_dict().values()),
            "non-finite trained model")
    span = endpoint_span(agent, evaluation, normalizer, seeds["masking_seed"], raw=raw, key=key)
    require(np.isfinite(span) and all(np.isfinite(raw[f"{key}_{suffix}"]).all()
                                    for suffix in ("full", "masked")), "non-finite endpoint returns")
    checkpoint = f"checkpoints/{key}.pt"
    logfile = f"checkpoints/{key}_log.json"
    agent.save(str(directory / checkpoint))
    write(directory / logfile, diagnostic_values(log.as_dict()))
    return {"key": key, **seeds, "signal_strength": strength, "span": span,
            "mean_return": float(raw[f"{key}_full"].mean()),
            "normalizer": normalizer.state_dict(), "checkpoint": checkpoint, "training_log": logfile,
            "train_corpus_sha256": corpus_digest(training),
            "evaluation_corpus_sha256": corpus_digest(evaluation)}


def run_bank(config, plan, out):
    torch.set_num_threads(1)
    directory = Path(out) / f"bank_{plan['bank']:03d}"
    (directory / "checkpoints").mkdir(parents=True)
    raw, references, cases = {}, [], []
    for index in range(config["n_null"]):
        references.append(train_record(config, plan["agents"][index], 0., f"reference_{index}", directory, raw))
    nulls = [row["span"] for row in references]
    for case_index, strength in enumerate(config["strengths"]):
        rows = []
        for trial in range(config["trials"]):
            position = config["n_null"] + case_index * config["trials"] + trial
            row = train_record(config, plan["agents"][position], strength,
                               f"strength_{case_index}_trial_{trial}", directory, raw)
            row.update(decisions(row["span"], nulls))
            rows.append(row)
        cases.append({"signal_strength": strength, "conditional": summarize(rows), "trials": rows})
        print(f"bank {plan['bank'] + 1}/{config['banks']}, strength {strength:g}: "
              f"{sum(row['methods']['legacy']['flagged'] for row in rows)}/{len(rows)} legacy flags", flush=True)
    payload = {"bank": plan["bank"], "references": references, "null_spans": nulls, "cases": cases,
               "endpoint_returns_file": "endpoints.npz"}
    np.savez_compressed(directory / "endpoints.npz", **raw)
    write(directory / "bank.json", payload)
    return plan["bank"]


def safe_path(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"missing or escaping artifact: {name}")
    return path


def validate_bank(out, config, plan, *, replay=False):
    directory = out / f"bank_{plan['bank']:03d}"
    payload = read(directory / "bank.json")
    require(payload["bank"] == plan["bank"] and len(payload["references"]) == config["n_null"],
            "reference bank membership differs")
    require(payload["endpoint_returns_file"] == "endpoints.npz", "endpoint archive name differs")
    require([case["signal_strength"] for case in payload["cases"]] == config["strengths"]
            and all(len(case["trials"]) == config["trials"] for case in payload["cases"]),
            "bank case/trial membership differs")
    records = payload["references"] + [row for case in payload["cases"] for row in case["trials"]]
    keys = [f"reference_{index}" for index in range(config["n_null"])] + [
        f"strength_{case}_trial_{trial}" for case in range(len(config["strengths"]))
        for trial in range(config["trials"])]
    require([row["key"] for row in records] == keys, "bank endpoint keys differ")
    with np.load(safe_path(directory, payload["endpoint_returns_file"]), allow_pickle=False) as raw:
        require(set(raw.files) == {f"{key}_{suffix}" for key in keys for suffix in ("full", "masked", "open_epoch")},
                "endpoint archive membership differs")
        for row, seeds in zip(records, plan["agents"], strict=True):
            require({key: row[key] for key in seeds} == seeds, "recorded seed plan differs")
            arrays = [raw[f"{row['key']}_{suffix}"] for suffix in ("full", "masked", "open_epoch")]
            require(all(a.shape == (config["episodes"],) and np.isfinite(a).all() for a in arrays)
                    and np.array_equal(arrays[2], np.arange(config["episodes"], dtype=float) * 900.),
                    "endpoint shape, identity or finiteness differs")
            reconstructed(float(arrays[0].mean() - arrays[1].mean()), row["span"], "bank endpoint span")
            reconstructed(float(arrays[0].mean()), row["mean_return"], "bank endpoint return")
            log = read(safe_path(directory, row["training_log"]))
            require(log["updates"] == list(range(config["updates"])), "bank training budget differs")
            require(row["checkpoint"] == f"checkpoints/{row['key']}.pt"
                    and row["training_log"] == f"checkpoints/{row['key']}_log.json",
                    "model/log membership differs")
            checkpoint = safe_path(directory, row["checkpoint"])
            if replay:
                training = population(config, row["signal_strength"], seeds["train_seed"])
                evaluation = population(config, row["signal_strength"], seeds["evaluation_seed"])
                require(corpus_digest(training) == row["train_corpus_sha256"]
                        and corpus_digest(evaluation) == row["evaluation_corpus_sha256"], "corpus replay hash differs")
                normalizer = fit_normalizer(training)
                reconstructed(normalizer.state_dict(), row["normalizer"], "training normalizer")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                require(saved["cfg"] == asdict(PPOConfig(seed=seeds["policy_seed"])), "checkpoint configuration differs")
                require(all(torch.isfinite(value).all().item() for value in saved["model"].values()),
                        "non-finite checkpoint weights")
                agent = PPOAgent(N_FEATURES, 3, PPOConfig(seed=seeds["policy_seed"]))
                agent.net.load_state_dict(saved["model"])
                replayed = {}
                endpoint_span(agent, evaluation, normalizer, seeds["masking_seed"], raw=replayed, key=row["key"])
                require(all(np.allclose(replayed[f"{row['key']}_{suffix}"], a, rtol=0, atol=1e-9)
                            for suffix, a in zip(("full", "masked"), arrays[:2], strict=True)),
                        "checkpoint replay differs from archived endpoints")
        nulls = [row["span"] for row in payload["references"]]
        reconstructed(nulls, payload["null_spans"], "bank reference spans")
        for row in payload["references"]:
            require(row["signal_strength"] == 0., "reference population must have zero strength")
        for case in payload["cases"]:
            for row in case["trials"]:
                require(row["signal_strength"] == case["signal_strength"], "observed population differs")
                reconstructed(decisions(row["span"], nulls), {key: row[key] for key in
                    ("result", "rank_comparators", "methods")}, "bank decision")
            reconstructed(summarize(case["trials"]), case["conditional"], "conditional bank rates")
    return payload


def verify_sources(provenance, allow_quick):
    if allow_quick and (provenance["tracked_source_dirty"] or provenance["untracked_source"]):
        require(source_state()["source_sha256"] == provenance["source_sha256"], "execution source changed")
        return
    require(not provenance["tracked_source_dirty"] and not provenance["untracked_source"], "study source was not clean")
    commit = provenance["commit"]
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "invalid training commit")
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True).splitlines()
    expected = {name for name in names if Path(name).suffix in SUFFIXES or name == PROTOCOL}
    require(set(provenance["source_sha256"]) == expected, "training source membership differs")
    for name, expected_hash in provenance["source_sha256"].items():
        contents = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        require(hashlib.sha256(contents).hexdigest() == expected_hash, f"historical source hash differs: {name}")
        if name.startswith("services/agent-rl/nano_rl/") or name in {
            "services/agent-rl/scripts/calibrate_reference_banks.py", "services/agent-rl/scripts/bank_calibration_statistics.py",
            "services/agent-rl/scripts/sanity_check_explanations.py", "services/agent-rl/scripts/calibrate_detection.py",
            "services/agent-rl/scripts/portable_validation.py", "services/agent-rl/scripts/run_manifest.py"}:
            require(digest(ROOT / name) == expected_hash, f"reconstruction source differs: {name}")


def verify(out, *, allow_quick=False, replay=False):
    manifest = read(out / "study_manifest.json")
    require(manifest["schema_version"] == 1 and manifest["procedure"] == procedure(),
            "study schema or procedure differs")
    require(manifest["phase"] == "complete", "bank study has not completed")
    config = manifest["config"]
    validate_config(config)
    full = all(config[key] == value for key, value in STANDARD.items())
    require(manifest["scientific_budgets_checked"] is full, "study budget flag differs")
    require(full or allow_quick, "reduced/custom studies cannot verify the full scientific protocol")
    verify_sources(manifest["provenance"], allow_quick)
    require(manifest["seed_plan"] == seed_plan(config), "study seed membership differs")
    files = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p != out / "study_manifest.json"}
    expected = {"summary.json"}
    for plan in manifest["seed_plan"]:
        prefix = f"bank_{plan['bank']:03d}"
        expected.update({f"{prefix}/bank.json", f"{prefix}/endpoints.npz"})
        keys = [f"reference_{index}" for index in range(config["n_null"])] + [
            f"strength_{case}_trial_{trial}" for case in range(len(config["strengths"]))
            for trial in range(config["trials"])]
        expected.update(f"{prefix}/checkpoints/{key}{suffix}"
                        for key in keys for suffix in (".pt", "_log.json"))
    require(files == set(manifest["artifacts_sha256"]) == expected, "bank artifact membership differs")
    for name, expected in manifest["artifacts_sha256"].items():
        require(digest(safe_path(out, name)) == expected, f"artifact hash differs: {name}")
    banks = [validate_bank(out, config, plan, replay=replay) for plan in manifest["seed_plan"]]
    corpus_hashes = [row[key] for bank in banks for row in
                    bank["references"] + [row for case in bank["cases"] for row in case["trials"]]
                    for key in ("train_corpus_sha256", "evaluation_corpus_sha256")]
    require(all(isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value) for value in corpus_hashes)
            and len(set(corpus_hashes)) == len(corpus_hashes), "corpus fingerprints are invalid or reused")
    reconstructed(summarize_banks(banks), read(out / "summary.json"), "bank-level study summary")
    return {"banks_checked": len(banks), "models_archived": sum(len(plan["agents"]) for plan in manifest["seed_plan"]),
            "endpoints_replayed": replay, "scientific_budgets_checked": full,
            "source_commit": manifest["provenance"]["commit"]}


def run(config, out):
    validate_config(config)
    torch.set_num_threads(1)
    require(not out.exists() or not any(out.iterdir()), "refusing nonempty study output")
    require(not any(out.resolve().is_relative_to(ROOT / name) for name in ("data/releases", "docs")),
            "study outputs must be separate from released evidence and manuscripts")
    plans = seed_plan(config)
    provenance = source_state()
    full = all(config[key] == value for key, value in STANDARD.items())
    require(not full or (not provenance["tracked_source_dirty"] and not provenance["untracked_source"]),
            "full study requires committed clean source")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "phase": "started", "config": config, "seed_plan": plans,
                "provenance": provenance, "scientific_budgets_checked": full,
                "design": "fresh reference and observed training/evaluation corpora in every independent bank",
                "procedure": procedure()}
    write(out / "study_manifest.json", manifest)
    started = time.monotonic()
    if config["jobs"] == 1:
        for plan in plans:
            run_bank(config, plan, str(out))
    else:
        with ProcessPoolExecutor(max_workers=min(config["jobs"], config["banks"]),
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(run_bank, config, plan, str(out)) for plan in plans]
            for future in futures:
                future.result()
    banks = [validate_bank(out, config, plan) for plan in plans]
    write(out / "summary.json", summarize_banks(banks))
    require(source_state() == provenance, "source or runtime changed during the bank study")
    manifest.update(phase="complete", seconds=time.monotonic() - started,
                    artifacts_sha256={str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*"))
                                      if p.is_file() and p != out / "study_manifest.json"})
    write(out / "study_manifest.json", manifest)
    return verify(out, allow_quick=not full)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    for name in ("banks", "n-null", "trials", "updates", "episodes", "seed"):
        ap.add_argument(f"--{name}", type=int, default=STANDARD[name.replace('-', '_')])
    ap.add_argument("--strengths", type=float, nargs="+", default=STANDARD["strengths"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--allow-quick", action="store_true")
    ap.add_argument("--replay-models", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.verify_only:
        result = verify(args.out, allow_quick=args.allow_quick, replay=args.replay_models)
    else:
        config = {key: getattr(args, key) for key in (*STANDARD, "jobs")}
        if args.quick:
            config.update(banks=2, n_null=2, trials=2, updates=1, episodes=32, strengths=[0., 1.])
        else:
            check_resolving_power(config["n_null"], what="the independent bank reference")
        result = run(config, args.out.resolve())
        if args.replay_models:
            result = verify(args.out, allow_quick=not result["scientific_budgets_checked"], replay=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
