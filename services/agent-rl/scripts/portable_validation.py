"""Read frozen experiment summaries across CPUs without changing their evidence.

The original producer validates its outputs on one runtime using exact dict
equality. Portable reading reconstructs the same procedures and permits only
small numeric roundoff. Counts, decisions, metadata and membership stay exact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run_manifest import test_span_against_null
from nano_rl.explain.rank import rank_comparisons

ABSOLUTE_TOLERANCE = 1e-9


def reconstructed(actual, recorded, context, differences=None):
    def fail():
        raise ValueError(f"reconstruction differs at {context}: {actual!r} != {recorded!r}")
    if isinstance(recorded, dict):
        if not isinstance(actual, dict) or actual.keys() != recorded.keys():
            fail()
        for key in recorded:
            reconstructed(actual[key], recorded[key], f"{context}/{key}", differences)
    elif isinstance(recorded, list):
        if not isinstance(actual, list) or len(actual) != len(recorded):
            fail()
        for index, (a, b) in enumerate(zip(actual, recorded, strict=True)):
            reconstructed(a, b, f"{context}/{index}", differences)
    elif isinstance(recorded, bool):
        if not isinstance(actual, bool) or actual is not recorded:
            fail()
    elif isinstance(recorded, int):
        if not isinstance(actual, int) or isinstance(actual, bool) or actual != recorded:
            fail()
    elif isinstance(recorded, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            fail()
        if math.isnan(actual) or math.isnan(recorded):
            fail()
        if actual == recorded:
            return
        if not (math.isfinite(actual) and math.isfinite(recorded)
                and abs(actual - recorded) <= ABSOLUTE_TOLERANCE):
            fail()
        if differences is not None:
            differences.append((abs(actual - recorded), context, actual, recorded))
    elif type(actual) is not type(recorded) or actual != recorded:
        fail()


def validate_primary_endpoints(out: Path, differences=None):
    def endpoint(raw, key, count):
        full, masked, identities = (raw[f"{key}_{suffix}"] for suffix in ("full", "masked", "open_epoch"))
        if any(values.shape != (count,) or not np.isfinite(values).all()
               for values in (full, masked, identities)) or len(np.unique(identities)) != count:
            raise ValueError(f"invalid primary endpoint returns or episode identities: {key}")
        return float(full.mean() - masked.mean()), float(full.mean())

    def comparison(raw, key, count, nulls, recorded):
        span, mean = endpoint(raw, key, count)
        reconstructed(test_span_against_null(span, nulls).as_dict(), recorded,
                      f"primary comparison/{key}", differences)
        return mean

    for path in sorted(out.rglob("sanity_test.json")):
        payload = json.loads(path.read_text())
        with np.load(path.parent / payload["endpoint_returns_file"], allow_pickle=False) as raw:
            nulls = [endpoint(raw, key, payload["synthetic_episodes"])[0]
                     for key in payload["reference_endpoint_keys"]]
            reconstructed(nulls, payload["null_spans"], "primary reference spans", differences)
            for name, key in payload["observed_endpoint_keys"].items():
                count = (payload["endpoint_evaluation_episodes"] if name == "real_market"
                         else payload["synthetic_episodes"])
                comparison(raw, key, count, nulls, payload["results"][name])
            for key, span, result in zip(payload["real_endpoint_keys"], payload["real_spans_per_seed"],
                                         payload["real_results_per_seed"], strict=True):
                comparison(raw, key, payload["endpoint_evaluation_episodes"], nulls, result)
                reconstructed(result["statistic"], span, f"primary market span/{key}", differences)

    from calibrate_detection import summarize
    for path in sorted(out.rglob("detection_calibration.json")):
        payload = json.loads(path.read_text())
        with np.load(path.parent / payload["endpoint_returns_file"], allow_pickle=False) as raw:
            nulls = [endpoint(raw, key, payload["synthetic_episodes"])[0]
                     for key in payload["reference_endpoint_keys"]]
            reconstructed(nulls, payload["null_spans"], "calibration reference spans", differences)
            for case in payload["cases"]:
                for row in case["trials"]:
                    key = row["endpoint_key"]
                    mean = comparison(raw, key, row["data"]["evaluation_episodes"], nulls, row["result"])
                    reconstructed(mean, row["return"], f"calibration return/{key}", differences)
                    if "rank_comparators" in row:
                        reconstructed(rank_comparisons(row["result"]["statistic"], nulls),
                                      row["rank_comparators"], f"calibration ranks/{key}", differences)
                expected = summarize(case["trials"])
                reconstructed(expected, {key: case[key] for key in expected},
                              "calibration rate", differences)


def validate_reference_bootstraps(out: Path, differences=None):
    from bootstrap_z import bootstrap, collect
    for path in sorted(out.rglob("z_intervals.json")):
        payload = json.loads(path.read_text())
        expected = {label: bootstrap(observed, nulls, payload[label]["n_boot"], payload[label]["seed"])
                    for label, observed, nulls in collect(path.parent)}
        reconstructed(expected, payload, "reference resampling summary", differences)


def validate_horizon_returns(out: Path, differences=None):
    from horizon_scaling import fit_power_law
    for path in sorted(out.rglob("horizon_scaling.json")):
        payload = json.loads(path.read_text())
        count = payload["evaluation_episodes"]
        with np.load(path.parent / payload["endpoint_returns_file"], allow_pickle=False) as raw:
            for row in payload["rows"]:
                if "clock_feature_deadline_seconds" in row:
                    reconstructed(60 * (row["n_steps"] + 1), row["clock_feature_deadline_seconds"],
                                  "horizon clock deadline", differences)
                widths = {}
                for arm in ("weight", "env"):
                    spans = []
                    for key in row["endpoint_keys"][arm]:
                        full, masked, ids = [raw[f"{key}_{suffix}"]
                                             for suffix in ("full", "masked", "open_epoch")]
                        if (any(a.shape != (count,) or not np.isfinite(a).all() for a in (full, masked, ids))
                                or len(np.unique(ids)) != count):
                            raise ValueError(f"invalid horizon endpoint returns or identities: {key}")
                        spans.append(float(full.mean() - masked.mean()))
                    widths[arm] = float(np.std(spans, ddof=1))
                    reconstructed(spans, row[f"{arm}_spans"], f"horizon {arm} spans", differences)
                    reconstructed(widths[arm], row[f"{arm}_std"], f"horizon {arm} width", differences)
                ratio = widths["weight"] / widths["env"] if widths["env"] > 1e-12 else None
                reconstructed(ratio, row["ratio"], "horizon width ratio", differences)
        x = np.array([row["n_steps"] for row in payload["rows"]], dtype=float)
        for arm in ("weight", "env"):
            y = np.array([row[f"{arm}_std"] for row in payload["rows"]])
            slope, quality = fit_power_law(x, y)
            reconstructed(slope, payload.get(f"alpha_{arm}", slope), f"horizon {arm} fit slope", differences)
            reconstructed(quality, payload.get(f"r2_{arm}", quality), f"horizon {arm} fit quality", differences)
            if "fit_horizons" in payload:
                reconstructed(x[y > 0].tolist(), payload["fit_horizons"][arm],
                              f"horizon {arm} fit membership", differences)
