"""Process isolation must preserve every control result and raw episode record."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL / "scripts"))
from run_manifest import validate_control_returns


def test_spawned_control_workers_match_serial_execution(tmp_path):
    env = dict(os.environ, ALLOW_UNRESOLVED_NULLS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    outputs = []
    for jobs in (1, 2):
        out = tmp_path / str(jobs)
        result = subprocess.run([
            sys.executable, str(RL / "scripts/generalize_gym.py"),
            "--envs", "FrozenLake-v1", "CliffWalking-v0", "--trainer", "sb3",
            "--steps", "64", "--null-steps", "64", "--n-null", "2",
            "--agent-seeds", "2", "--attr-episodes", "2", "--jobs", str(jobs), "--out", str(out),
        ], cwd=RL, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        validate_control_returns(out)
        rows = json.loads((out / "generalize_gym.json").read_text())
        assert [row["env_id"] for row in rows] == ["FrozenLake-v1", "CliffWalking-v0"]
        for row in rows:
            execution = row.pop("execution")
            assert execution["workers"] == jobs
            assert execution["torch_threads"] == 1
            assert execution["start_method"] == ("spawn" if jobs > 1 else "single process")
        outputs.append(rows)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_nonpositive_worker_count_is_rejected(jobs):
    result = subprocess.run([sys.executable, str(RL / "scripts/generalize_gym.py"),
                             "--jobs", jobs], cwd=RL, capture_output=True, text=True)
    assert result.returncode != 0
    assert "jobs must be positive" in result.stderr
