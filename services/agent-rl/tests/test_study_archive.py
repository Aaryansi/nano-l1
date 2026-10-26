"""Compact study distribution preserves bytes and rejects unsafe/altered members."""
import hashlib
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from study_archive import digest, pack, unpack


def source(tmp_path):
    root = tmp_path / "source"
    (root / "models").mkdir(parents=True)
    (root / "models/a.pt").write_bytes(bytes(range(256)))
    (root / "manifest.json").write_text('{"complete": true}\n')
    return root


def test_round_trip_is_exact_deterministic_and_does_not_overwrite(tmp_path):
    root = source(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    pack(root, a)
    pack(root, b)
    assert (a / "study.zip").read_bytes() == (b / "study.zip").read_bytes()
    assert (a / "archive.json").read_bytes() == (b / "archive.json").read_bytes()
    output = tmp_path / "restored"
    unpack(a, output)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="existing"):
        unpack(a, output)
    with pytest.raises(ValueError, match="new separate"):
        pack(root, a)


@pytest.mark.parametrize("change", ["archive_hash", "member_hash", "size", "extra", "too_large"])
def test_corruption_rejects_without_installing_partial_evidence(tmp_path, change):
    bundle, output = tmp_path / "bundle", tmp_path / "restored"
    pack(source(tmp_path), bundle)
    index = json.loads((bundle / "archive.json").read_text())
    if change == "archive_hash":
        index["archive_sha256"] = "0" * 64
    elif change == "member_hash":
        index["members"]["manifest.json"]["sha256"] = "0" * 64
    elif change == "size":
        index["members"]["manifest.json"]["bytes"] += 1
    elif change == "extra":
        with zipfile.ZipFile(bundle / "study.zip", "a") as z:
            z.writestr("extra", b"x")
        index["archive_sha256"] = digest(bundle / "study.zip")
    else:
        index["members"]["manifest.json"]["bytes"] = 2 * 1024 ** 3
    (bundle / "archive.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        unpack(bundle, output)
    assert not output.exists()


@pytest.mark.parametrize("name,mode", [("../escape", stat.S_IFREG), ("/absolute", stat.S_IFREG),
    ("a/../escape", stat.S_IFREG), ("a\\escape", stat.S_IFREG), (".", stat.S_IFREG),
    ("link", stat.S_IFLNK), ("directory/", stat.S_IFDIR)])
def test_noncanonical_paths_and_nonregular_files_are_rejected(tmp_path, name, mode):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    data = b"elsewhere"
    with zipfile.ZipFile(bundle / "study.zip", "w") as z:
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = (mode | 0o644) << 16
        z.writestr(info, data)
    (bundle / "archive.json").write_text(json.dumps({"schema_version": 1,
        "archive_sha256": digest(bundle / "study.zip"),
        "members": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}))
    with pytest.raises(ValueError):
        unpack(bundle, tmp_path / "output")
    assert not (tmp_path / "output").exists() and not (tmp_path / "escape").exists()


def test_source_symlinks_cannot_enter_a_study_archive(tmp_path):
    root = source(tmp_path)
    (root / "alias").symlink_to(root / "manifest.json")
    with pytest.raises(ValueError, match="symlinks"):
        pack(root, tmp_path / "bundle")
