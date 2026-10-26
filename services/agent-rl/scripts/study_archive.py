"""Package immutable study files without losing checkpoint replay evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from run_manifest import digest

MAX_BYTES = 1024 ** 3
MAX_FILES = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member_name(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and p.parts and not p.is_absolute() and ".." not in p.parts
            and "\\" not in name and str(p) == name, "unsafe study archive path")
    return name


def pack(source, bundle):
    source, bundle = source.resolve(), bundle.resolve()
    require(source.is_dir() and not bundle.exists() and not bundle.is_relative_to(source),
            "require an existing study and a new separate archive destination")
    paths = sorted(source.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "study archive cannot contain symlinks")
    paths = [p for p in paths if p.is_file()]
    require(0 < len(paths) <= MAX_FILES and sum(p.stat().st_size for p in paths) <= MAX_BYTES,
            "study archive exceeds supported limits")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=bundle.parent, prefix=".study-pack-") as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        members = {}
        with zipfile.ZipFile(stage / "study.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in paths:
                name = member_name(path.relative_to(source).as_posix())
                data = path.read_bytes()
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
                members[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        index = {"schema_version": 1, "archive_sha256": digest(stage / "study.zip"), "members": members}
        (stage / "archive.json").write_text(json.dumps(index, indent=2) + "\n")
        unpack(stage, Path(temporary) / "checked")
        require(not bundle.exists(), "archive destination appeared while packaging")
        stage.rename(bundle)
    return index


def unpack(bundle, output):
    require(not output.exists(), "refusing existing study extraction destination")
    index = json.loads((bundle / "archive.json").read_text())
    require(set(index) == {"schema_version", "archive_sha256", "members"}
            and type(index["schema_version"]) is int and index["schema_version"] == 1,
            "invalid study archive schema")
    records = index["members"]
    require(isinstance(records, dict) and 0 < len(records) <= MAX_FILES, "invalid study archive membership")
    for name, row in records.items():
        member_name(name)
        require(set(row) == {"bytes", "sha256"} and type(row["bytes"]) is int and row["bytes"] >= 0
                and isinstance(row["sha256"], str) and len(row["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["sha256"]), "invalid study archive record")
    require(sum(row["bytes"] for row in records.values()) <= MAX_BYTES, "study archive exceeds supported limits")
    require(digest(bundle / "study.zip") == index["archive_sha256"], "study archive content hash differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle / "study.zip") as archive, tempfile.TemporaryDirectory(
            dir=output.parent, prefix=".study-unpack-") as temporary:
        infos = archive.infolist()
        require(len(infos) == len(records) and {row.filename for row in infos} == set(records),
                "study archive member list differs")
        stage = Path(temporary) / "study"
        stage.mkdir()
        for info in infos:
            name = member_name(info.filename)
            require(stat.S_ISREG(info.external_attr >> 16) and not info.is_dir()
                    and info.file_size == records[name]["bytes"], "invalid study archive member type or size")
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            require(target.stat().st_size == records[name]["bytes"]
                    and digest(target) == records[name]["sha256"], "study member content hash differs")
        require(not output.exists(), "extraction destination appeared during verification")
        stage.rename(output)
    return index


@contextmanager
def materialized(bundle):
    with tempfile.TemporaryDirectory(prefix="nano-l1-study-") as temporary:
        output = Path(temporary) / "study"
        unpack(bundle, output)
        yield output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack", "unpack"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = (pack if args.operation == "pack" else unpack)(args.source, args.out)
    print(json.dumps({"files": len(index["members"]), "archive_sha256": index["archive_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
