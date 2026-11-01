"""Explicit manifests and guarded, journalled archival cleanup.

No scheduler discovery, network transfer or implicit deletion lives here. Callers
supply fresh writer/dependency evidence and verify the destination separately.
"""
from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .hashes import _sha256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def document_digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _root(path: Path) -> Path:
    path = Path(path).absolute()
    if path != path.resolve(strict=True) or not path.is_dir() or path == Path("/"):
        raise ValueError("Root must be an existing, non-symlink directory other than /")
    return path


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def _file(root: Path, value: str) -> Path:
    path = root / _relative(value)
    current = root
    for part in PurePosixPath(value).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Symlink refused: {current}")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Not a regular file: {path}")
    return path


def _files(root: Path) -> list[str]:
    result = []
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError(f"Symlink refused: {path}")
        for name in files:
            path = Path(base) / name
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError(f"Non-regular file refused: {path}")
            result.append(path.relative_to(root).as_posix())
    return sorted(result)


def snapshot(root: Path, *, source_id: str, removable: list[str]) -> dict:
    """Hash every file; only the explicit removable list gets output role."""
    root = _root(root)
    allowed = {_relative(value) for value in removable}
    paths = _files(root)
    if not allowed.issubset(paths):
        raise ValueError("Removable paths must exist in this snapshot")
    entries = []
    for value in paths:
        path = _file(root, value)
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino
        ):
            raise ValueError(f"File changed while hashing: {value}")
        entries.append({"path": value, "size": after.st_size, "sha256": digest,
                        "mtime_ns": after.st_mtime_ns, "inode": after.st_ino,
                        "role": "removable_output" if value in allowed else "retained"})
    identity = root.stat()
    return {"schema": 1, "source_id": source_id, "root": str(root),
            "root_identity": [identity.st_dev, identity.st_ino],
            "created_utc": utc_now(), "files": entries}


def verify(manifest: dict, destination: Path) -> dict:
    """Require an exact file set and full SHA256 equality on the archive."""
    destination = _root(destination)
    expected = [item["path"] for item in manifest["files"]]
    if len(set(expected)) != len(expected) or sorted(expected) != _files(destination):
        raise ValueError("Destination file set differs from manifest")
    for item in manifest["files"]:
        path = _file(destination, item["path"])
        if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"Archive content differs: {item['path']}")
    return {"status": "ARCHIVE_VERIFIED", "manifest_sha256": document_digest(manifest),
            "destination": str(destination), "created_utc": utc_now(),
            "file_count": len(expected), "logical_bytes": sum(x["size"] for x in manifest["files"])}


def cleanup_plan(manifest: dict, archive_receipt: dict, *, allowlist: list[str]) -> dict:
    """Freeze an explicit allowlist; this function never modifies source files."""
    digest = document_digest(manifest)
    if (archive_receipt.get("status") != "ARCHIVE_VERIFIED"
            or archive_receipt.get("manifest_sha256") != digest):
        raise ValueError("An archive verification receipt for this exact manifest is required")
    entries = {x["path"]: x for x in manifest["files"]}
    allowed = [_relative(value) for value in allowlist]
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("An explicit, nonempty, duplicate-free allowlist is required")
    for value in allowed:
        if value not in entries or entries[value]["role"] != "removable_output":
            raise ValueError(f"Not explicitly removable: {value}")
    return {"schema": 1, "manifest_sha256": digest,
            "archive_receipt_sha256": document_digest(archive_receipt),
            "source_root": manifest["root"], "allowlist": sorted(allowed),
            "created_utc": utc_now()}


def _proof(proof: dict, plan: dict, now: datetime) -> None:
    if proof.get("plan_sha256") != document_digest(plan):
        raise ValueError("Dependency proof does not identify this exact cleanup plan")
    if proof.get("writers_stopped") is not True or proof.get("dependencies_clear") is not True:
        raise ValueError("Writer and dependency clearance are required")
    if proof.get("active_references") != [] or not proof.get("evidence"):
        raise ValueError("No active references and explicit evidence records are required")
    for item in proof["evidence"]:
        digest = item.get("sha256", "")
        if not item.get("path") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Each evidence record requires a path and SHA256")
        evidence_path = Path(item["path"])
        if (not evidence_path.is_file() or evidence_path.is_symlink()
                or _sha256(evidence_path) != digest):
            raise ValueError("Dependency evidence missing or changed")
    age = (now - datetime.fromisoformat(proof["checked_utc"])).total_seconds()
    if not 0 <= age <= 900:
        raise ValueError("Dependency proof must be refreshed within 15 minutes")


def cleanup(manifest: dict, archive_receipt: dict, plan: dict, proof: dict, *,
            expected_plan_sha256: str, source_root: Path, journal: Path) -> dict:
    """Delete only a fresh, exact plan after full source rehash; never recursive.

    Journal must be new and outside the source root. An interrupted cleanup is
    intentionally not auto-resumed: retain its journal, audit, and freeze a new
    same-scope manifest/plan for remaining files if another attempt is authorized.
    """
    if document_digest(plan) != expected_plan_sha256:
        raise ValueError("Explicit plan SHA256 does not match")
    canonical = cleanup_plan(manifest, archive_receipt, allowlist=plan["allowlist"])
    for key in ("manifest_sha256", "archive_receipt_sha256", "source_root", "allowlist"):
        if canonical[key] != plan.get(key):
            raise ValueError(f"Cleanup plan field differs: {key}")
    root = _root(source_root)
    if str(root) != manifest["root"] or str(root) != plan["source_root"]:
        raise ValueError("Source root differs from frozen identity")
    identity = root.stat()
    if [identity.st_dev, identity.st_ino] != manifest["root_identity"]:
        raise ValueError("Source root inode changed")
    _proof(proof, plan, datetime.now(timezone.utc))
    journal = Path(journal).absolute()
    if journal == root or root in journal.parents or journal.exists():
        raise ValueError("Journal must be new and outside source root")
    # Exact inventory plus full content rehash protects retained inputs too.
    verify(manifest, root)
    for item in manifest["files"]:
        st = _file(root, item["path"]).stat()
        if [st.st_ino, st.st_mtime_ns] != [item["inode"], item["mtime_ns"]]:
            raise ValueError(f"Source metadata changed: {item['path']}")
    entries = {x["path"]: x for x in manifest["files"]}
    removed = []
    with journal.open("x", encoding="utf-8") as handle:
        def record(event: dict) -> None:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        record({"event": "START", "plan_sha256": expected_plan_sha256, "utc": utc_now()})
        for value in plan["allowlist"]:
            path = _file(root, value)
            item = entries[value]
            record({"event": "DELETE_INTENT", "path": value, "sha256": item["sha256"]})
            # Hold each parent directory open; refuse links at every component.
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                parts = PurePosixPath(value).parts
                for part in parts[:-1]:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = child
                st = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
                if (not stat.S_ISREG(st.st_mode) or st.st_ino != item["inode"]
                        or st.st_size != item["size"] or st.st_mtime_ns != item["mtime_ns"]):
                    raise ValueError(f"File changed before deletion: {value}")
                os.unlink(parts[-1], dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(fd)
            removed.append(value)
            record({"event": "DELETED", "path": value, "size": item["size"], "utc": utc_now()})
        retained = [x for x in manifest["files"] if x["path"] not in removed]
        if sorted(x["path"] for x in retained) != _files(root):
            raise ValueError("Post-cleanup remaining file set differs")
        for item in retained:
            path = _file(root, item["path"])
            if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
                raise ValueError(f"Retained input changed: {item['path']}")
        result = {"status": "SOURCE_OUTPUTS_REMOVED", "plan_sha256": expected_plan_sha256,
                  "removed": removed, "retained": [x["path"] for x in retained],
                  "logical_bytes_removed": sum(entries[x]["size"] for x in removed),
                  "created_utc": utc_now()}
        record({"event": "POST_AUDIT_PASS", **result})
    return result
