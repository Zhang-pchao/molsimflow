"""Read and verify portable SHA256 manifests with explicit path substitutions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class ChecksumRecord:
    """One GNU ``sha256sum`` manifest record."""

    digest: str
    path: Path


def read_sha256_manifest(path: Path) -> list[ChecksumRecord]:
    """Return strict records from a GNU two-column SHA256 manifest."""
    records = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, filename = raw.partition("  ")
        if not separator or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{path}:{line_number}: invalid SHA256 manifest record")
        if not filename:
            raise ValueError(f"{path}:{line_number}: empty manifest path")
        records.append(ChecksumRecord(digest=digest, path=Path(filename)))
    if not records:
        raise ValueError(f"{path}: manifest is empty")
    return records


def _sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256_manifest(
    manifest: Path,
    *,
    replacements: Mapping[Path, Path] | None = None,
) -> list[ChecksumRecord]:
    """Verify a manifest, allowing only caller-declared missing-path replacements.

    A replacement is deliberately keyed by the original path recorded in the
    manifest. This makes transient scheduler paths auditable without silently
    ignoring any missing or changed input.
    """
    replacement_map = {Path(source): Path(target) for source, target in (replacements or {}).items()}
    records = read_sha256_manifest(manifest)
    for record in records:
        source = record.path if record.path.is_file() else replacement_map.get(record.path)
        if source is None or not source.is_file():
            raise ValueError(f"{manifest}: unavailable input {record.path}")
        actual = _sha256(source)
        if actual != record.digest:
            raise ValueError(
                f"{manifest}: SHA256 mismatch for {record.path}; expected {record.digest}, got {actual}"
            )
    return records
