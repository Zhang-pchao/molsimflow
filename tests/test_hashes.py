from pathlib import Path

import pytest

from molsimflow.io.hashes import read_sha256_manifest, verify_sha256_manifest


def test_verify_sha256_manifest_with_explicit_replacement(tmp_path):
    stable = tmp_path / "stable.txt"
    stable.write_text("content\n", encoding="utf-8")
    transient = Path("/var/spool/slurmd/job1/slurm_script")
    digest = "434728a410a78f56fc1b5899c3593436e61ab0c731e9072d95e96db290205e53"
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{digest}  {transient}\n", encoding="utf-8")

    records = verify_sha256_manifest(manifest, replacements={transient: stable})
    assert records == read_sha256_manifest(manifest)


def test_verify_sha256_manifest_rejects_unavailable_or_mismatched_input(tmp_path):
    missing = Path("/definitely/missing/input")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("0" * 64 + f"  {missing}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable input"):
        verify_sha256_manifest(manifest)

    available = tmp_path / "available.txt"
    available.write_text("different\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_sha256_manifest(manifest, replacements={missing: available})
