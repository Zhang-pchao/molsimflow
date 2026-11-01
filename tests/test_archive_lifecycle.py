import json
import shutil
from datetime import datetime, timedelta, timezone

import pytest

from molsimflow.io.archive_lifecycle import (
    cleanup, cleanup_plan, document_digest, snapshot, utc_now, verify,
)


@pytest.fixture
def package(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "initial.data").write_text("protected input")
    (source / "trajectory.zst").write_bytes(b"archived output")
    destination = tmp_path / "archive"
    shutil.copytree(source, destination)
    manifest = snapshot(source, source_id="run-1", removable=["trajectory.zst"])
    receipt = verify(manifest, destination)
    plan = cleanup_plan(manifest, receipt, allowlist=["trajectory.zst"])
    evidence = tmp_path / "dependency-audit.json"
    evidence.write_text("fresh dependency audit")
    from molsimflow.io.hashes import _sha256
    proof = {"plan_sha256": document_digest(plan), "checked_utc": utc_now(),
             "writers_stopped": True, "dependencies_clear": True, "active_references": [],
             "evidence": [{"path": str(evidence), "sha256": _sha256(evidence)}]}
    return source, destination, manifest, receipt, plan, proof


def execute(package, tmp_path):
    source, _, manifest, receipt, plan, proof = package
    return cleanup(manifest, receipt, plan, proof,
                   expected_plan_sha256=document_digest(plan), source_root=source,
                   journal=tmp_path / "journal.jsonl")


def test_plan_and_verified_cleanup_keep_inputs(package, tmp_path):
    source, destination, _, _, _, _ = package
    assert (source / "trajectory.zst").exists()  # planning is non-mutating
    result = execute(package, tmp_path)
    assert result["status"] == "SOURCE_OUTPUTS_REMOVED"
    assert result["logical_bytes_removed"] == len(b"archived output")
    assert not (source / "trajectory.zst").exists()
    assert (source / "initial.data").read_text() == "protected input"
    assert (destination / "trajectory.zst").exists()
    records = [json.loads(s) for s in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert records[-1]["event"] == "POST_AUDIT_PASS"
    assert [x["event"] for x in records] == ["START", "DELETE_INTENT", "DELETED", "POST_AUDIT_PASS"]


@pytest.mark.parametrize("change", ["changed_output", "changed_input", "extra", "symlink", "missing"])
def test_changed_source_fails_without_deletion(package, tmp_path, change):
    source = package[0]
    if change == "changed_output":
        (source / "trajectory.zst").write_bytes(b"new output")
    elif change == "changed_input":
        (source / "initial.data").write_text("new input")
    elif change == "extra":
        (source / "unexpected").write_text("new file")
    elif change == "symlink":
        (source / "link").symlink_to(source / "initial.data")
    else:
        (source / "initial.data").unlink()
    with pytest.raises(ValueError):
        execute(package, tmp_path)
    assert (source / "trajectory.zst").exists()
    assert not (tmp_path / "journal.jsonl").exists()


@pytest.mark.parametrize("change", ["writer", "dependency", "stale", "wrong_plan", "no_evidence"])
def test_dependency_gates(package, tmp_path, change):
    proof = package[5]
    if change == "writer":
        proof["writers_stopped"] = False
    elif change == "dependency":
        proof["active_references"] = ["consumer"]
    elif change == "stale":
        proof["checked_utc"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    elif change == "wrong_plan":
        proof["plan_sha256"] = "bad"
    else:
        proof["evidence"] = []
    with pytest.raises(ValueError):
        execute(package, tmp_path)
    assert (package[0] / "trajectory.zst").exists()


@pytest.mark.parametrize("path", ["initial.data", "../outside", "/outside", "a/../trajectory.zst"])
def test_allowlist_rejects_input_and_escape(package, path):
    with pytest.raises(ValueError):
        cleanup_plan(package[2], package[3], allowlist=[path])


def test_wrong_destination_and_plan_digest(package, tmp_path):
    (package[1] / "trajectory.zst").write_text("corrupt")
    with pytest.raises(ValueError):
        verify(package[2], package[1])
    with pytest.raises(ValueError):
        cleanup(package[2], package[3], package[4], package[5], expected_plan_sha256="bad",
                source_root=package[0], journal=tmp_path / "journal.jsonl")
    assert (package[0] / "trajectory.zst").exists()


def test_root_guard_and_existing_journal(package, tmp_path):
    with pytest.raises(ValueError):
        cleanup(package[2], package[3], package[4], package[5],
                expected_plan_sha256=document_digest(package[4]), source_root=package[1],
                journal=tmp_path / "journal.jsonl")
    (tmp_path / "journal.jsonl").write_text("prior interrupted attempt")
    with pytest.raises(ValueError):
        execute(package, tmp_path)
    assert (package[0] / "trajectory.zst").exists()


def test_mutated_dependency_evidence_fails(package, tmp_path):
    (tmp_path / "dependency-audit.json").write_text("changed audit")
    with pytest.raises(ValueError):
        execute(package, tmp_path)
    assert (package[0] / "trajectory.zst").exists()
