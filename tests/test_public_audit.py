import subprocess
from pathlib import Path

from scripts.audit_public_repo import iter_files, scan_text_file


def test_public_audit_detects_host_alias_addresses_and_tokens(tmp_path):
    path = tmp_path / "private.md"
    host_alias = "ssh " + "server"
    address = "10" + ".20.30.40"
    token = "ghp" + "_" + "123456789012345678901234567890"
    path.write_text(
        f"{host_alias}\nserver={address}\n{token}\n",
        encoding="utf-8",
    )

    hits = scan_text_file(path)

    assert any("ssh" in hit for hit in hits)
    assert any("\\d{1,3}" in hit for hit in hits)
    assert any("ghp" in hit for hit in hits)


def test_public_audit_ignores_safe_public_text(tmp_path):
    path = tmp_path / "public.md"
    path.write_text(
        "Install with python -m pip install -e .\nUse explicit input paths.\n",
        encoding="utf-8",
    )

    assert scan_text_file(path) == []


def test_public_audit_skips_gitignored_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored.log\n__pycache__/\n", encoding="utf-8")
    (tmp_path / "public.py").write_text("print('safe')\n", encoding="utf-8")
    (tmp_path / "ignored.log").write_text("generated\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"generated")

    files = iter_files(tmp_path, include_ignored=False)

    assert Path("public.py") in files
    assert Path("ignored.log") not in files
    assert Path("__pycache__/module.pyc") not in files
