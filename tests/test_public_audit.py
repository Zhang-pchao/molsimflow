from scripts.audit_public_repo import scan_text_file


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
