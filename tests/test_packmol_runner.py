import sys
from pathlib import Path

from molsimflow.structure.packmol import PackmolInput, infer_packmol_output_xyz, run_packmol, write_packmol_input


def test_infer_packmol_output_xyz(tmp_path: Path):
    packmol_path = tmp_path / "packmol.in"
    packmol_path.write_text(
        "tolerance 2.4\nfiletype xyz\noutput packed.xyz\nseed -1\n",
        encoding="utf-8",
    )
    assert infer_packmol_output_xyz(packmol_path) == "packed.xyz"


def test_run_packmol_with_fake_command(tmp_path: Path):
    packmol_path = write_packmol_input(
        PackmolInput(output_xyz="packed.xyz", structures=[]),
        tmp_path / "packmol.in",
    )
    fake = tmp_path / "fake_packmol.py"
    fake.write_text(
        "\n".join(
            [
                "import pathlib, sys",
                "text = sys.stdin.read()",
                "output = 'packed.xyz'",
                "for line in text.splitlines():",
                "    parts = line.split(maxsplit=1)",
                "    if len(parts) == 2 and parts[0].lower() == 'output':",
                "        output = parts[1]",
                "pathlib.Path(output).write_text('1\\nfake\\nX 0 0 0\\n')",
                "print('fake packmol ok')",
            ]
        ),
        encoding="utf-8",
    )
    result = run_packmol(packmol_path, command=[sys.executable, str(fake)], cwd=tmp_path)
    assert result.returncode == 0
    assert result.output_xyz.exists()
    assert "fake packmol ok" in result.log_path.read_text(encoding="utf-8")
