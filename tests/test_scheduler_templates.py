from pathlib import Path


def test_double_bubble_slurm_template_is_parameterized():
    root = Path(__file__).resolve().parents[1]
    template = root / "templates" / "slurm" / "double_bubble_preprocess.slurm"
    text = template.read_text(encoding="utf-8")
    assert "ENV_SETUP_SCRIPT" in text
    assert "STRUCTURE_MODE" in text
    assert "RUN_PACKMOL" in text
    assert "add-extxyz-pbc" in text
    assert "extxyz-to-lammps-data" in text
    assert "/home/" not in text
    assert "conda activate" not in text
