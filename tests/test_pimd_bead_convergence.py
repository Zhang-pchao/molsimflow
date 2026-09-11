import csv

import pytest

from molsimflow.postprocess.pimd_bead_convergence import analyze


def _write_log(path, offset):
    path.write_text(
        "Step Time Temp f_pi[7]\n"
        + "\n".join(f"{step} {step / 10} 300 {offset + step % 3}" for step in range(10))
        + "\nLoop time of 1 on 1 procs for 10 steps with 1 atoms\n",
        encoding="utf-8",
    )


def test_analyze_bead_convergence(tmp_path):
    _write_log(tmp_path / "p16.log", 10)
    _write_log(tmp_path / "p32.log", 10.2)
    _write_log(tmp_path / "p36.log", 10.1)
    manifest = tmp_path / "cases.csv"
    manifest.write_text(
        "label,beads,log\nP16,16,p16.log\nP32,32,p32.log\nP36,36,p36.log\n",
        encoding="utf-8",
    )

    output = tmp_path / "result"
    analyze(manifest, output, burn_in_ps=0.2, blocks=4)

    with (output / "reference_comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["beads"]) for row in rows] == [16, 32]
    assert all(int(row["reference_beads"]) == 36 for row in rows)
    assert float(rows[1]["difference_eV"]) == pytest.approx(0.1)


def test_refuses_existing_output(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    with pytest.raises(FileExistsError):
        analyze(tmp_path / "missing.csv", output)
