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


def _write_multi_estimator_log(path, offset):
    path.write_text(
        "Step Time Temp f_pi[5] f_pi[6] f_pi[7] f_pi[10]\n"
        + "\n".join(
            f"{step} {step / 10} 300 "
            f"{offset + step % 3} {offset + 2 + step % 3} "
            f"{offset + 4 + step % 3} {100 * offset + 10 * (step % 3)}"
            for step in range(10)
        )
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
    analyze(manifest, output, burn_in_ps=0.2, blocks=4, write_plot=True)

    with (output / "reference_comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["beads"]) for row in rows] == [16, 32]
    assert all(int(row["reference_beads"]) == 36 for row in rows)
    assert float(rows[1]["difference_eV"]) == pytest.approx(0.1)
    assert (output / "bead_convergence.png").stat().st_size > 0


def test_refuses_existing_output(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    with pytest.raises(FileExistsError):
        analyze(tmp_path / "missing.csv", output)


def test_analyze_multiple_estimators_with_physical_names_and_units(tmp_path):
    _write_multi_estimator_log(tmp_path / "p16.log", 10)
    _write_multi_estimator_log(tmp_path / "p32.log", 10.2)
    _write_multi_estimator_log(tmp_path / "p36.log", 10.1)
    manifest = tmp_path / "cases.csv"
    manifest.write_text(
        "label,beads,log\nP16,16,p16.log\nP32,32,p32.log\nP36,36,p36.log\n",
        encoding="utf-8",
    )

    output = tmp_path / "result"
    fields = ["f_pi[5]", "f_pi[6]", "f_pi[7]", "f_pi[10]"]
    analyze(manifest, output, field=fields, burn_in_ps=0.2, blocks=4)

    with (output / "estimator_summary.csv").open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 12
    assert {row["physical_quantity"] for row in summaries} == {
        "Primitive kinetic-energy estimator",
        "Virial energy estimator",
        "Centroid-virial energy estimator",
        "Centroid-virial pressure estimator",
    }
    assert {row["unit"] for row in summaries if row["field"] == "f_pi[10]"} == {"bar"}
    assert {row["unit"] for row in summaries if row["field"] != "f_pi[10]"} == {"eV"}

    with (output / "estimator_reference_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 8
    assert {int(row["reference_beads"]) for row in comparisons} == {36}
    assert not (output / "bead_summary.csv").exists()


def test_multiple_estimator_plot(tmp_path):
    pytest.importorskip("matplotlib")
    _write_multi_estimator_log(tmp_path / "p16.log", 10)
    _write_multi_estimator_log(tmp_path / "p32.log", 10.2)
    manifest = tmp_path / "cases.csv"
    manifest.write_text(
        "label,beads,log\nP16,16,p16.log\nP32,32,p32.log\n",
        encoding="utf-8",
    )

    output = tmp_path / "result"
    analyze(
        manifest,
        output,
        field=["f_pi[5]", "f_pi[6]", "f_pi[7]", "f_pi[10]"],
        write_plot=True,
    )
    assert (output / "estimator_convergence.png").stat().st_size > 0
