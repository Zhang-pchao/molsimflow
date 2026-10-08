from pathlib import Path

from molsimflow.postprocess.tpcl_state_diagnostics import read_lammps_thermo, run_summary


def _write_sources(tmp_path: Path) -> Path:
    frame = tmp_path / "frames.csv"
    frame.write_text(
        "time_ns,largest_cluster_size,contact_line_area_A2\n"
        "10.0,10,100\n10.5,11,101\n11.0,12,102\n",
        encoding="utf-8",
    )
    thermo = tmp_path / "lmp.out"
    thermo.write_text(
        "noise\nStep Time Temp Density PotEng KinEng TotEng Volume Press Pxx Pyy Pzz Pxy Pxz Pyz\n"
        "100 10000 300 1 -10 1 -9 1000 2 1 2 3 0 0 0\n"
        "200 10500 301 1 -11 1 -10 1000 3 2 3 4 0 0 0\n"
        "300 11000 302 1 -12 1 -11 1000 4 3 4 5 0 0 0\n",
        encoding="utf-8",
    )
    stress = tmp_path / "stress.dat"
    stress.write_text(
        "# Time-averaged data\n"
        "# TimeStep v_pxx_check v_pyy_check v_pzz_check v_pxy_check v_pxz_check v_pyz_check\n"
        "100 1 2 3 0 0 0\n200 2 3 4 0 0 0\n300 3 4 5 0 0 0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "sources.tsv"
    manifest.write_text(
        f"case_id\tframe_metrics\tthermo_log\tglobal_stress\ncase\t{frame}\t{thermo}\t{stress}\n",
        encoding="utf-8",
    )
    return manifest


def test_read_thermo_and_run_summary(tmp_path):
    manifest = _write_sources(tmp_path)
    thermo = read_lammps_thermo(tmp_path / "lmp.out")
    assert thermo["Temp"][0].tolist() == [10.0, 10.5, 11.0]
    summary = run_summary(
        manifest,
        tmp_path / "out",
        timestep_fs=0.5,
        block_ps=500.0,
        font_path=tmp_path / "unused.ttf",
        frame_fields=("largest_cluster_size", "contact_line_area_A2"),
        make_plots=False,
    )
    assert summary["status"] == "PASS"
    assert (tmp_path / "out" / "important_data.csv").is_file()
