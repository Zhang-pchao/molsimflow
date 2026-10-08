import hashlib
import json
from pathlib import Path

from molsimflow.workflows.tpcl_force_step import build_package


def _write_model_data(path: Path) -> None:
    path.write_text(
        """10 atoms
4 atom types

0 10 xlo xhi
0 10 ylo yhi
0 20 zlo zhi

Masses

1 1.008 # H
2 15.999 # O
3 12.011 # C
4 28.085 # Si

Atoms # atomic

1 4 1 1 1
2 2 2 2 2
3 3 3 3 9
4 1 4 4 10
5 2 5 5 12
6 1 5.8 5 12
7 1 4.2 5 12
8 2 7 7 12
9 1 7.8 7 12
10 1 6.2 7 12
""",
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_package_freezes_paired_parent_and_static_selection(tmp_path: Path):
    model = tmp_path / "model.pt2"
    model.write_bytes(b"model")
    cases = []
    for case_id in ("ch3_only", "mixed291"):
        source = tmp_path / case_id
        source.mkdir()
        data = source / "model.data"
        _write_model_data(data)
        groups = source / "groups.lmp"
        groups.write_text(
            "group substrate id 1:4\n"
            "group water id 5:10\n"
            "group waterO id 5 8\n"
            "group upperAllO id 5 8\n"
            "group anchor id 1\n"
            "group mobile_substrate subtract substrate anchor\n",
            encoding="utf-8",
        )
        restart = source / "final.restart"
        restart.write_bytes(f"{case_id}-parent".encode())
        cases.append(
            {
                "case_id": case_id,
                "model_data": str(data),
                "group_file": str(groups),
                "parent_restart": str(restart),
                "parent_job_id": "123",
                "parent_restart_sha256": _digest(restart),
                "natoms": 10,
                "substrate_atoms": 4,
                "anchor_atoms": 1,
                "water_oxygen": 2,
            }
        )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "runtime": {
                    "model": str(model),
                    "lammps_module": "lammps/test",
                    "python": "/opt/python",
                    "partition": "gpu",
                    "qos": "normal",
                    "nodes": 1,
                    "tasks": 1,
                    "gpus_per_node": 1,
                },
                "cases": cases,
                "job_name_suffix": "-r1",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "package"
    result = build_package(config, output)
    assert result["status"] == "BUILT"
    envs = [
        (output / "03_cases/mixed291" / branch / "CASE.env").read_text()
        for branch in ("f0_shared", "f8e-5_x", "f8e-5_y")
    ]
    assert all(
        "PARENT_RESTART=02_parents/mixed291/f0_final.restart" in text for text in envs
    )
    assert "FORCE_X=0" in envs[0]
    assert "FORCE_X=8e-05" in envs[1]
    assert "FORCE_Y=8e-05" in envs[2]
    groups = (output / "02_parents/mixed291/groups.tpcl.lmp").read_text()
    assert "group           topSurface id 3 4" in groups
    assert "group           tpclAnalysis union water topSurface" in groups
    smoke = (output / "04_jobs/q_smoke_m291_fx.sh").read_text()
    assert "#SBATCH --partition=gpu" in smoke
    assert "4000 smoke" in smoke
    assert "#SBATCH --job-name=ndhf-smoke-m291-fx-r1" in smoke
    assert (output / "00_contract/RUNTIME-SHA256SUMS").is_file()
    runner = (output / "01_common/run_tpcl_force_step.sh").read_text(encoding="utf-8")
    assert runner.index("printf 'status=PASS") < runner.index('find "$output_root"')
    assert "RUN-RESULT.txt" in runner
    assert "sbatch " not in smoke
    snapshot_init = output / "05_postprocess/code_snapshot/molsimflow/postprocess/__init__.py"
    assert "centroids" not in snapshot_init.read_text()
    assert not list((output / "05_postprocess/code_snapshot").rglob("__pycache__"))
