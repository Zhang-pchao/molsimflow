"""Build an immutable high-frequency TPCL paired force-step run package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from molsimflow.postprocess.constant_force_species_timeseries import (
    read_model_arrays,
    read_type_symbols,
)

BRANCHES = (
    ("f0_shared", "none", 0.0, 0.0, "f0"),
    ("f8e-5_x", "X", 8.0e-5, 0.0, "fx"),
    ("f8e-5_y", "Y", 0.0, 8.0e-5, "fy"),
)
CASE_JOB_LABELS = {"ch3_only": "ch3", "mixed291": "m291"}
TEMPLATE_ROOT = Path(__file__).with_name("templates")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source(value: object) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty source file: {path}")
    return path


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    model_data: Path
    group_file: Path
    parent_restart: Path
    parent_job_id: str
    parent_restart_sha256: str
    natoms: int
    substrate_atoms: int
    anchor_atoms: int
    water_oxygen: int
    start_step: int = 36_200_000
    top_surface_depth_A: float = 5.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "CaseSpec":
        spec = cls(
            case_id=str(raw["case_id"]),
            model_data=_require_source(raw["model_data"]),
            group_file=_require_source(raw["group_file"]),
            parent_restart=_require_source(raw["parent_restart"]),
            parent_job_id=str(raw["parent_job_id"]),
            parent_restart_sha256=str(raw["parent_restart_sha256"]),
            natoms=int(raw["natoms"]),
            substrate_atoms=int(raw["substrate_atoms"]),
            anchor_atoms=int(raw["anchor_atoms"]),
            water_oxygen=int(raw["water_oxygen"]),
            start_step=int(raw.get("start_step", 36_200_000)),
            top_surface_depth_A=float(raw.get("top_surface_depth_A", 5.0)),
        )
        if spec.case_id not in CASE_JOB_LABELS:
            raise ValueError(f"unsupported case: {spec.case_id}")
        if spec.start_step != 36_200_000:
            raise ValueError("D1 parents must start at step 36200000")
        if spec.natoms <= spec.substrate_atoms or min(
            spec.anchor_atoms,
            spec.water_oxygen,
        ) <= 0:
            raise ValueError(f"invalid atom counts for {spec.case_id}")
        if spec.top_surface_depth_A <= 0:
            raise ValueError("top-surface depth must be positive")
        if sha256(spec.parent_restart) != spec.parent_restart_sha256:
            raise ValueError(f"parent restart hash mismatch for {spec.case_id}")
        return spec


@dataclass(frozen=True)
class RuntimeSpec:
    model: Path
    lammps_module: str
    python: str
    partition: str
    qos: str
    nodes: int
    tasks: int
    gpus_per_node: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "RuntimeSpec":
        spec = cls(
            model=_require_source(raw["model"]),
            lammps_module=str(raw["lammps_module"]),
            python=str(raw["python"]),
            partition=str(raw["partition"]),
            qos=str(raw["qos"]),
            nodes=int(raw["nodes"]),
            tasks=int(raw["tasks"]),
            gpus_per_node=int(raw["gpus_per_node"]),
        )
        for value in (
            spec.lammps_module,
            spec.python,
            spec.partition,
            spec.qos,
        ):
            if not value or "\n" in value:
                raise ValueError("runtime strings must be non-empty single lines")
        if (spec.nodes, spec.tasks, spec.gpus_per_node) != (1, 1, 1):
            raise ValueError("D1 requires one node, one task, and one GPU")
        return spec


def _top_surface_ids(spec: CaseSpec) -> tuple[int, ...]:
    ids, types, coordinates, _ = read_model_arrays(spec.model_data)
    if len(ids) != spec.natoms:
        raise ValueError(f"{spec.case_id}: model atom count mismatch")
    type_symbols = read_type_symbols(spec.model_data)
    substrate = ids <= spec.substrate_atoms
    substrate_z = coordinates[substrate, 2]
    threshold = float(substrate_z.max() - spec.top_surface_depth_A)
    selected = tuple(
        int(atom_id)
        for atom_id, atom_type, z in zip(ids, types, coordinates[:, 2])
        if int(atom_id) <= spec.substrate_atoms
        and float(z) >= threshold
        and type_symbols[int(atom_type)] in {"H", "O", "C", "Si"}
    )
    if not selected:
        raise ValueError(f"{spec.case_id}: empty top-surface selection")
    return selected


def _append_static_analysis_groups(group_file: Path, ids: Sequence[int]) -> str:
    text = group_file.read_text(encoding="utf-8").rstrip() + "\n"
    for offset in range(0, len(ids), 250):
        values = " ".join(str(atom_id) for atom_id in ids[offset : offset + 250])
        text += f"group           topSurface id {values}\n"
    text += "group           tpclAnalysis union water topSurface\n"
    return text


def _case_environment(
    spec: CaseSpec,
    branch_name: str,
    direction: str,
    force_x: float,
    force_y: float,
    selected_count: int,
) -> str:
    parent_root = Path("02_parents") / spec.case_id
    rows = {
        "CASE_NAME": spec.case_id,
        "BRANCH_NAME": branch_name,
        "DRIVE_DIRECTION": direction,
        "FORCE_X": f"{force_x:.8g}",
        "FORCE_Y": f"{force_y:.8g}",
        "NATOMS": str(spec.natoms),
        "NSUB": str(spec.substrate_atoms),
        "NWATER": str(spec.natoms - spec.substrate_atoms),
        "NANCHOR": str(spec.anchor_atoms),
        "NDRIVE_O": str(spec.water_oxygen),
        "NSELECTED": str(selected_count),
        "START_STEP": str(spec.start_step),
        "TIMESTEP_FS": "0.5",
        "MAX_RAW_FORCE": "50.0",
        "MAX_O_SPEED": "50.0",
        "MAX_WATER_TEMP": "2000.0",
        "PARENT_JOB_ID": spec.parent_job_id,
        "PARENT_RESTART_SHA256": spec.parent_restart_sha256,
        "PARENT_RESTART": str(parent_root / "f0_final.restart"),
        "MODEL_DATA": str(parent_root / "model.data"),
        "GROUP_FILE": str(parent_root / "groups.tpcl.lmp"),
    }
    return "".join(f"{key}={value}\n" for key, value in rows.items())


def _slurm_script(
    runtime: RuntimeSpec,
    job_name: str,
    case_id: str,
    branch_name: str,
    nsteps: int,
    mode: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={runtime.partition}
#SBATCH --nodes={runtime.nodes}
#SBATCH --ntasks={runtime.tasks}
#SBATCH --gpus-per-node={runtime.gpus_per_node}
#SBATCH --qos={runtime.qos}

source /etc/profile
set -eo pipefail
package_root=$(readlink -f "${{SLURM_SUBMIT_DIR:-$PWD}}")
[[ -f "$package_root/00_contract/RUNTIME-SHA256SUMS" ]]
branch="$package_root/03_cases/{case_id}/{branch_name}"
output="$branch/{'smoke' if mode == 'smoke' else 'run_100ps'}/$SLURM_JOB_ID"
exec "$package_root/01_common/run_tpcl_force_step.sh" \\
  "$branch" "$output" {nsteps} {mode}
"""


def _copy_code_snapshot(root: Path) -> None:
    package_source = Path(__file__).parents[1]
    target = root / "05_postprocess/code_snapshot/molsimflow"
    selected = (
        Path("__init__.py"),
        Path("io/__init__.py"),
        Path("io/lammps_dump.py"),
        Path("postprocess/__init__.py"),
        Path("postprocess/constant_force_oxygen.py"),
        Path("postprocess/constant_force_species_timeseries.py"),
        Path("postprocess/tpcl_force_step_io.py"),
    )
    for relative in selected:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_source / relative, destination)


def _write_runtime_manifest(root: Path) -> None:
    excluded = {
        Path("00_contract/RUNTIME-SHA256SUMS"),
        Path("04_jobs/SUBMISSION.tsv"),
    }
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root) not in excluded
    ]
    manifest = root / "00_contract/RUNTIME-SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(root)}\n"
            for path in sorted(files, key=lambda item: str(item.relative_to(root)))
        ),
        encoding="utf-8",
    )


def _validate_shell_scripts(root: Path) -> None:
    for script in sorted(root.rglob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)
    for script in sorted((root / "04_jobs").glob("q_*.sh")):
        text = script.read_text(encoding="utf-8")
        directives = {
            line.split("=", 1)[0].strip()
            for line in text.splitlines()
            if line.startswith("#SBATCH --")
        }
        allowed = {
            "#SBATCH --job-name",
            "#SBATCH --partition",
            "#SBATCH --nodes",
            "#SBATCH --ntasks",
            "#SBATCH --gpus-per-node",
            "#SBATCH --qos",
        }
        if directives != allowed:
            raise ValueError(f"{script}: unexpected Slurm directives {directives ^ allowed}")
        if "sbatch " in text or "--test-only" in text or "--export" in text:
            raise ValueError(f"{script}: forbidden submission option")


def build_package(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Build a fresh immutable package from explicit local source files."""

    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    runtime = RuntimeSpec.from_mapping(config["runtime"])
    cases = [CaseSpec.from_mapping(raw) for raw in config["cases"]]
    if {case.case_id for case in cases} != set(CASE_JOB_LABELS):
        raise ValueError("D1 requires exactly ch3_only and mixed291")
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"refusing existing package directory: {root}")
    for name in (
        "00_contract",
        "01_common/model",
        "02_parents",
        "03_cases",
        "04_jobs",
        "05_postprocess",
        "06_review",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)

    shutil.copy2(runtime.model, root / "01_common/model/mini500k-compressed.pt2")
    shutil.copy2(
        TEMPLATE_ROOT / "tpcl_force_step.in.lmp",
        root / "01_common/in.tpcl_force_step.lmp",
    )
    runner = (TEMPLATE_ROOT / "run_tpcl_force_step.sh").read_text(encoding="utf-8")
    runner = runner.replace("@@LAMMPS_MODULE@@", runtime.lammps_module)
    runner = runner.replace("@@PYTHON_EXEC@@", runtime.python)
    runner_path = root / "01_common/run_tpcl_force_step.sh"
    runner_path.write_text(runner, encoding="utf-8")
    runner_path.chmod(runner_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    _copy_code_snapshot(root)

    parent_rows = []
    branch_rows = []
    for spec in cases:
        parent_root = root / "02_parents" / spec.case_id
        parent_root.mkdir(parents=True)
        shutil.copy2(spec.parent_restart, parent_root / "f0_final.restart")
        shutil.copy2(spec.model_data, parent_root / "model.data")
        top_ids = _top_surface_ids(spec)
        group_text = _append_static_analysis_groups(spec.group_file, top_ids)
        (parent_root / "groups.tpcl.lmp").write_text(group_text, encoding="utf-8")
        (parent_root / "top_surface.ids").write_text(
            "\n".join(str(atom_id) for atom_id in top_ids) + "\n",
            encoding="utf-8",
        )
        selected_count = spec.natoms - spec.substrate_atoms + len(top_ids)
        parent_rows.append(
            (
                spec.case_id,
                spec.parent_job_id,
                spec.start_step,
                spec.parent_restart_sha256,
                len(top_ids),
                selected_count,
            )
        )
        for branch, direction, force_x, force_y, short in BRANCHES:
            branch_root = root / "03_cases" / spec.case_id / branch
            branch_root.mkdir(parents=True)
            (branch_root / "CASE.env").write_text(
                _case_environment(
                    spec,
                    branch,
                    direction,
                    force_x,
                    force_y,
                    selected_count,
                ),
                encoding="utf-8",
            )
            label = CASE_JOB_LABELS[spec.case_id]
            job_name = f"ndhf-{label}-{short}"
            job_path = root / "04_jobs" / f"q_{label}_{short}.sh"
            job_path.write_text(
                _slurm_script(runtime, job_name, spec.case_id, branch, 200_000, "production"),
                encoding="utf-8",
            )
            branch_rows.append(
                (spec.case_id, branch, direction, force_x, force_y, job_name, job_path.name)
            )
    smoke_path = root / "04_jobs/q_smoke_m291_fx.sh"
    smoke_path.write_text(
        _slurm_script(
            runtime,
            "ndhf-smoke-m291-fx",
            "mixed291",
            "f8e-5_x",
            4_000,
            "smoke",
        ),
        encoding="utf-8",
    )

    for script in root.rglob("*.sh"):
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    (root / "00_contract/PARENT-MANIFEST.tsv").write_text(
        "case_id\tparent_job_id\tstart_step\tparent_restart_sha256\t"
        "top_surface_atoms\tselected_atoms\n"
        + "".join("\t".join(map(str, row)) + "\n" for row in parent_rows),
        encoding="utf-8",
    )
    (root / "00_contract/BRANCH-MANIFEST.tsv").write_text(
        "case_id\tbranch_id\tdirection\tforce_x_eV_A\tforce_y_eV_A\tjob_name\tjob_script\n"
        + "".join("\t".join(map(str, row)) + "\n" for row in branch_rows),
        encoding="utf-8",
    )
    (root / "00_contract/EXPERIMENT-CONTRACT.md").write_text(
        "# Stage D1 high-frequency TPCL force-step contract\n\n"
        "- Same accepted F0 position and velocity restart for F0, +Fx, and +Fy.\n"
        "- No velocity reseeding.\n"
        "- 0.5 fs timestep and 330 K thermostat contract inherited from the parent.\n"
        "- 2 ps mixed291/+Fx smoke precedes every 100 ps production submission.\n"
        "- Coordinate cadence: 10 fs for 0-20 ps, then 50 fs for 20-100 ps.\n"
        "- Selected dynamics cadence: 50 fs; full reference cadence: 0.5 ps.\n"
        "- Global motion/force cadence: 10 fs; unique restart cadence: 5 ps.\n"
        "- Per-job projected output ceiling: 3 GiB, calibrated by the smoke.\n"
        "- No automatic retry, requeue, extension, or force-release submission.\n",
        encoding="utf-8",
    )
    (root / "04_jobs/SUBMISSION.tsv").write_text(
        "kind\tcase_id\tbranch_id\tjob_id\tscript_sha256\tsubmitted_at\tstatus\n",
        encoding="utf-8",
    )
    build_report = {
        "schema_version": 1,
        "source_config": str(config_file),
        "source_config_sha256": sha256(config_file),
        "cases": [case.case_id for case in cases],
        "branches": [branch for branch, *_ in BRANCHES],
        "smoke": "mixed291/f8e-5_x",
        "status": "BUILT",
    }
    (root / "00_contract/BUILD.json").write_text(
        json.dumps(build_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _validate_shell_scripts(root)
    _write_runtime_manifest(root)
    subprocess.run(
        ["sha256sum", "-c", "00_contract/RUNTIME-SHA256SUMS"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(build_package(args.config, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
