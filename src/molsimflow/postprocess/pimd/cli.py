"""PIMD CLI boundary; legacy command entry points and parser order are retained."""

from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_postprocess_pimd_reweight(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.pimd_reweight import run

    return run(["--contract", str(args.contract), "--output", str(args.output)])


def _cmd_postprocess_quantum_path(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.quantum_path_analysis import run

    return run(["--contract", str(args.contract), "--output", str(args.output)])


def _cmd_postprocess_pimd_reweight_compare(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.pimd_reweight_compare import run

    return run(["--contract", str(args.contract), "--output", str(args.output)])


def _cmd_postprocess_pimd_bead_convergence(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.pimd_bead_convergence import main as convergence_main

    workflow_args = [
        "--manifest",
        str(args.manifest),
        "--output",
        str(args.output),
        "--burn-in-ps",
        str(args.burn_in_ps),
        "--blocks",
        str(args.blocks),
        "--sigma",
        str(args.sigma),
    ]
    for field in args.fields or ["f_pi[7]"]:
        workflow_args.extend(("--field", field))
    if args.write_plot:
        workflow_args.append("--write-plot")
    return convergence_main(workflow_args)


def _add_pimd_reweight_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--contract", type=Path, required=True, help="Path-explicit PIMD analysis contract"
    )
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")


def _add_pimd_bead_convergence_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with label,beads,log")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        help="LAMMPS thermo estimator; repeat to compare several estimators",
    )
    parser.add_argument("--burn-in-ps", type=float, default=0.0)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--write-plot", action="store_true")


def register_pimd_subcommands(postprocess_subparsers) -> None:
    pimd_reweight = postprocess_subparsers.add_parser(
        "pimd-reweight",
        help="Reweight centroid-, bead-mean-, or shared-bead-density-biased PIMD",
    )
    _add_pimd_reweight_postprocess_args(pimd_reweight)
    pimd_reweight.set_defaults(func=_cmd_postprocess_pimd_reweight)

    quantum_path = postprocess_subparsers.add_parser(
        "quantum-path",
        help="Extract quantum-path descriptors and whole-frame conditional diagnostics",
    )
    _add_pimd_reweight_postprocess_args(quantum_path)
    quantum_path.set_defaults(func=_cmd_postprocess_quantum_path)

    pimd_reweight_compare = postprocess_subparsers.add_parser(
        "pimd-reweight-compare",
        help="Compare two completed PIMD reweighting analyses",
    )
    _add_pimd_reweight_postprocess_args(pimd_reweight_compare)
    pimd_reweight_compare.set_defaults(func=_cmd_postprocess_pimd_reweight_compare)

    pimd_bead_convergence = postprocess_subparsers.add_parser(
        "pimd-bead-convergence",
        help="Compare a LAMMPS PIMD thermo estimator across bead counts",
    )
    _add_pimd_bead_convergence_args(pimd_bead_convergence)
    pimd_bead_convergence.set_defaults(func=_cmd_postprocess_pimd_bead_convergence)
