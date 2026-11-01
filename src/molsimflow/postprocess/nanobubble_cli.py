"""Lazy CLI adapters for explicit nanobubble analysis contracts."""
from __future__ import annotations
import argparse
from pathlib import Path

def _cmd_postprocess_nanobubble_flow_response(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.nanobubble_flow_response import run_contract

    summary = run_contract(args.contract, args.output)
    print(args.output.resolve())
    print(
        f"case={summary['case_id']} "
        f"direction={summary['direction']} "
        f"frames={summary['frames']} "
        f"driven_time_ns={summary['driven_time_ns']}"
    )
    return 0


def _cmd_postprocess_nanobubble_local_flow(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.nanobubble_local_flow import run_contract

    summary = run_contract(args.contract, args.output)
    print(args.output.resolve())
    print(
        f"status={summary['status']} frames={summary['frames']} "
        f"sample_interval_ps={summary['sample_interval_ps']}"
    )
    return 0


def _cmd_postprocess_nanobubble_edge_events(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.nanobubble_edge_events import run_contract

    summary = run_contract(args.contract, args.output)
    print(args.output.resolve())
    print(
        f"status={summary['status']} events={summary['events']} "
        f"valid_events={summary['valid_events']}"
    )
    return 0


def _cmd_postprocess_nanobubble_edge_water(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.nanobubble_edge_water import run_contract

    summary = run_contract(args.contract, args.output)
    print(args.output.resolve())
    print(
        f"status={summary['status']} frames={summary['frames']} "
        f"patch_edge_rows={summary['patch_edge_frame_rows']}"
    )
    return 0


def _cmd_postprocess_nanobubble_edge_mechanism(args: argparse.Namespace) -> int:
    from molsimflow.postprocess.nanobubble_edge_mechanism import run_contract

    summary = run_contract(args.contract, args.output)
    print(args.output.resolve())
    print(
        f"status={summary['status']} valid_events={summary['valid_events']} "
        f"prediction_decision={summary['prediction_decision']}"
    )
    return 0


def register_nanobubble_subcommands(subparsers) -> None:
    """Register domain adapters without importing numerical dependencies."""
    commands = (
        ("nanobubble-flow-response", "Compare driven gas and surrounding flow", _cmd_postprocess_nanobubble_flow_response),
        ("nanobubble-local-flow", "Describe supported local flow regions", _cmd_postprocess_nanobubble_local_flow),
        ("nanobubble-edge-events", "Describe sampled contact edge events", _cmd_postprocess_nanobubble_edge_events),
        ("nanobubble-edge-water", "Describe contact edge water observations", _cmd_postprocess_nanobubble_edge_water),
        ("nanobubble-edge-mechanism", "Compare declared edge-event diagnostics", _cmd_postprocess_nanobubble_edge_mechanism),
    )
    for name, description, handler in commands:
        parser = subparsers.add_parser(name, help=description)
        parser.add_argument("--contract", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.set_defaults(func=handler)
