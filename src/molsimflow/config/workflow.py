"""Lightweight INI configuration helpers for workflow templates.

The package keeps scheduler and environment settings outside Python logic.  The
helpers here provide a small standard-library bridge from public example config
files to shell variables consumed by scheduler templates.
"""

from __future__ import annotations

import argparse
import configparser
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class WorkflowConfig:
    """Parsed workflow configuration with a stable base directory."""

    path: Path
    sections: Mapping[str, Mapping[str, str]]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def section(self, name: str) -> Mapping[str, str]:
        if name not in self.sections:
            raise KeyError(f"Config section not found: {name}")
        return self.sections[name]

    def get(self, section: str, key: str, default: Optional[str] = None) -> Optional[str]:
        values = self.sections.get(section)
        if values is None:
            return default
        return values.get(key, default)

    def resolve_path(self, section: str, key: str, must_exist: bool = False) -> Path:
        raw_value = self.get(section, key)
        if raw_value is None or str(raw_value).strip() == "":
            raise KeyError(f"Config value is empty: {section}.{key}")
        path = Path(str(raw_value)).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        path = path.resolve()
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path


def load_workflow_config(path: Path) -> WorkflowConfig:
    """Load an INI workflow config while preserving option case."""

    parser = configparser.ConfigParser()
    parser.optionxform = str
    read_paths = parser.read(path)
    if not read_paths:
        raise FileNotFoundError(path)
    sections: Dict[str, Dict[str, str]] = {}
    for section in parser.sections():
        sections[section] = {key: value for key, value in parser.items(section)}
    return WorkflowConfig(path=Path(path).resolve(), sections=sections)


def iter_export_items(config: WorkflowConfig, section_names: Optional[Sequence[str]] = None) -> List[Tuple[str, str, str]]:
    """Return exportable `(section, name, value)` items from selected sections."""

    names = list(section_names) if section_names else list(config.sections.keys())
    items: List[Tuple[str, str, str]] = []
    for section in names:
        values = config.section(section)
        for key, value in values.items():
            if ENV_NAME_RE.match(key):
                items.append((section, key, str(value)))
    return items


def shell_export_lines(config: WorkflowConfig, section_names: Optional[Sequence[str]] = None) -> List[str]:
    """Format selected config values as POSIX shell export lines."""

    lines: List[str] = []
    for _section, key, value in iter_export_items(config, section_names=section_names):
        lines.append(f"export {key}={shlex.quote(value)}")
    return lines


def summarize_config(config: WorkflowConfig) -> List[Dict[str, object]]:
    """Build a compact table describing sections and keys."""

    rows: List[Dict[str, object]] = []
    for section, values in config.sections.items():
        rows.append(
            {
                "section": section,
                "n_keys": len(values),
                "keys": ",".join(values.keys()),
            }
        )
    return rows


def _print_summary(rows: Sequence[Mapping[str, object]]) -> None:
    print("section,n_keys,keys")
    for row in rows:
        print(f"{row['section']},{row['n_keys']},{row['keys']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workflow configuration helpers")
    subparsers = parser.add_subparsers(dest="config_command", required=True)

    summary = subparsers.add_parser("summary", help="Print config sections and keys")
    summary.add_argument("--config", type=Path, required=True)

    env = subparsers.add_parser("env", help="Print shell export lines for selected config sections")
    env.add_argument("--config", type=Path, required=True)
    env.add_argument("--section", action="append", help="Section to export; may be repeated")

    resolve = subparsers.add_parser("resolve-path", help="Resolve one path value relative to the config file")
    resolve.add_argument("--config", type=Path, required=True)
    resolve.add_argument("--section", required=True)
    resolve.add_argument("--key", required=True)
    resolve.add_argument("--must-exist", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_workflow_config(args.config)
        if args.config_command == "summary":
            _print_summary(summarize_config(config))
        elif args.config_command == "env":
            for line in shell_export_lines(config, section_names=args.section):
                print(line)
        elif args.config_command == "resolve-path":
            print(config.resolve_path(args.section, args.key, must_exist=args.must_exist))
        else:  # pragma: no cover
            raise ValueError(f"Unknown config command: {args.config_command}")
    except Exception as exc:
        print(f"Config command failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
