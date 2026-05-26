#!/usr/bin/env python3
"""Audit the repository for files that should not enter a public GitHub repo."""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
from pathlib import Path
from typing import List, Optional

PRIVATE_TEXT_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        "/home/" + "pengchao",
        "/" + "WORK" + "/",
        "/opt/" + "gengzi",
        "bubble_ion/" + "TiO",
        r"conda activate\s+/(?:home|opt|WORK)",
    ]
]

GENERATED_NAME_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "output",
    "outputs",
    "output_*",
    "result",
    "results",
    "logs",
    "plots",
    "backup*",
    "backups*",
    "slurm-*.out",
]

GENERATED_SUFFIXES = {
    ".bak",
    ".csv",
    ".dat",
    ".dcd",
    ".err",
    ".lammpstrj",
    ".log",
    ".npy",
    ".npz",
    ".out",
    ".pdf",
    ".png",
    ".trr",
    ".tsv",
    ".xtc",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".slurm",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def iter_files(root: Path, include_ignored: bool) -> list[Path]:
    """Return repository files to audit."""
    excluded_dirs = {".git"}
    if not include_ignored:
        excluded_dirs.add("legacy_sources")

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in excluded_dirs
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            rel_path = path.relative_to(root)
            files.append(rel_path)
    return sorted(files)


def path_has_pattern(rel_path: Path, patterns: list[str]) -> bool:
    """Return whether any path part matches one of the shell-style patterns."""
    return any(fnmatch.fnmatch(part, pattern) for part in rel_path.parts for pattern in patterns)


def is_generated_path(rel_path: Path) -> bool:
    """Return whether a path looks like generated output or a backup."""
    if path_has_pattern(rel_path, GENERATED_NAME_PATTERNS):
        return True
    name = rel_path.name
    if ".bak" in name or name.endswith(".orig") or name.endswith("~"):
        return True
    return rel_path.suffix.lower() in GENERATED_SUFFIXES


def scan_text_file(path: Path) -> list[str]:
    """Scan one text file for private patterns."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [f"could_not_read: {exc}"]
    matches = []
    for pattern in PRIVATE_TEXT_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def audit(root: Path, include_ignored: bool = False) -> int:
    """Run the audit and return the number of issues."""
    files = iter_files(root, include_ignored=include_ignored)
    issues = 0

    generated = [path for path in files if is_generated_path(path)]
    if generated:
        print("Generated/output-like files:")
        for path in generated:
            print(f"  {path}")
        issues += len(generated)

    private_hits: list[tuple[Path, list[str]]] = []
    for rel_path in files:
        hits = scan_text_file(root / rel_path)
        if hits:
            private_hits.append((rel_path, hits))

    if private_hits:
        print("Private path/environment text hits:")
        for rel_path, hits in private_hits:
            print(f"  {rel_path}: {', '.join(hits)}")
        issues += len(private_hits)

    if issues == 0:
        print("audit_ok")
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to audit",
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Also audit ignored migration snapshots such as legacy_sources",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return 1 if audit(args.root.resolve(), include_ignored=args.include_ignored) else 0


if __name__ == "__main__":
    raise SystemExit(main())
