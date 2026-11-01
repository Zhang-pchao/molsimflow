#!/usr/bin/env python3
"""Generate an import-free inventory of reusable package modules and entrypoints."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def render(root: Path) -> str:
    """Read source syntax without importing optional numerical dependencies."""
    package = root / "src" / "molsimflow"
    groups = {}
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if path.name == "__init__.py" or len(relative.parts) < 2:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        group = "postprocess/pimd" if relative.parts[:2] == ("postprocess", "pimd") else relative.parts[0]
        description = (ast.get_docstring(tree) or "See the owning source and tests.").splitlines()[0]
        public = [node.name for node in tree.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and not node.name.startswith("_")]
        if not public:
            public = [alias.asname or alias.name for node in tree.body
                      if isinstance(node, ast.ImportFrom) and node.module
                      and (node.module.startswith("molsimflow") or node.level)
                      for alias in node.names if alias.name != "*" and not alias.name.startswith("_")]
        names = ", ".join("`" + name + "`" for name in dict.fromkeys(public)) or "See source/facade exports"
        module = "molsimflow." + str(relative.with_suffix("")).replace("/", ".")
        link = "../" + path.relative_to(root).as_posix()
        groups.setdefault(group, []).append((module, link, description.replace("|", "\\|"), names))
    lines = ["# Module catalog", "", "Generated with `python scripts/generate_module_catalog.py`; do not edit individual rows.",
             "This syntax inventory does not import optional dependencies or certify scientific applicability.",
             "Read the [architecture guide](postprocess_architecture.md), owning docs and tests before reuse.",
             "The names column lists top-level public definitions, or explicit facade re-exports.", ""]
    for group, rows in sorted(groups.items()):
        lines.extend(["## " + group, "", "| Module | Scope | Public definitions / exports |", "| --- | --- | --- |"])
        for module, link, description, names in rows:
            lines.append(f"| [{module}]({link}) | {description} | {names} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the tracked catalog needs regeneration")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "docs" / "module_catalog.md"
    expected = render(root)
    if args.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != expected:
            parser.exit(1, "Module catalog is stale; run python scripts/generate_module_catalog.py\n")
        print("module_catalog_ok")
    else:
        destination.write_text(expected, encoding="utf-8")
        print("Updated docs/module_catalog.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
