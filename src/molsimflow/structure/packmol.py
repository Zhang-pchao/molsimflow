"""PACKMOL input generation helpers."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union


@dataclass(frozen=True)
class PackmolStructure:
    """One `structure ... end structure` block in a PACKMOL input."""

    path: Path
    number: int
    comment: str = ""
    constraints: List[str] = field(default_factory=list)
    fixed: Optional[str] = None

    def render(self) -> str:
        """Render a PACKMOL structure block."""
        if self.number < 0:
            raise ValueError("PACKMOL structure number cannot be negative")
        lines: List[str] = []
        if self.comment:
            lines.append(f"# {self.comment}")
        lines.append(f"structure {self.path}")
        lines.append(f"  number {self.number}")
        if self.fixed:
            lines.append(f"  fixed {self.fixed}")
        lines.extend(f"  {constraint}" for constraint in self.constraints)
        lines.append("end structure")
        return "\n".join(lines)


@dataclass(frozen=True)
class PackmolInput:
    """Full PACKMOL input document."""

    output_xyz: str
    structures: List[PackmolStructure]
    tolerance: float = 2.4
    filetype: str = "xyz"
    seed: int = -1

    def render(self) -> str:
        """Render the full PACKMOL input."""
        lines = [
            f"tolerance {self.tolerance:g}",
            f"filetype {self.filetype}",
            f"output {self.output_xyz}",
            f"seed {self.seed}",
            "",
        ]
        lines.extend(block.render() + "\n" for block in self.structures)
        return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class PackmolRunResult:
    """Result from an optional Packmol execution."""

    command: List[str]
    input_path: Path
    cwd: Path
    output_xyz: Path
    log_path: Path
    returncode: int


def write_packmol_input(packmol_input: PackmolInput, output_path: Path) -> Path:
    """Write a PACKMOL input file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packmol_input.render(), encoding="utf-8")
    return output_path


def infer_packmol_output_xyz(input_path: Path) -> str:
    """Infer the PACKMOL output XYZ filename from an input file."""
    for raw_line in input_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() == "output":
            return parts[1].strip()
    raise ValueError(f"Could not infer PACKMOL output file from {input_path}")


def run_packmol(
    input_path: Path,
    *,
    command: Union[str, Sequence[str]] = "packmol",
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
    check: bool = True,
) -> PackmolRunResult:
    """Run PACKMOL with an input file passed through standard input.

    Execution is deliberately opt-in.  Library callers and CLI users can still
    generate files without invoking an external executable.
    """
    input_file = Path(input_path).resolve()
    run_cwd = Path(cwd).resolve() if cwd is not None else input_file.parent
    run_cwd.mkdir(parents=True, exist_ok=True)

    output_name = infer_packmol_output_xyz(input_file)
    output_path = Path(output_name)
    if not output_path.is_absolute():
        output_path = run_cwd / output_path

    run_log = Path(log_path) if log_path is not None else run_cwd / "packmol.out"
    if not run_log.is_absolute():
        run_log = run_cwd / run_log
    run_log.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(command, str):
        command_parts = shlex.split(command)
    else:
        command_parts = [str(part) for part in command]
    if not command_parts:
        raise ValueError("PACKMOL command cannot be empty")

    proc = subprocess.run(
        command_parts,
        input=input_file.read_text(encoding="utf-8"),
        cwd=run_cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = proc.stdout
    if proc.stderr:
        log_text += "\n[stderr]\n" + proc.stderr
    run_log.write_text(log_text, encoding="utf-8")

    if check and proc.returncode != 0:
        raise RuntimeError(f"PACKMOL failed with exit code {proc.returncode}. See log: {run_log}")
    if check and not output_path.exists():
        raise FileNotFoundError(f"PACKMOL completed but output file was not found: {output_path}")

    return PackmolRunResult(
        command=command_parts,
        input_path=input_file,
        cwd=run_cwd,
        output_xyz=output_path,
        log_path=run_log,
        returncode=proc.returncode,
    )


def resolve_template_path(template_path: Optional[Path], molecule_dir: Optional[Path], filename: str) -> Path:
    """Resolve an explicit template path or a filename under a molecule directory."""
    if template_path is not None:
        return template_path
    if molecule_dir is None:
        raise ValueError(f"Missing template path and molecule directory for {filename}")
    return molecule_dir / filename


def validate_template_paths(paths: Iterable[Path]) -> None:
    """Validate that all molecule template files exist."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing molecule template file(s): " + ", ".join(missing))
