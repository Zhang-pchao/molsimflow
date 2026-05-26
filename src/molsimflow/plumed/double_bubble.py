#!/usr/bin/env python3
"""Generate slab-style double-bubble PLUMED files from PACKMOL and LAMMPS data.

Key points:
- Designed for reusable multi-case generation.
- Infers ordering/ranges from PACKMOL block order and structure metadata.
- Uses model_atomic.data header only for total-atom consistency check.
- Keeps bridge-cylinder + cluster-wall template-style PLUMED organization.
"""

import argparse
import ast
import math
import re
from collections import Counter
from pathlib import Path


TOTAL_ATOMS_RE = re.compile(r"\s*(\d+)\s+atoms\s*$")


class StructureBlock(object):
    def __init__(
        self,
        index,
        path_raw,
        path_resolved,
        number,
        inside_sphere,
        has_fixed,
        atoms_per_mol,
        symbols,
        category,
    ):
        self.index = index
        self.path_raw = path_raw
        self.path_resolved = path_resolved
        self.number = int(number)
        self.inside_sphere = inside_sphere
        self.has_fixed = bool(has_fixed)
        self.atoms_per_mol = int(atoms_per_mol)
        self.symbols = list(symbols)
        self.category = category
        self.range_start = None
        self.range_end = None


class ParseSummary(object):
    def __init__(
        self,
        total_atoms,
        slab_ranges,
        water_ranges,
        ion_ranges,
        n2_ranges,
        water_oxygen_expr,
        bubble_a_pairs,
        bubble_b_pairs,
        bubble_a_range,
        bubble_b_range,
        gas_radius_a,
        gas_radius_b,
        bubble_spacing,
        axis,
        bridge_radius,
        bridge_half_length,
        assumptions,
        oxygen_ion_labels,
        case_label,
        block_lines,
    ):
        self.total_atoms = int(total_atoms)
        self.slab_ranges = slab_ranges
        self.water_ranges = water_ranges
        self.ion_ranges = ion_ranges
        self.n2_ranges = n2_ranges
        self.water_oxygen_expr = water_oxygen_expr
        self.bubble_a_pairs = bubble_a_pairs
        self.bubble_b_pairs = bubble_b_pairs
        self.bubble_a_range = bubble_a_range
        self.bubble_b_range = bubble_b_range
        self.gas_radius_a = float(gas_radius_a)
        self.gas_radius_b = float(gas_radius_b)
        # Keep a single representative radius for backward-compatible comments/diagnostics.
        self.gas_radius = max(self.gas_radius_a, self.gas_radius_b)
        self.bubble_spacing = float(bubble_spacing)
        self.axis = axis
        self.bridge_radius = float(bridge_radius)
        self.bridge_half_length = float(bridge_half_length)
        self.assumptions = assumptions
        self.oxygen_ion_labels = oxygen_ion_labels
        self.case_label = case_label
        self.block_lines = block_lines


def safe_eval_expr(expr, variables):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    allowed = [
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Name,
        ast.Load,
    ]
    if hasattr(ast, "Constant"):
        allowed.append(ast.Constant)
    if hasattr(ast, "Num"):
        allowed.append(ast.Num)
    if hasattr(ast, "NameConstant"):
        allowed.append(ast.NameConstant)
    allowed = tuple(allowed)

    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return None
        if isinstance(node, ast.Name) and node.id not in variables:
            return None

    try:
        value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, variables)
    except Exception:
        return None

    try:
        return float(value)
    except Exception:
        return None


def parse_build_script(path):
    out = {}
    if path is None or not path.exists():
        return out

    # Read simple top-level assignments from the build script.  This is only a
    # fallback path; when packmol.in contains "inside sphere" lines, the actual
    # PACKMOL radii and centers are preferred.
    assignment_exprs = {}
    wanted = set([
        "GAS_RADIUS",
        "GAS_RADIUS_1",
        "GAS_RADIUS_2",
        "BUBBLE_SPACING",
    ])

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        lhs, rhs = [x.strip() for x in line.split("=", 1)]
        if lhs in wanted:
            assignment_exprs[lhs] = rhs

    variables = {}
    for name in ["GAS_RADIUS", "GAS_RADIUS_1", "GAS_RADIUS_2"]:
        expr = assignment_exprs.get(name)
        if expr is None:
            continue
        value = safe_eval_expr(expr, variables)
        if value is not None:
            out[name] = value
            variables[name] = value

    # Backward/forward compatibility:
    # - old builder: GAS_RADIUS
    # - new builder: GAS_RADIUS_1 and GAS_RADIUS_2
    if "GAS_RADIUS_1" not in out and "GAS_RADIUS" in out:
        out["GAS_RADIUS_1"] = out["GAS_RADIUS"]
        variables["GAS_RADIUS_1"] = out["GAS_RADIUS_1"]
    if "GAS_RADIUS_2" not in out and "GAS_RADIUS" in out:
        out["GAS_RADIUS_2"] = out["GAS_RADIUS"]
        variables["GAS_RADIUS_2"] = out["GAS_RADIUS_2"]
    if "GAS_RADIUS" not in out and "GAS_RADIUS_1" in out:
        out["GAS_RADIUS"] = out["GAS_RADIUS_1"]
        variables["GAS_RADIUS"] = out["GAS_RADIUS"]

    spacing_expr = assignment_exprs.get("BUBBLE_SPACING")
    if spacing_expr is not None:
        s = safe_eval_expr(spacing_expr, variables)
        if s is not None:
            out["BUBBLE_SPACING"] = s

    return out

def normalize_symbol(sym):
    s = (sym or "").strip()
    if not s:
        return s
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()


def parse_xyz_metadata(path, max_symbols=50):
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not text:
        raise ValueError("Empty xyz file: {}".format(path))
    natoms = int(text[0].strip().split()[0])

    symbols = []
    start = 2
    end = min(start + natoms, len(text))
    for line in text[start:end]:
        parts = line.split()
        if not parts:
            continue
        symbols.append(normalize_symbol(parts[0]))
        if len(symbols) >= max_symbols and natoms > max_symbols:
            break

    return natoms, symbols


def fallback_molecule_metadata(path_raw):
    lower = path_raw.lower()
    if "h2o" in lower:
        return 3, ["O", "H", "H"]
    if "n2" in lower:
        return 2, ["N", "N"]
    if "h3o" in lower:
        return 4, ["O", "H", "H", "H"]
    if "oh" in lower:
        return 2, ["O", "H"]
    if lower.endswith("/na.xyz") or "/na." in lower or lower.endswith("na.xyz"):
        return 1, ["Na"]
    if lower.endswith("/cl.xyz") or "/cl." in lower or lower.endswith("cl.xyz"):
        return 1, ["Cl"]
    return None, []


def resolve_structure_path(path_raw, packmol_dir):
    p = Path(path_raw)
    if p.is_absolute():
        return p
    return (packmol_dir / p).resolve()


def classify_block(path_raw, symbols, has_fixed):
    lower = path_raw.lower()
    counts = Counter(symbols)

    if "n2" in lower or counts == Counter({"N": 2}):
        return "n2"

    if "h2o" in lower or counts == Counter({"O": 1, "H": 2}):
        return "water"

    if "tio2" in lower or ("Ti" in counts and "O" in counts) or has_fixed:
        return "slab"

    return "ion_or_other"


def parse_packmol_blocks(packmol_path):
    lines = packmol_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    block_index = 0
    assumptions = []
    blocks = []

    while i < len(lines):
        line = lines[i].strip()
        if not line.lower().startswith("structure "):
            i += 1
            continue

        block_index += 1
        path_raw = line.split(None, 1)[1].strip()
        number = None
        inside_sphere = None
        has_fixed = False

        i += 1
        while i < len(lines):
            sub = lines[i].strip()
            low = sub.lower()
            if low.startswith("number "):
                try:
                    number = int(sub.split()[1])
                except Exception:
                    number = None
            elif low.startswith("inside sphere") and inside_sphere is None:
                toks = sub.split()
                if len(toks) >= 6:
                    try:
                        inside_sphere = (
                            float(toks[2]),
                            float(toks[3]),
                            float(toks[4]),
                            float(toks[5]),
                        )
                    except Exception:
                        inside_sphere = None
            elif low.startswith("fixed"):
                has_fixed = True
            elif low.startswith("end structure"):
                break
            i += 1

        if number is None:
            number = 0
            assumptions.append("PACKMOL block '{}' missing number; treated as 0".format(path_raw))

        resolved = resolve_structure_path(path_raw, packmol_path.parent)
        atoms_per_mol = None
        symbols = []

        if resolved.exists() and resolved.suffix.lower() == ".xyz":
            try:
                atoms_per_mol, symbols = parse_xyz_metadata(resolved)
                if atoms_per_mol <= 50:
                    # Read full symbols for small molecules so oxygen offset is reliable.
                    full_lines = resolved.read_text(encoding="utf-8", errors="ignore").splitlines()
                    full_symbols = []
                    for line_xyz in full_lines[2:2 + atoms_per_mol]:
                        parts = line_xyz.split()
                        if parts:
                            full_symbols.append(normalize_symbol(parts[0]))
                    if len(full_symbols) == atoms_per_mol:
                        symbols = full_symbols
            except Exception as exc:
                assumptions.append("Failed to parse xyz '{}': {}".format(resolved, exc))

        if atoms_per_mol is None:
            atoms_per_mol, symbols_fb = fallback_molecule_metadata(path_raw)
            symbols = symbols if symbols else symbols_fb
            if atoms_per_mol is None:
                raise ValueError(
                    "Cannot infer atoms-per-molecule for PACKMOL block '{}'. "
                    "Provide readable structure xyz or extend fallback mapping.".format(path_raw)
                )
            assumptions.append("Used fallback atom metadata for '{}'".format(path_raw))

        category = classify_block(path_raw, symbols, has_fixed)

        blocks.append(
            StructureBlock(
                index=block_index,
                path_raw=path_raw,
                path_resolved=resolved,
                number=number,
                inside_sphere=inside_sphere,
                has_fixed=has_fixed,
                atoms_per_mol=atoms_per_mol,
                symbols=symbols,
                category=category,
            )
        )

        i += 1

    return blocks, assumptions


def parse_total_atoms_header(data_path):
    with data_path.open(encoding="utf-8", errors="ignore") as f:
        for _ in range(300):
            line = f.readline()
            if not line:
                break
            m = TOTAL_ATOMS_RE.match(line)
            if m:
                return int(m.group(1))
    raise ValueError("Could not find total atom count in header: {}".format(data_path))


def assign_block_ranges(blocks):
    cursor = 1
    for blk in blocks:
        blk_atoms = blk.number * blk.atoms_per_mol
        if blk_atoms < 0:
            raise ValueError("Negative block atom count for '{}'".format(blk.path_raw))
        blk.range_start = cursor
        blk.range_end = cursor + blk_atoms - 1 if blk_atoms > 0 else cursor - 1
        cursor = blk.range_end + 1
    return cursor - 1


def compress_indices(indices):
    if not indices:
        raise ValueError("Cannot compress empty index list")

    if len(indices) == 1:
        return str(indices[0])

    stride = indices[1] - indices[0]
    if stride > 0 and all(indices[i] - indices[i - 1] == stride for i in range(2, len(indices))):
        if stride == 1:
            return "{}-{}".format(indices[0], indices[-1])
        return "{}-{}:{}".format(indices[0], indices[-1], stride)

    parts = []
    s = indices[0]
    e = s
    for x in indices[1:]:
        if x == e + 1:
            e = x
        else:
            parts.append(str(s) if s == e else "{}-{}".format(s, e))
            s = e = x
    parts.append(str(s) if s == e else "{}-{}".format(s, e))
    return ",".join(parts)


def range_to_str(rng):
    return "{}-{}".format(rng[0], rng[1])


def ranges_to_str(ranges):
    if not ranges:
        return "(none)"
    return ",".join([range_to_str(r) for r in ranges])


def choose_axis(c1, c2):
    deltas = [abs(c2[0] - c1[0]), abs(c2[1] - c1[1]), abs(c2[2] - c1[2])]
    idx = max(range(3), key=lambda i: deltas[i])
    return ["X", "Y", "Z"][idx]


def infer_summary(packmol_file, data_file, build_info, case_label):
    blocks, assumptions = parse_packmol_blocks(packmol_file)
    if not blocks:
        raise ValueError("No PACKMOL structure blocks found in {}".format(packmol_file))

    total_from_blocks = assign_block_ranges(blocks)
    total_from_data = parse_total_atoms_header(data_file)

    if total_from_blocks != total_from_data:
        raise ValueError(
            "Block-derived total atoms ({}) != model_atomic.data header ({}) for case '{}'.".format(
                total_from_blocks, total_from_data, case_label
            )
        )

    slab_blocks = [b for b in blocks if b.category == "slab"]
    water_blocks = [b for b in blocks if b.category == "water"]
    n2_blocks = [b for b in blocks if b.category == "n2"]
    ion_blocks = [b for b in blocks if b.category == "ion_or_other"]

    if not water_blocks:
        raise ValueError("No water blocks found for case '{}'".format(case_label))
    if len(n2_blocks) < 2:
        raise ValueError("Need at least two N2 blocks for case '{}'".format(case_label))

    # Build water oxygen ID list from water blocks only.
    water_oxygen_ids = []
    for wb in water_blocks:
        symbols = wb.symbols
        o_offsets = [i for i, sym in enumerate(symbols) if sym == "O"]
        if len(o_offsets) != 1:
            assumptions.append(
                "Water block '{}' expected one O per molecule; using first O offset from symbols {}".format(
                    wb.path_raw, symbols
                )
            )
        if not o_offsets:
            raise ValueError("Water block '{}' has no oxygen in molecule symbols {}".format(wb.path_raw, symbols))

        o_offset = o_offsets[0]
        for m in range(wb.number):
            water_oxygen_ids.append(wb.range_start + m * wb.atoms_per_mol + o_offset)

    water_oxygen_expr = compress_indices(water_oxygen_ids)

    # Bubble A/B from first two N2 blocks in PACKMOL order.
    n2a = n2_blocks[0]
    n2b = n2_blocks[1]

    def make_n2_pairs(block):
        if block.atoms_per_mol != 2:
            raise ValueError(
                "N2 block '{}' has atoms_per_mol={} (expected 2).".format(
                    block.path_raw, block.atoms_per_mol
                )
            )
        pairs = []
        for m in range(block.number):
            base = block.range_start + m * block.atoms_per_mol
            pairs.append((base, base + 1))
        return pairs

    bubble_a_pairs = make_n2_pairs(n2a)
    bubble_b_pairs = make_n2_pairs(n2b)

    if n2a.inside_sphere is not None and n2b.inside_sphere is not None:
        c1 = (n2a.inside_sphere[0], n2a.inside_sphere[1], n2a.inside_sphere[2])
        c2 = (n2b.inside_sphere[0], n2b.inside_sphere[1], n2b.inside_sphere[2])
        axis = choose_axis(c1, c2)
        spacing = math.sqrt(sum((c1[i] - c2[i]) ** 2 for i in range(3)))
        radius_a = n2a.inside_sphere[3]
        radius_b = n2b.inside_sphere[3]
    else:
        axis = "X"
        spacing = build_info.get("BUBBLE_SPACING", 0.0)
        radius_a = build_info.get("GAS_RADIUS_1", build_info.get("GAS_RADIUS", 0.0))
        radius_b = build_info.get("GAS_RADIUS_2", build_info.get("GAS_RADIUS", radius_a))
        assumptions.append("N2 inside-sphere geometry missing; used build-script fallback")

    if radius_a <= 0:
        radius_a = build_info.get("GAS_RADIUS_1", build_info.get("GAS_RADIUS", 0.0))
    if radius_b <= 0:
        radius_b = build_info.get("GAS_RADIUS_2", build_info.get("GAS_RADIUS", radius_a))
    if spacing <= 0:
        spacing = build_info.get("BUBBLE_SPACING", 0.0)

    if radius_a <= 0 or radius_b <= 0 or spacing <= 0:
        raise ValueError(
            "Could not infer valid GAS_RADIUS_1/GAS_RADIUS_2/BUBBLE_SPACING for case '{}'".format(case_label)
        )

    # For unequal bubbles, use the smaller radius to define the bridge cylinder
    # so the bridge-water descriptor remains localized between the two bubbles
    # and does not become too wide for the smaller bubble.
    effective_bridge_radius_source = min(radius_a, radius_b)
    bridge_radius = max(4.0, min(0.35 * effective_bridge_radius_source, 8.0))
    bridge_radius = round(bridge_radius * 2.0) / 2.0
    bridge_half = max(5.0, min(0.16 * spacing, 10.0))
    bridge_half = round(bridge_half * 2.0) / 2.0

    slab_ranges = [(b.range_start, b.range_end) for b in slab_blocks if b.range_end >= b.range_start]
    water_ranges = [(b.range_start, b.range_end) for b in water_blocks if b.range_end >= b.range_start]
    ion_ranges = [(b.range_start, b.range_end) for b in ion_blocks if b.range_end >= b.range_start]
    n2_ranges = [(b.range_start, b.range_end) for b in n2_blocks if b.range_end >= b.range_start]

    oxygen_ion_labels = []
    for b in ion_blocks:
        if "O" in Counter(b.symbols):
            oxygen_ion_labels.append(b.path_raw)

    block_lines = []
    for b in blocks:
        blk_atoms = b.number * b.atoms_per_mol
        block_lines.append(
            "# - block{idx:02d} [{cat}] {path} | number={num} atoms_per_mol={apm} total_atoms={tot} range={r}".format(
                idx=b.index,
                cat=b.category,
                path=b.path_raw,
                num=b.number,
                apm=b.atoms_per_mol,
                tot=blk_atoms,
                r=range_to_str((b.range_start, b.range_end)) if b.range_end >= b.range_start else "(empty)",
            )
        )

    return ParseSummary(
        total_atoms=total_from_data,
        slab_ranges=slab_ranges,
        water_ranges=water_ranges,
        ion_ranges=ion_ranges,
        n2_ranges=n2_ranges,
        water_oxygen_expr=water_oxygen_expr,
        bubble_a_pairs=bubble_a_pairs,
        bubble_b_pairs=bubble_b_pairs,
        bubble_a_range=(n2a.range_start, n2a.range_end),
        bubble_b_range=(n2b.range_start, n2b.range_end),
        gas_radius_a=radius_a,
        gas_radius_b=radius_b,
        bubble_spacing=spacing,
        axis=axis,
        bridge_radius=bridge_radius,
        bridge_half_length=bridge_half,
        assumptions=assumptions,
        oxygen_ion_labels=oxygen_ion_labels,
        case_label=case_label,
        block_lines=block_lines,
    )


def generate_plumed(summary, output_file, data_file, packmol_file, build_py):
    a_labels = ["a{:03d}".format(i) for i in range(1, len(summary.bubble_a_pairs) + 1)]
    b_labels = ["b{:03d}".format(i) for i in range(1, len(summary.bubble_b_pairs) + 1)]

    wall_a = max(100.0, round(2.5 * len(summary.bubble_a_pairs), 1))
    wall_b = max(100.0, round(2.5 * len(summary.bubble_b_pairs), 1))

    lines = []
    lines.append("UNITS LENGTH=A")
    lines.append("")
    lines.append("# ============================================================================")
    lines.append("# {}".format(output_file.name))
    lines.append("# Auto-generated by molsimflow.plumed.double_bubble")
    lines.append("# Case label: {}".format(summary.case_label))
    lines.append("#")
    lines.append("# Sources used for automatic inference:")
    lines.append("# - data:    {}".format(data_file))
    lines.append("# - packmol: {}".format(packmol_file))
    if build_py is not None:
        lines.append("# - build:   {}".format(build_py))
    lines.append("#")
    lines.append("# Inferred ordering/ranges from PACKMOL block sequence:")
    lines.append("# - slab ranges: {}".format(ranges_to_str(summary.slab_ranges)))
    lines.append("# - water ranges: {}".format(ranges_to_str(summary.water_ranges)))
    lines.append("# - electrolyte/other ranges: {}".format(ranges_to_str(summary.ion_ranges)))
    lines.append("# - N2 ranges: {}".format(ranges_to_str(summary.n2_ranges)))
    lines.append("# - bubble A N2 range: {}".format(range_to_str(summary.bubble_a_range)))
    lines.append("# - bubble B N2 range: {}".format(range_to_str(summary.bubble_b_range)))
    lines.append("# - water oxygen group (water-only): {}".format(summary.water_oxygen_expr))
    lines.append("#   slab oxygens and electrolyte species are excluded by using H2O blocks only.")
    if summary.oxygen_ion_labels:
        lines.append("#   oxygen-containing electrolyte blocks excluded from bridge-water group: {}".format(
            ", ".join(summary.oxygen_ion_labels)
        ))
    lines.append("#")
    lines.append("# Bubble geometry inference:")
    lines.append("# - bubble A radius: {:.3f} A".format(summary.gas_radius_a))
    lines.append("# - bubble B radius: {:.3f} A".format(summary.gas_radius_b))
    lines.append("# - representative/max bubble radius: {:.3f} A".format(summary.gas_radius))
    lines.append("# - nominal center spacing: {:.3f} A".format(summary.bubble_spacing))
    lines.append("# - bridge axis direction: {}".format(summary.axis))
    lines.append("#")
    lines.append("# Bridge-cylinder settings (bridge-focused, not full-bubble-sized):")
    lines.append("# - cylinder radius R_0={:.1f} A".format(summary.bridge_radius))
    lines.append("# - axial half-window +/-{:.1f} A".format(summary.bridge_half_length))
    lines.append("#")
    lines.append("# PACKMOL block summary used for range inference:")
    lines.extend(summary.block_lines)
    if summary.assumptions:
        lines.append("#")
        lines.append("# Assumptions/notes:")
        for s in summary.assumptions:
            lines.append("# - {}".format(s))
    lines.append("# ============================================================================")
    lines.append("")

    lines.append("# Main scientific CVs:")
    lines.append("# - d3d_all: 3D distance between all-member bubble centers")
    lines.append("# - bridge_cyl_env.{sum,mean}: bridge-region local water-environment descriptor")
    lines.append("# - bridge_mid is optional and commented out by default")
    lines.append("")

    lines.append("# ----------------------------------------------------------------------------")
    lines.append("# A. Bubble A N2 molecular COM virtual atoms")
    lines.append("# Bubble A uses the first N2 PACKMOL sphere block in output order.")
    lines.append("# ----------------------------------------------------------------------------")
    for i, (a1, a2) in enumerate(summary.bubble_a_pairs, start=1):
        lines.append("a{:03d}: COM ATOMS={},{}".format(i, a1, a2))
    lines.append("")

    lines.append("# ----------------------------------------------------------------------------")
    lines.append("# B. Bubble B N2 molecular COM virtual atoms")
    lines.append("# Bubble B uses the second N2 PACKMOL sphere block in output order.")
    lines.append("# ----------------------------------------------------------------------------")
    for i, (b1, b2) in enumerate(summary.bubble_b_pairs, start=1):
        lines.append("b{:03d}: COM ATOMS={},{}".format(i, b1, b2))
    lines.append("")

    lines.append("# Groups of N2 molecular COMs")
    lines.append("repsA: GROUP ATOMS=" + ",".join(a_labels))
    lines.append("repsB: GROUP ATOMS=" + ",".join(b_labels))
    lines.append("")

    lines.append("# All-member bubble centers and main distance CV")
    lines.append("bubA_all: CENTER ATOMS=" + ",".join(a_labels))
    lines.append("bubB_all: CENTER ATOMS=" + ",".join(b_labels))
    lines.append("d3d_all: DISTANCE ATOMS=bubA_all,bubB_all")
    lines.append("")

    lines.append("# ----------------------------------------------------------------------------")
    lines.append("# C. Per-bubble cluster-integrity channels (A and B separated)")
    lines.append("# ----------------------------------------------------------------------------")
    lines.append("matA: CONTACT_MATRIX ATOMS=repsA SWITCH={CUBIC D_0=5.0 D_MAX=6.0}")
    lines.append("dfsA: DFSCLUSTERING MATRIX=matA LOWMEM")
    lines.append("n2A_num: CLUSTER_NATOMS CLUSTERS=dfsA CLUSTER=1")
    lines.append("sumA_cn: CLUSTER_PROPERTIES CLUSTERS=dfsA CLUSTER=1 SUM")
    lines.append("")
    lines.append("matB: CONTACT_MATRIX ATOMS=repsB SWITCH={CUBIC D_0=5.0 D_MAX=6.0}")
    lines.append("dfsB: DFSCLUSTERING MATRIX=matB LOWMEM")
    lines.append("n2B_num: CLUSTER_NATOMS CLUSTERS=dfsB CLUSTER=1")
    lines.append("sumB_cn: CLUSTER_PROPERTIES CLUSTERS=dfsB CLUSTER=1 SUM")
    lines.append("")

    lines.append("# Cluster-integrity walls (auxiliary restraints for integrity control)")
    lines.append("wallA: LOWER_WALLS ARG=sumA_cn.sum AT={:.1f} KAPPA=5 EXP=2".format(wall_a))
    lines.append("wallB: LOWER_WALLS ARG=sumB_cn.sum AT={:.1f} KAPPA=5 EXP=2".format(wall_b))
    lines.append("")

    lines.append("# ----------------------------------------------------------------------------")
    lines.append("# D. Bridge-water descriptors (water oxygen only)")
    lines.append("# Keep old bridge_mid as a comparison channel.")
    lines.append("# Add bridge_cyl_env as bridge-region local water-environment average.")
    lines.append("# Bridge axis: {} ; R_0={:.1f} A ; axial window=[-{:.1f}, +{:.1f}] A".format(
        summary.axis, summary.bridge_radius, summary.bridge_half_length, summary.bridge_half_length
    ))
    lines.append("# Water-oxygen group is built ONLY from H2O blocks; slab O and ions are excluded.")
    lines.append("wO: GROUP ATOMS={}".format(summary.water_oxygen_expr))
    lines.append("mid_all: CENTER ATOMS=bubA_all,bubB_all")
    lines.append("wcn_bridge: COORDINATIONNUMBER SPECIES=wO R_0=3.2 NN=6 MM=12")
    lines.append(
        "#bridge_mid: COORDINATION GROUPA=mid_all GROUPB={} R_0=4.0 NN=6 MM=12 NLIST NL_CUTOFF=6.0 NL_STRIDE=20".format(
            summary.water_oxygen_expr
        )
    )
    lines.append(
        "bridge_cyl_env: INCYLINDER ATOM=mid_all DATA=wcn_bridge DIRECTION={} RADIUS={{TANH R_0={:.1f}}} LOWER=-{:.1f} UPPER={:.1f} KERNEL=gaussian MEAN SUM".format(
            summary.axis, summary.bridge_radius, summary.bridge_half_length, summary.bridge_half_length
        )
    )
    lines.append("")

    lines.append("# ----------------------------------------------------------------------------")
    lines.append("# E. Lightweight contamination diagnostics (all-member centers, unchanged)")
    lines.append("# ----------------------------------------------------------------------------")
    diag_r_a = max(1.0, 0.95 * summary.gas_radius_a)
    diag_r_b = max(1.0, 0.95 * summary.gas_radius_b)
    lines.append("#ownA_all:   COORDINATION GROUPA=bubA_all GROUPB=repsA R_0={:.1f} NN=6 MM=12 NLIST NL_CUTOFF=7.0 NL_STRIDE=20".format(diag_r_a))
    lines.append("#crossA_all: COORDINATION GROUPA=bubA_all GROUPB=repsB R_0={:.1f} NN=6 MM=12 NLIST NL_CUTOFF=7.0 NL_STRIDE=20".format(diag_r_a))
    lines.append("#ownB_all:   COORDINATION GROUPA=bubB_all GROUPB=repsB R_0={:.1f} NN=6 MM=12 NLIST NL_CUTOFF=7.0 NL_STRIDE=20".format(diag_r_b))
    lines.append("#crossB_all: COORDINATION GROUPA=bubB_all GROUPB=repsA R_0={:.1f} NN=6 MM=12 NLIST NL_CUTOFF=7.0 NL_STRIDE=20".format(diag_r_b))
    lines.append("")

    lines.append("#OPES_METAD_EXPLORE ...")
    lines.append("#  RESTART=NO")
    lines.append("#  LABEL=opes_e")
    lines.append("#  ARG=d3d_all")
    lines.append("#  FILE=HILLS_e")
    lines.append("#  TEMP=330")
    lines.append("#  PACE=500")
    lines.append("#  BARRIER=25")
    lines.append("#  STATE_WFILE=STATE_e")
    lines.append("#  STATE_WSTRIDE=5000")
    lines.append("#  STORE_STATES")
    lines.append("#  WALKERS_MPI")
    lines.append("#... OPES_METAD_EXPLORE")
    lines.append("")

    lines.append("OPES_METAD ...")
    lines.append("  RESTART=NO")
    lines.append("  LABEL=opes")
    lines.append("  ARG=d3d_all")
    lines.append("  FILE=HILLS")
    lines.append("  TEMP=330")
    lines.append("  PACE=500")
    lines.append("  BARRIER=50")
    lines.append("  STATE_WFILE=STATE")
    lines.append("  STATE_WSTRIDE=5000")
    lines.append("  STORE_STATES")
    lines.append("  WALKERS_MPI")
    lines.append("... OPES_METAD")
    lines.append("")

    lines.append("# Output for monitoring/debug")
    lines.append("PRINT STRIDE=1  FILE=COLVAR ARG=d3d_all,n2A_num,n2B_num,sumA_cn.sum,sumB_cn.sum,opes.bias #,opes_e.bias")
    lines.append("PRINT STRIDE=10 FILE=COLVAR_post ARG=bridge_cyl_env.sum,bridge_cyl_env.mean,wallA.bias,wallB.bias")
    lines.append("FLUSH STRIDE=1000")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_case_label(packmol_path, explicit_label):
    if explicit_label:
        return explicit_label
    parent = packmol_path.parent.parent.name if packmol_path.parent.parent else packmol_path.parent.name
    return parent


def generate_double_bubble_plumed(data_file, packmol_file, output_file, build_py=None, case_label=""):
    """Generate a PLUMED file and return the inferred parse summary."""
    data_path = Path(data_file).expanduser()
    packmol_path = Path(packmol_file).expanduser()
    output_path = Path(output_file).expanduser()
    build_path = Path(build_py).expanduser() if build_py else None

    if not data_path.exists():
        raise FileNotFoundError("LAMMPS data file not found: {}".format(data_path))
    if not packmol_path.exists():
        raise FileNotFoundError("PACKMOL input file not found: {}".format(packmol_path))
    if build_path is not None and not build_path.exists():
        raise FileNotFoundError("Build metadata Python file not found: {}".format(build_path))

    build_info = parse_build_script(build_path)
    inferred_case_label = infer_case_label(packmol_path, case_label)
    summary = infer_summary(
        packmol_file=packmol_path,
        data_file=data_path,
        build_info=build_info,
        case_label=inferred_case_label,
    )
    generate_plumed(
        summary=summary,
        output_file=output_path,
        data_file=data_path,
        packmol_file=packmol_path,
        build_py=build_path,
    )
    return summary


def build_arg_parser():
    """Build the standalone argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate slab/electrolyte PLUMED file with water-only bridge group and unequal double-bubble support."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to model_atomic.data",
    )
    parser.add_argument(
        "--packmol",
        required=True,
        help="Path to PACKMOL input",
    )
    parser.add_argument(
        "--build-py",
        default=None,
        help="Optional build python for GAS_RADIUS/BUBBLE_SPACING fallback",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PLUMED file",
    )
    parser.add_argument(
        "--case-label",
        default="",
        help="Optional label for comments (default inferred from packmol directory)",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = generate_double_bubble_plumed(
        data_file=args.data,
        packmol_file=args.packmol,
        output_file=args.output,
        build_py=args.build_py,
        case_label=args.case_label,
    )

    print("Generated PLUMED file:")
    print("  {}".format(Path(args.output).expanduser()))
    print("Case label: {}".format(summary.case_label))
    print("Summary:")
    print("  total_atoms={}".format(summary.total_atoms))
    print("  slab_ranges={}".format(ranges_to_str(summary.slab_ranges)))
    print("  water_ranges={}".format(ranges_to_str(summary.water_ranges)))
    print("  ion_ranges={}".format(ranges_to_str(summary.ion_ranges)))
    print("  n2_ranges={}".format(ranges_to_str(summary.n2_ranges)))
    print("  water_oxygen_expr={}".format(summary.water_oxygen_expr))
    print("  bubbleA_range={} bubbleB_range={}".format(range_to_str(summary.bubble_a_range), range_to_str(summary.bubble_b_range)))
    print("  n2_pairs_A={} n2_pairs_B={}".format(len(summary.bubble_a_pairs), len(summary.bubble_b_pairs)))
    print("  gas_radius_A={:.3f} gas_radius_B={:.3f} spacing={:.3f} axis={}".format(
        summary.gas_radius_a, summary.gas_radius_b, summary.bubble_spacing, summary.axis
    ))
    print("  bridge_radius={:.1f} bridge_half_length={:.1f}".format(summary.bridge_radius, summary.bridge_half_length))
    if summary.oxygen_ion_labels:
        print("  oxygen-containing ion blocks excluded from water bridge group: {}".format(", ".join(summary.oxygen_ion_labels)))


if __name__ == "__main__":
    main()
