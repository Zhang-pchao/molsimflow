#!/usr/bin/env python3
"""Visualize manifest-listed DPA4C training systems in descriptor space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from molsimflow.postprocess.deepmd_dataset_sketch import (
    DatasetBundle,
    PERIODIC_DISPLAY_ORDER,
    build_model_type_indices,
    compute_pca_embedding,
    compute_tsne_embedding,
    preprocess_descriptor_features,
    select_representative_per_config,
)

ALLOWED_ELEMENTS = {"H", "O", "N", "Na", "Cl", "Ti", "Si", "C"}
FAMILY_TITLES = {
    "hon_primary": "HON", "hon_rest": "HON", "nacl_95pct": "NaCl",
    "tio2_95pct": r"TiO$_2$", "sio2_dft": r"SiO$_2$",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if [int(row["index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Training manifest indices are not contiguous")
    if not rows:
        raise ValueError("Training manifest contains no systems")
    for row in rows:
        if not Path(row["system_path"]).is_dir():
            raise FileNotFoundError(row["system_path"])
    return rows


def sample_indices(nframes: int, count: int) -> np.ndarray:
    if nframes <= 0:
        raise ValueError("Dataset contains no frames")
    count = min(nframes, count)
    return np.linspace(0, nframes - 1, count, dtype=int)


def load_deepmd_npy(
    path: Path, frame_limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str], int, np.ndarray]:
    """Load uniformly spaced frames without materializing the full dataset."""

    type_symbols = path.joinpath("type_map.raw").read_text().split()
    atom_types = np.loadtxt(path / "type.raw", dtype=int, ndmin=1).reshape(-1).tolist()
    set_records = []
    for set_dir in sorted(path.glob("set.*")):
        energy = np.load(set_dir / "energy.npy", mmap_mode="r")
        set_records.append((set_dir, len(energy)))
    available_frames = sum(nframes for _, nframes in set_records)
    if not available_frames:
        raise ValueError(f"No set.* data found in {path}")
    selected = sample_indices(available_frames, frame_limit)
    offsets = np.cumsum([0] + [nframes for _, nframes in set_records])
    coords, cells, energies = [], [], []
    arrays = {}
    for global_index in selected:
        set_index = int(np.searchsorted(offsets[1:], global_index, side="right"))
        set_dir = set_records[set_index][0]
        local_index = int(global_index - offsets[set_index])
        if set_dir not in arrays:
            arrays[set_dir] = (
                np.load(set_dir / "coord.npy", mmap_mode="r"),
                np.load(set_dir / "box.npy", mmap_mode="r"),
                np.load(set_dir / "energy.npy", mmap_mode="r"),
            )
        coord, box, energy = arrays[set_dir]
        coords.append(np.asarray(coord[local_index], dtype=np.float64).reshape(len(atom_types), 3))
        cells.append(np.asarray(box[local_index], dtype=np.float64).reshape(3, 3))
        energies.append(float(np.asarray(energy[local_index]).reshape(-1)[0]))
    return (
        np.stack(coords), np.stack(cells), np.asarray(energies), atom_types,
        type_symbols, available_frames, selected,
    )


def composition_counts(
    atom_types: list[int], type_symbols: list[str]
) -> dict[str, int]:
    return {
        symbol: atom_types.count(index)
        for index, symbol in enumerate(type_symbols)
        if atom_types.count(index)
    }


def composition_key(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counts.items()))


def composition_label(atom_types: list[int], type_symbols: list[str]) -> str:
    counts = composition_counts(atom_types, type_symbols)
    ordered = [symbol for symbol in PERIODIC_DISPLAY_ORDER if counts.get(symbol, 0) > 0]
    ordered.extend(sorted(symbol for symbol, count in counts.items() if count > 0 and symbol not in ordered))
    return "".join(f"{symbol}{counts[symbol]}" for symbol in ordered)


def formula_key(formula: str) -> tuple[tuple[str, int], ...]:
    counts = Counter()
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[element] += int(count or 1)
    if not counts:
        raise ValueError(f"Cannot parse composition formula: {formula}")
    return composition_key(dict(counts))


def read_table_s4(
    path: Path, expected_rows: int | None = 296, expected_systems: int | None = 192
) -> tuple[list[tuple[tuple[str, int], ...]], list[dict[str, object]]]:
    text = path.read_text()
    start = text.index(r"\label{tab:dpa4c_inventory}")
    end = text.index(r"\end{longtable}", start)
    canonical_ids: dict[tuple[tuple[str, int], ...], int] = {}
    rows = []
    for line in text[start:end].splitlines():
        match = re.match(
            r"\s*(?:\\num\{\d+\}\s*&\s*)?\\ce\{([^}]*)\}\s*&\s*([^&]+)&", line
        )
        if not match:
            continue
        formula, reference = match.groups()
        key = formula_key(formula)
        canonical_ids.setdefault(key, len(canonical_ids) + 1)
        rows.append({
            "table_row": len(rows) + 1,
            "subsystem_id": canonical_ids[key],
            "composition": formula,
            "reference": reference.strip(),
            "composition_key": key,
        })
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} Table S4 rows, found {len(rows)}")
    if expected_systems is not None and len(canonical_ids) != expected_systems:
        raise ValueError(
            f"Expected {expected_systems} unique Table S4 systems, found {len(canonical_ids)}"
        )
    return list(canonical_ids), rows


def select_unique_compositions(
    rows: list[dict[str, str]],
    ordered_keys: list[tuple[tuple[str, int], ...]] | None = None,
    expected_count: int | None = 192,
) -> list[dict[str, str]]:
    """Keep one DFT-preferred system per composition in Table S4 order."""

    candidates: dict[tuple[tuple[str, int], ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["block"] == "sio2_distilled":
            continue
        path = Path(row["system_path"])
        type_symbols = path.joinpath("type_map.raw").read_text().split()
        atom_types = np.loadtxt(path / "type.raw", dtype=int, ndmin=1).reshape(-1).tolist()
        used_elements = {type_symbols[index] for index in atom_types}
        if not used_elements <= ALLOWED_ELEMENTS:
            raise ValueError(f"Unsupported elements in {path}: {sorted(used_elements - ALLOWED_ELEMENTS)}")
        candidates[composition_key(composition_counts(atom_types, type_symbols))].append(row)
    if ordered_keys is None:
        ordered_keys = list(candidates)
    selected = []
    for key in ordered_keys:
        if key not in candidates:
            raise ValueError(f"Table S4 composition has no DFT-preferred training system: {key}")
        row = candidates[key][0]
        selected_row = dict(row)
        selected_row["original_manifest_index"] = row["index"]
        selected_row["index"] = str(len(selected))
        selected.append(selected_row)
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique DFT-preferred compositions, found {len(selected)}"
        )
    return selected


def load_sampled_bundle(
    rows: list[dict[str, str]], frames_per_system: int
) -> tuple[DatasetBundle, list[dict[str, object]], list[np.ndarray]]:
    coords_list, cells_list, energies_list = [], [], []
    atom_types_list, type_symbols_list = [], []
    n_atoms_list, system_configs, source_paths, frame_to_dataset = [], [], [], []
    records: list[dict[str, object]] = []
    selected_local_indices: list[np.ndarray] = []

    for loaded_index, row in enumerate(rows):
        path = Path(row["system_path"])
        (
            coords, cells, energies, atom_types, type_symbols, available_frames, indices,
        ) = load_deepmd_npy(path, frames_per_system)
        selected_local_indices.append(indices)
        config = composition_label(atom_types, type_symbols)
        natoms = int(coords.shape[1])

        coords_list.append(coords)
        cells_list.append(cells)
        energies_list.append(energies)
        atom_types_list.append(atom_types)
        type_symbols_list.append(type_symbols)
        n_atoms_list.extend([natoms] * len(coords))
        system_configs.extend([config] * len(coords))
        source_paths.extend([str(path)] * len(coords))
        frame_to_dataset.extend([loaded_index] * len(coords))
        records.append({
            "dataset_id": loaded_index + 1,
            "manifest_index": int(row["index"]),
            "training_system_id": int(row.get("original_manifest_index", row["index"])) + 1,
            "family": row["block"],
            "composition": config,
            "system_name": path.name,
            "element_counts": ";".join(
                f"{element}:{count}"
                for element, count in composition_key(composition_counts(atom_types, type_symbols))
            ),
            "element_count": len(composition_counts(atom_types, type_symbols)),
            "natoms": natoms,
            "available_frames": available_frames,
            "selected_frames": len(indices),
            "selected_local_indices": ";".join(map(str, indices.tolist())),
            "system_path": str(path),
        })
        if loaded_index == 0 or (loaded_index + 1) % 25 == 0 or loaded_index + 1 == len(rows):
            print(f"loaded={loaded_index + 1}/{len(rows)} sampled_frames={sum(len(x) for x in selected_local_indices)}", flush=True)

    bundle = DatasetBundle(
        coords_list=coords_list,
        cells_list=cells_list,
        energies=np.concatenate(energies_list),
        n_atoms_list=n_atoms_list,
        atom_types_list=atom_types_list,
        type_symbols_list=type_symbols_list,
        system_configs=system_configs,
        source_paths=source_paths,
        frame_to_dataset=frame_to_dataset,
    )
    return bundle, records, selected_local_indices


def table_matching_blocks(table_row: dict[str, object]) -> set[str]:
    elements = dict(table_row["composition_key"])
    if str(table_row["reference"]) == "DPA4 pseudo-label":
        return {"sio2_distilled"}
    if "Si" in elements:
        return {"sio2_dft"}
    if "Ti" in elements:
        return {"tio2_95pct"}
    if "Na" in elements or "Cl" in elements:
        return {"nacl_95pct"}
    return {"hon_primary", "hon_rest"}


def enrich_table_rows(
    table_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    manifest_by_key: dict[tuple[tuple[str, int], ...], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        path = Path(row["system_path"])
        symbols = path.joinpath("type_map.raw").read_text().split()
        atom_types = np.loadtxt(path / "type.raw", dtype=int, ndmin=1).reshape(-1).tolist()
        manifest_by_key[composition_key(composition_counts(atom_types, symbols))].append(row)
    representative = {
        key: row for key, row in zip(dict.fromkeys(r["composition_key"] for r in table_rows), selected_rows)
    }
    enriched = []
    for table_row in table_rows:
        key = table_row["composition_key"]
        matches = [
            row for row in manifest_by_key[key] if row["block"] in table_matching_blocks(table_row)
        ]
        rep = representative[key]
        enriched.append({
            "table_row": table_row["table_row"],
            "subsystem_id": table_row["subsystem_id"],
            "composition": table_row["composition"],
            "reference": table_row["reference"],
            "matching_training_system_ids": ";".join(
                str(int(row["index"]) + 1) for row in matches
            ),
            "matching_system_names": ";".join(Path(row["system_path"]).name for row in matches),
            "representative_training_system_id": int(rep["original_manifest_index"]) + 1,
            "representative_system_name": Path(rep["system_path"]).name,
        })
    return enriched


def write_catalogs(
    output: Path,
    records: list[dict[str, object]],
    table_rows: list[dict[str, object]],
) -> None:
    catalog_fields = [
        "subsystem_id", "table_first_row", "composition", "element_count",
        "element_counts", "natoms", "representative_family",
        "representative_training_system_id", "system_name", "system_path",
        "available_frames", "selected_frames", "selected_local_indices",
    ]
    first_table_row = {}
    table_formula = {}
    for row in table_rows:
        first_table_row.setdefault(int(row["subsystem_id"]), int(row["table_row"]))
        table_formula.setdefault(int(row["subsystem_id"]), str(row["composition"]))
    with (output / "subsystem_catalog.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=catalog_fields)
        writer.writeheader()
        for record in records:
            subsystem_id = int(record["dataset_id"])
            writer.writerow({
                "subsystem_id": subsystem_id,
                "table_first_row": first_table_row[subsystem_id],
                "composition": table_formula[subsystem_id],
                "element_count": record["element_count"],
                "element_counts": record["element_counts"],
                "natoms": record["natoms"],
                "representative_family": record["family"],
                "representative_training_system_id": record["training_system_id"],
                "system_name": record["system_name"],
                "system_path": record["system_path"],
                "available_frames": record["available_frames"],
                "selected_frames": record["selected_frames"],
                "selected_local_indices": record["selected_local_indices"],
            })
    with (output / "table_s4_subsystem_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)


def save_sampled_bundle(
    output: Path,
    bundle: DatasetBundle,
    records: list[dict[str, object]],
    selected_local_indices: list[np.ndarray],
) -> None:
    sampled_input = output / "sampled_inputs"
    sampled_input.mkdir(parents=True, exist_ok=True)
    frame_offset = 0
    for dataset_index, record in enumerate(records):
        nframes = len(bundle.coords_list[dataset_index])
        np.savez_compressed(
            sampled_input / f"S{dataset_index + 1:03d}.npz",
            coords=bundle.coords_list[dataset_index],
            cells=bundle.cells_list[dataset_index],
            energies=bundle.energies[frame_offset : frame_offset + nframes],
            atom_types=np.asarray(bundle.atom_types_list[dataset_index], dtype=np.int32),
            type_symbols=np.asarray(bundle.type_symbols_list[dataset_index]),
            selected_local_indices=selected_local_indices[dataset_index],
        )
        frame_offset += nframes
    (output / "sampling_records.json").write_text(json.dumps(records, indent=2) + "\n")


def load_saved_bundle(output: Path) -> tuple[DatasetBundle, list[dict[str, object]], list[np.ndarray]]:
    records = json.loads((output / "sampling_records.json").read_text())
    coords_list, cells_list, energies_list = [], [], []
    atom_types_list, type_symbols_list, selected_local_indices = [], [], []
    n_atoms_list, system_configs, source_paths, frame_to_dataset = [], [], [], []
    for dataset_index, record in enumerate(records):
        data = np.load(output / "sampled_inputs" / f"S{dataset_index + 1:03d}.npz")
        coords = data["coords"]
        cells = data["cells"]
        energies = data["energies"]
        atom_types = data["atom_types"].astype(int).tolist()
        type_symbols = data["type_symbols"].astype(str).tolist()
        selected = data["selected_local_indices"].astype(int)
        nframes, natoms = len(coords), int(coords.shape[1])
        coords_list.append(coords)
        cells_list.append(cells)
        energies_list.append(energies)
        atom_types_list.append(atom_types)
        type_symbols_list.append(type_symbols)
        selected_local_indices.append(selected)
        n_atoms_list.extend([natoms] * nframes)
        system_configs.extend([str(record["composition"])] * nframes)
        source_paths.extend([str(record["system_path"])] * nframes)
        frame_to_dataset.extend([dataset_index] * nframes)
    bundle = DatasetBundle(
        coords_list=coords_list,
        cells_list=cells_list,
        energies=np.concatenate(energies_list),
        n_atoms_list=n_atoms_list,
        atom_types_list=atom_types_list,
        type_symbols_list=type_symbols_list,
        system_configs=system_configs,
        source_paths=source_paths,
        frame_to_dataset=frame_to_dataset,
    )
    return bundle, records, selected_local_indices


def evaluate_descriptors(
    bundle: DatasetBundle, model_path: Path, batch_size: int
) -> tuple[np.ndarray, list[int], list[int]]:
    import torch

    from deepmd.dpmodel.utils.neighbor_graph import NeighborGraph
    from deepmd.infer import DeepPot
    from deepmd.pt_expt.utils.env import DEVICE
    from deepmd.pt_expt.utils.vesin_neighbor_list import VesinNeighborList

    model = DeepPot(str(model_path))
    atomic_model = model.deep_eval._dpmodel.get_dp_atomic_model()
    if atomic_model is None or not hasattr(atomic_model.descriptor, "call_graph"):
        raise NotImplementedError("The selected model does not expose a graph descriptor")
    neighbor_builder = VesinNeighborList()
    descriptor_blocks, descriptor_raw_shapes = [], []
    frame_offsets = [0]
    for dataset_index, (coords, cells, atom_types, type_symbols) in enumerate(
        zip(bundle.coords_list, bundle.cells_list, bundle.atom_types_list, bundle.type_symbols_list)
    ):
        mapped_types = build_model_type_indices(type_symbols, atom_types, model)
        descriptor_chunks, edge_counts = [], []
        for start in range(0, len(coords), batch_size):
            stop = min(start + batch_size, len(coords))
            for frame in range(start, stop):
                coord_t = torch.tensor(np.array(coords[frame : frame + 1], copy=True), dtype=torch.float64, device=DEVICE)
                atype_t = torch.tensor([mapped_types], dtype=torch.int64, device=DEVICE)
                cell_t = torch.tensor(np.array(cells[frame : frame + 1], copy=True), dtype=torch.float64, device=DEVICE)
                edges = neighbor_builder.build(
                    coord_t, atype_t, cell_t, model.deep_eval._rcut,
                    model.deep_eval._sel, return_mode="edges",
                )
                natoms = len(mapped_types)
                graph = NeighborGraph(
                    n_node=torch.tensor([natoms], dtype=torch.int64, device=DEVICE),
                    n_local=torch.tensor([natoms], dtype=torch.int64, device=DEVICE),
                    edge_index=edges.edge_index,
                    edge_vec=edges.edge_vec,
                    edge_mask=edges.edge_mask,
                )
                with torch.no_grad():
                    local_descriptor, _ = atomic_model.descriptor.call_graph(
                        graph, edges.atype.reshape(-1)
                    )
                local_descriptor = local_descriptor.detach().cpu().numpy()
                descriptor_chunks.append(local_descriptor.mean(axis=0, keepdims=True))
                edge_counts.append(int(edges.edge_mask.sum().item()))
        reduced = np.concatenate(descriptor_chunks, axis=0)
        descriptor_raw_shapes.append([len(coords), len(mapped_types), reduced.shape[1]])
        if not np.isfinite(reduced).all():
            raise ValueError(f"Non-finite descriptor values in dataset {dataset_index}")
        descriptor_blocks.append(reduced)
        frame_offsets.append(frame_offsets[-1] + len(coords))
        if dataset_index == 0 or (dataset_index + 1) % 25 == 0 or dataset_index + 1 == len(bundle.coords_list):
            print(
                f"descriptors={dataset_index + 1}/{len(bundle.coords_list)} "
                f"reduced_shape={reduced.shape} edges={edge_counts}", flush=True,
            )
    widths = {block.shape[1] for block in descriptor_blocks}
    if len(widths) != 1:
        raise ValueError(f"Descriptor feature widths vary across systems: {sorted(widths)}")
    return np.concatenate(descriptor_blocks), frame_offsets, descriptor_raw_shapes[0]


def energy_per_atom(bundle: DatasetBundle) -> np.ndarray:
    """Return the reference energy per atom used by the figure color scale."""

    return bundle.energies / np.asarray(bundle.n_atoms_list, dtype=float)


def l2_normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    """Apply the structure-level normalization used by DP-EVA."""

    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / (norms + 1e-12)


def write_extxyz(
    path: Path, symbols: list[str], coords_list: np.ndarray, cells_list: np.ndarray
) -> None:
    with path.open("w") as handle:
        for coords, cell in zip(coords_list, cells_list):
            lattice = " ".join(f"{value:.16g}" for value in cell.reshape(-1))
            handle.write(f"{len(symbols)}\n")
            handle.write(
                f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
            )
            for symbol, xyz in zip(symbols, coords):
                handle.write(
                    f"{symbol} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}\n"
                )


def export_structures(
    bundle: DatasetBundle,
    records: list[dict[str, object]],
    representative_local: dict[str, int],
    output: Path,
) -> None:
    sampled_dir = output / "sampled_structures"
    representative_dir = output / "representative_structures"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    representative_dir.mkdir(parents=True, exist_ok=True)
    for dataset_index, record in enumerate(records):
        symbols_by_type = bundle.type_symbols_list[dataset_index]
        symbols = [symbols_by_type[index] for index in bundle.atom_types_list[dataset_index]]
        source = str(record["system_path"])
        identifier = f"S{int(record['dataset_id']):03d}"
        sampled_path = sampled_dir / f"{identifier}_selected.extxyz"
        write_extxyz(
            sampled_path,
            symbols,
            bundle.coords_list[dataset_index],
            bundle.cells_list[dataset_index],
        )
        global_index = representative_local[source]
        local_index = global_index - sum(len(bundle.coords_list[i]) for i in range(dataset_index))
        representative_path = representative_dir / f"{identifier}_representative.extxyz"
        write_extxyz(
            representative_path,
            symbols,
            bundle.coords_list[dataset_index][local_index : local_index + 1],
            bundle.cells_list[dataset_index][local_index : local_index + 1],
        )


def write_points(
    output: Path,
    embedding: np.ndarray,
    color_values: np.ndarray,
    bundle: DatasetBundle,
    records: list[dict[str, object]],
    selected_local_indices: list[np.ndarray],
    frame_offsets: list[int],
) -> None:
    with (output / "embedding_points.csv").open("w", newline="") as handle:
        fieldnames = [
            "dataset_id", "family", "composition", "source_local_frame",
            "embedding_1", "embedding_2",
            "energy_eV_per_atom", "system_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_index, record in enumerate(records):
            start = frame_offsets[dataset_index]
            for slot, source_local_frame in enumerate(selected_local_indices[dataset_index]):
                index = start + slot
                writer.writerow({
                    "dataset_id": int(record["dataset_id"]),
                    "family": record["family"],
                    "composition": record["composition"],
                    "source_local_frame": int(source_local_frame),
                    "embedding_1": float(embedding[index, 0]),
                    "embedding_2": float(embedding[index, 1]),
                    "energy_eV_per_atom": float(color_values[index]),
                    "system_path": record["system_path"],
                })


def plot_numbered(
    output: Path,
    embedding: np.ndarray,
    color_values: np.ndarray,
    source_paths: list[str],
    source_to_number: dict[str, int],
    representative_local: dict[str, int],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 1.0,
    })
    figure, ax = plt.subplots(figsize=(18, 8.2), constrained_layout=True)
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=color_values, cmap="viridis",
        s=20, alpha=0.45, edgecolors="none", rasterized=True,
    )
    representative_indices = np.asarray(list(representative_local.values()), dtype=int)
    ax.scatter(
        embedding[representative_indices, 0], embedding[representative_indices, 1],
        c=color_values[representative_indices], cmap="viridis", s=42, alpha=0.95,
        edgecolors="white", linewidths=0.35, zorder=3,
    )
    for source, frame_index in representative_local.items():
        x, y = embedding[frame_index]
        number = source_to_number[source]
        ax.annotate(
            str(number), xy=(x, y), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=6.0, color="#8b1a1a", fontweight="bold", zorder=4,
        )
    colorbar = figure.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.045, pad=0.035, aspect=35)
    colorbar.set_label(r"Reference energy (eV atom$^{-1}$)", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    figure.savefig(output / "sketch_map_num.png", dpi=450, bbox_inches="tight")
    figure.savefig(output / "sketch_map_num.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_plain(output: Path, embedding: np.ndarray, color_values: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(18, 8.2), constrained_layout=True)
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=color_values, cmap="viridis",
        s=22, alpha=0.72, edgecolors="none", rasterized=True,
    )
    colorbar = figure.colorbar(
        scatter, ax=ax, orientation="horizontal", fraction=0.045, pad=0.035, aspect=35
    )
    colorbar.set_label(r"Reference energy (eV atom$^{-1}$)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    figure.savefig(output / "sketch_map.png", dpi=450, bbox_inches="tight")
    plt.close(figure)


def family_frame_indices(
    records: list[dict[str, object]], frame_offsets: list[int]
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record["family"])].extend(range(frame_offsets[index], frame_offsets[index + 1]))
    return {family: np.asarray(indices, dtype=int) for family, indices in grouped.items()}


def compute_family_tsne_embedding(
    descriptors: np.ndarray,
    records: list[dict[str, object]],
    frame_offsets: list[int],
    random_state: int,
    perplexity: int,
    pca_components: int,
    early_exaggeration: float,
    learning_rate: float,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    """Fit an independent standardized PCA+t-SNE embedding for each figure panel."""

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for family, indices in family_frame_indices(records, frame_offsets).items():
        grouped[FAMILY_TITLES[family]].append(indices)
    embedding = np.empty((len(descriptors), 2), dtype=float)
    details = {}
    for title, chunks in grouped.items():
        indices = np.concatenate(chunks)
        panel_input = preprocess_descriptor_features(
            descriptors[indices], mode="standardize-pca", pca_components=pca_components,
            random_state=random_state,
        )
        effective_perplexity = min(perplexity, len(indices) - 1)
        embedding[indices] = compute_tsne_embedding(
            panel_input, random_state=random_state, perplexity=effective_perplexity,
            early_exaggeration=early_exaggeration, learning_rate=learning_rate,
        )
        details[title.replace("$_2$", "2")] = {
            "points": int(len(indices)),
            "input_shape": list(map(int, panel_input.shape)),
            "perplexity": int(effective_perplexity),
        }
    return embedding, details


def choose_label_offset(
    point_px: tuple[float, float],
    text: str,
    occupied: list[tuple[float, float, float, float]],
    axes_bounds: tuple[float, float, float, float],
    dpi: float,
    font_size: float = 5.6,
) -> tuple[float, float, tuple[float, float, float, float]]:
    """Choose a deterministic nearby label position that avoids prior labels."""

    offsets = [(0.0, 7.0)]
    for radius in range(10, 121, 6):
        n_angles = max(12, int(2 * np.pi * radius / 10))
        for angle in np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_angles, endpoint=False):
            offsets.append((radius * float(np.cos(angle)), radius * float(np.sin(angle))))
    px_per_point = dpi / 72.0
    width = (len(text) * font_size * 0.62 + 4.5) * px_per_point
    height = (font_size + 4.5) * px_per_point
    ax_x0, ax_y0, ax_x1, ax_y1 = axes_bounds
    for dx, dy in offsets:
        center_x = point_px[0] + dx * px_per_point
        center_y = point_px[1] + dy * px_per_point
        box = (
            center_x - width / 2, center_y - height / 2,
            center_x + width / 2, center_y + height / 2,
        )
        if box[0] < ax_x0 or box[1] < ax_y0 or box[2] > ax_x1 or box[3] > ax_y1:
            continue
        if any(not (box[2] < other[0] or box[0] > other[2] or box[3] < other[1] or box[1] > other[3]) for other in occupied):
            continue
        return dx, dy, box
    raise ValueError(f"Unable to place label {text} without overlap")


def plot_family_panels(
    output: Path,
    embedding: np.ndarray,
    color_values: np.ndarray,
    records: list[dict[str, object]],
    frame_offsets: list[int],
    representative_local: dict[str, int],
    numbered: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.colors import Normalize

    titles = FAMILY_TITLES
    grouped = family_frame_indices(records, frame_offsets)
    panels: dict[str, list[np.ndarray]] = defaultdict(list)
    for family, indices in grouped.items():
        panels[titles[family]].append(indices)
    panel_indices = {
        title: np.concatenate(chunks) for title, chunks in panels.items()
    }
    source_to_record = {str(record["system_path"]): record for record in records}
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
    })
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 5.2), constrained_layout=True)
    color_norm = Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))
    scatter = None
    panel_specs = (
        ("a", "HON", r"N$_2$/H$_2$O"),
        ("b", "NaCl", r"N$_2$/H$_2$O/NaCl"),
        ("c", r"TiO$_2$", r"N$_2$/H$_2$O/TiO$_2$"),
        ("d", r"SiO$_2$", r"N$_2$/H$_2$O/SiO$_2$"),
    )
    label_rows = []
    for ax, (panel, title, display_title) in zip(axes.flat, panel_specs):
        indices = panel_indices[title]
        scatter = ax.scatter(
            embedding[indices, 0], embedding[indices, 1], c=color_values[indices],
            cmap="viridis", norm=color_norm, s=9, alpha=0.52,
            edgecolors="none", rasterized=True,
        )
        panel_text = ax.text(
            0.015, 0.985, f"{panel}  {display_title}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, fontweight="bold", zorder=6,
        )
        if numbered:
            panel_representatives = []
            for source, global_index in representative_local.items():
                record = source_to_record[source]
                if titles[str(record["family"])] != title:
                    continue
                panel_representatives.append((int(record["dataset_id"]), global_index))
            representative_indices = np.asarray(
                [global_index for _, global_index in panel_representatives], dtype=int
            )
            ax.scatter(
                embedding[representative_indices, 0], embedding[representative_indices, 1],
                c=color_values[representative_indices], cmap="viridis", norm=color_norm,
                s=22, alpha=0.95, edgecolors="white", linewidths=0.35, zorder=3,
            )
            figure.canvas.draw()
            title_box = panel_text.get_window_extent(figure.canvas.get_renderer()).expanded(1.05, 1.12)
            occupied: list[tuple[float, float, float, float]] = [
                tuple(map(float, title_box.extents))
            ]
            axes_bounds = tuple(map(float, ax.bbox.extents))
            for dataset_id, global_index in sorted(panel_representatives):
                point_px = tuple(map(float, ax.transData.transform(embedding[global_index])))
                dx, dy, box = choose_label_offset(
                    point_px, str(dataset_id), occupied, axes_bounds, figure.dpi
                )
                occupied.append(box)
                annotation = ax.annotate(
                    str(dataset_id), xy=embedding[global_index], xytext=(dx, dy),
                    textcoords="offset points", ha="center", va="center", fontsize=5.6,
                    color="#6f1515", fontweight="bold", zorder=5,
                    bbox={
                        "boxstyle": "circle,pad=0.16", "facecolor": (1, 1, 1, 0.52),
                        "edgecolor": (0.43, 0.08, 0.08, 0.62), "linewidth": 0.4,
                    },
                    arrowprops={
                        "arrowstyle": "-", "color": (0.43, 0.08, 0.08, 0.27),
                        "linewidth": 0.28, "shrinkA": 3, "shrinkB": 2,
                    },
                )
                annotation.get_bbox_patch().set_path_effects([
                    path_effects.SimplePatchShadow(
                        offset=(0.6, -0.6), shadow_rgbFace=(0, 0, 0), alpha=0.16
                    ),
                    path_effects.Normal(),
                ])
                label_rows.append({
                    "dataset_id": dataset_id, "panel": panel, "family": title,
                    "embedding_1": float(embedding[global_index, 0]),
                    "embedding_2": float(embedding[global_index, 1]),
                    "offset_x_points": dx, "offset_y_points": dy,
                })
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    assert scatter is not None
    colorbar = figure.colorbar(
        scatter, ax=axes, orientation="horizontal", fraction=0.035, pad=0.008,
        aspect=30, shrink=0.48,
    )
    colorbar.set_label(r"Reference energy (eV atom$^{-1}$)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8, length=2.5, width=0.7)
    suffix = "_num" if numbered else ""
    figure.savefig(output / f"sketch_map_family_panels{suffix}.png", dpi=600, bbox_inches="tight")
    figure.savefig(output / f"sketch_map_family_panels{suffix}.pdf", bbox_inches="tight")
    if numbered:
        with (output / "family_panel_label_positions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
            writer.writeheader()
            writer.writerows(label_rows)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--table-s4-tex", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path)
    parser.add_argument("--frames-per-system", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--perplexity", type=int, default=50)
    parser.add_argument(
        "--embedding-method", choices=("tsne", "pca", "family-tsne"), default="tsne"
    )
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--early-exaggeration", type=float, default=12.0)
    parser.add_argument("--learning-rate", type=float, default=200.0)
    parser.add_argument("--descriptor-batch-size", type=int, default=1)
    parser.add_argument("--unique-compositions-dft-preferred", action="store_true")
    parser.add_argument("--descriptor-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    if args.descriptor_only and args.postprocess_only:
        raise ValueError("descriptor-only and postprocess-only are mutually exclusive")
    if args.frames_per_system <= 0:
        raise ValueError("frames-per-system must be positive")
    if args.descriptor_batch_size <= 0:
        raise ValueError("descriptor-batch-size must be positive")
    if args.postprocess_only:
        source_output = args.source_output or args.output
        if source_output != args.output:
            args.output.mkdir(parents=True, exist_ok=False)
        bundle, records, selected_local_indices = load_saved_bundle(source_output)
        table_rows = json.loads((source_output / "table_s4_rows.json").read_text())
        descriptors = np.load(source_output / "sampled_descriptors.npy")
        descriptor_stage = json.loads((source_output / "descriptor_stage.json").read_text())
        descriptor_raw_shape = descriptor_stage["descriptor_raw_shape_first_system"]
        frame_offsets = np.cumsum([0] + [len(coords) for coords in bundle.coords_list]).tolist()
    else:
        if args.training_manifest is None or args.model is None or args.table_s4_tex is None:
            raise ValueError(
                "training-manifest, model, and table-s4-tex are required for descriptor evaluation"
            )
        args.output.mkdir(parents=True, exist_ok=False)
        manifest_rows = read_manifest(args.training_manifest)
        ordered_keys, raw_table_rows = read_table_s4(args.table_s4_tex)
        rows = manifest_rows
        if args.unique_compositions_dft_preferred:
            rows = select_unique_compositions(rows, ordered_keys=ordered_keys)
        table_rows = enrich_table_rows(raw_table_rows, manifest_rows, rows)
        (args.output / "table_s4_rows.json").write_text(
            json.dumps(table_rows, indent=2) + "\n"
        )
        bundle, records, selected_local_indices = load_sampled_bundle(rows, args.frames_per_system)
        descriptors, frame_offsets, descriptor_raw_shape = evaluate_descriptors(
            bundle, args.model, args.descriptor_batch_size
        )
        save_sampled_bundle(args.output, bundle, records, selected_local_indices)
        descriptor_stage = {
            "status": "PASS",
            "stage": "descriptor",
            "model": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "training_manifest": str(args.training_manifest.resolve()),
            "training_manifest_sha256": sha256(args.training_manifest),
            "table_s4_tex": str(args.table_s4_tex.resolve()),
            "table_s4_tex_sha256": sha256(args.table_s4_tex),
            "descriptor_raw_shape_first_system": descriptor_raw_shape,
        }
        (args.output / "descriptor_stage.json").write_text(json.dumps(descriptor_stage, indent=2) + "\n")
    expected_max = len(records) * args.frames_per_system
    if not len(records) <= bundle.n_frames <= expected_max:
        raise ValueError(
            f"Unexpected sampled frame count: {bundle.n_frames}; expected {len(records)}..{expected_max}"
        )
    np.save(args.output / "sampled_descriptors.npy", descriptors)
    if args.descriptor_only:
        print(json.dumps(descriptor_stage, indent=2))
        return
    normalized_descriptors = (
        l2_normalize_descriptors(descriptors) if args.l2_normalize else descriptors
    )
    family_embedding_details = None
    if args.embedding_method == "pca":
        embedding_input = preprocess_descriptor_features(
            normalized_descriptors, mode="standardize", pca_components=args.pca_components,
            random_state=args.random_state,
        )
        embedding = compute_pca_embedding(embedding_input, args.random_state)
    elif args.embedding_method == "tsne":
        embedding_input = preprocess_descriptor_features(
            normalized_descriptors, mode="standardize-pca",
            pca_components=args.pca_components, random_state=args.random_state,
        )
        embedding = compute_tsne_embedding(
            embedding_input, random_state=args.random_state, perplexity=args.perplexity,
            early_exaggeration=args.early_exaggeration, learning_rate=args.learning_rate,
        )
    else:
        embedding_input = normalized_descriptors
        embedding, family_embedding_details = compute_family_tsne_embedding(
            normalized_descriptors, records, frame_offsets, args.random_state,
            args.perplexity, args.pca_components, args.early_exaggeration,
            args.learning_rate,
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite t-SNE coordinates")

    color_values = energy_per_atom(bundle)
    source_to_number = {
        str(record["system_path"]): int(record["dataset_id"]) for record in records
    }
    representative_local = select_representative_per_config(embedding, bundle.source_paths)
    if len(representative_local) != len(records):
        raise ValueError(
            f"Expected {len(records)} representative subsystems, found {len(representative_local)}"
        )

    with (args.output / "dataset_id_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_catalogs(args.output, records, table_rows)
    write_points(
        args.output, embedding, color_values, bundle, records, selected_local_indices, frame_offsets
    )
    if args.embedding_method == "tsne":
        (args.output / "tsne_points.csv").write_bytes(
            (args.output / "embedding_points.csv").read_bytes()
        )
    export_structures(bundle, records, representative_local, args.output)
    plot_numbered(
        args.output, embedding, color_values, bundle.source_paths, source_to_number, representative_local
    )
    plot_plain(args.output, embedding, color_values)
    plot_family_panels(
        args.output, embedding, color_values, records, frame_offsets, representative_local,
        numbered=False,
    )
    plot_family_panels(
        args.output, embedding, color_values, records, frame_offsets, representative_local,
        numbered=True,
    )
    (args.output / "sketch_map_dataset_num.png").write_bytes(
        (args.output / "sketch_map_num.png").read_bytes()
    )

    metadata = {
        "status": "PASS",
        "model": descriptor_stage["model"],
        "model_sha256": descriptor_stage["model_sha256"],
        "training_manifest": descriptor_stage["training_manifest"],
        "training_manifest_sha256": descriptor_stage["training_manifest_sha256"],
        "table_s4_tex": descriptor_stage["table_s4_tex"],
        "table_s4_tex_sha256": descriptor_stage["table_s4_tex_sha256"],
        "table_s4_rows": len(table_rows),
        "systems": len(records),
        "frames_per_system_requested": args.frames_per_system,
        "sampled_frames": bundle.n_frames,
        "descriptor_raw_shape_first_system": descriptor_raw_shape,
        "descriptor_matrix_shape": list(map(int, descriptors.shape)),
        "embedding_input_shape": list(map(int, embedding_input.shape)),
        "embedding_method": args.embedding_method,
        "family_embedding_details": family_embedding_details,
        "l2_normalize": args.l2_normalize,
        "random_state": args.random_state,
        "perplexity": args.perplexity,
        "preprocess": (
            f"{'l2-' if args.l2_normalize else ''}standardize"
            + (
                f"-per-family-pca-{args.pca_components}"
                if args.embedding_method == "family-tsne"
                else (f"-pca-{args.pca_components}" if args.embedding_method == "tsne" else "")
            )
        ),
        "color_definition": "reference_energy_eV/N_atoms",
        "representative_definition": "sampled frame nearest each subsystem embedding centroid",
        "sampling_definition": (
            f"up to {args.frames_per_system} uniformly spaced source frames per subsystem"
        ),
        "family_panel_definition": (
            "2x2 N2/H2O, N2/H2O/NaCl, N2/H2O/TiO2, and N2/H2O/SiO2 panels "
            "with independent axis limits and a shared reference-energy color scale"
        ),
        "number_label_definition": (
            "one representative frame per subsystem; transparent circular labels with "
            "deterministic collision-aware offsets"
        ),
        "selection_definition": (
            "Table S4 first-occurrence order; repeated compositions reuse the same ID; "
            "SiO2 distilled represented by SCAN DFT"
        ),
        "allowed_input_elements": sorted(ALLOWED_ELEMENTS),
        "descriptor_evaluation": "DPA4C compact Vesin edge graph; atom-mean local descriptor",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
