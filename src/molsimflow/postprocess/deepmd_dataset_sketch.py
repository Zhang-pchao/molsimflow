"""DeepMD training-set descriptor sketch-map workflow.

This module analyzes one or more DeepMD ``npy`` datasets with a frozen Deep
Potential graph. It evaluates per-frame mean descriptor vectors, downsamples
large composition groups, embeds the descriptors with PCA or t-SNE, and writes
reproducible visualization artifacts.

Runtime dependencies such as ``deepmd-kit``, ``dpdata``, ``ase``,
``matplotlib``, and ``scikit-learn`` are imported lazily so that the reusable
package remains importable in lightweight test environments.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PERIODIC_DISPLAY_ORDER: Tuple[str, ...] = (
    "H",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Ti",
)

ELEMENT_COLORS: Mapping[str, str] = {
    "H": "#FFFFFF",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "Na": "#AB5CF2",
    "Cl": "#1FF01F",
    "Ti": "#BFC2C7",
    "Si": "#F0C8A0",
}

ELEMENT_RADII: Mapping[str, float] = {
    "H": 0.20,
    "C": 0.45,
    "N": 0.40,
    "O": 0.38,
    "Na": 0.55,
    "Cl": 0.60,
    "Ti": 0.50,
    "Si": 0.50,
}


@dataclass(frozen=True)
class DeepmdDatasetSketchConfig:
    """Configuration for descriptor-based DeepMD dataset visualization."""

    dataset_roots: Tuple[Path, ...]
    model: Path
    output_dir: Path
    batch_size: int = 100
    sample_count: int = 4
    random_state: int = 0
    overwrite: bool = False
    gpu: str = "0"
    method: str = "tsne"
    perplexity: int = 30
    tsne_early_exaggeration: float = 12.0
    tsne_learning_rate: Optional[float] = None
    descriptor_preprocess: str = "none"
    descriptor_pca_components: int = 50
    max_per_config: int = 200
    write_structure_images: bool = True
    write_latex_table: bool = True


@dataclass(frozen=True)
class DeepmdDatasetSketchResult:
    """Paths and counts produced by a sketch-map run."""

    output_dir: Path
    dataset_count: int
    n_frames_total: int
    n_frames_after_downsample: int
    config_count: int
    csv_path: Path
    metadata_path: Path
    figure_path: Path
    numbered_figure_path: Path
    dataset_numbered_figure_path: Path


@dataclass
class DatasetBundle:
    """Arrays and metadata loaded from one or more DeepMD datasets."""

    coords_list: List[np.ndarray]
    cells_list: List[np.ndarray]
    energies: np.ndarray
    n_atoms_list: List[int]
    atom_types_list: List[List[int]]
    type_symbols_list: List[List[str]]
    system_configs: List[str]
    source_paths: List[str]
    frame_to_dataset: List[int]

    @property
    def n_frames(self) -> int:
        """Total number of loaded frames."""

        return int(len(self.energies))

    def get_frame_data(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
        """Return coordinates, cell, atom type indices, and symbols for one global frame."""

        dataset_idx = self.frame_to_dataset[frame_idx]
        frame_start = sum(len(self.coords_list[i]) for i in range(dataset_idx))
        local_idx = frame_idx - frame_start
        return (
            self.coords_list[dataset_idx][local_idx],
            self.cells_list[dataset_idx][local_idx],
            self.atom_types_list[dataset_idx],
            self.type_symbols_list[dataset_idx],
        )


def _import_dpdata() -> Any:
    try:
        import dpdata
    except ImportError as exc:  # pragma: no cover - depends on cluster env
        raise RuntimeError("DeepMD dataset sketching requires dpdata") from exc
    return dpdata


def _import_deep_pot() -> Any:
    try:
        from deepmd.infer.deep_pot import DeepPot
    except ImportError as exc:  # pragma: no cover - depends on cluster env
        raise RuntimeError("DeepMD dataset sketching requires deepmd-kit") from exc
    return DeepPot


def _import_atoms_writer() -> Tuple[Any, Any]:
    try:
        from ase import Atoms
        from ase.io import write
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Structure export requires ase") from exc
    return Atoms, write


def _import_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Plotting requires matplotlib") from exc
    return plt


def configure_gpu_environment(gpu_id: str = "0") -> None:
    """Set GPU/threading environment variables before TensorFlow loads the graph."""

    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        logging.info("Set CUDA_VISIBLE_DEVICES to %s", gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logging.info("Disabled GPU by setting CUDA_VISIBLE_DEVICES to an empty string")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_INTRA_OP_PARALLELISM_THREADS", "1")
    os.environ.setdefault("TF_INTER_OP_PARALLELISM_THREADS", "1")

    try:  # pragma: no cover - TensorFlow availability is environment-specific
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            logging.warning("TensorFlow did not report any GPU devices")
            return
        logging.info("TensorFlow detected %d GPU device(s)", len(gpus))
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                logging.warning("Could not enable TensorFlow memory growth for %s: %s", gpu, exc)
    except ImportError:
        logging.warning("TensorFlow is unavailable for GPU probing")
    except Exception as exc:
        logging.warning("TensorFlow GPU probing failed: %s", exc)


def get_system_config_name(system: Any) -> str:
    """Build a compact composition label such as ``128H64O6N`` from dpdata metadata."""

    atom_names = list(system.get_atom_names())
    atom_numbs = list(system.get_atom_numbs())
    count_by_element = {name: int(count) for name, count in zip(atom_names, atom_numbs)}
    ordered: List[Tuple[int, str]] = []

    for element in PERIODIC_DISPLAY_ORDER:
        count = count_by_element.get(element, 0)
        if count > 0:
            ordered.append((count, element))

    for element in sorted(count_by_element):
        if element in PERIODIC_DISPLAY_ORDER:
            continue
        count = count_by_element[element]
        if count > 0:
            ordered.append((count, element))

    return "".join(f"{count}{element}" for count, element in ordered)


def find_deepmd_datasets(root_paths: Sequence[Path]) -> List[Path]:
    """Find DeepMD ``npy`` dataset directories below the provided roots."""

    dataset_dirs: List[Path] = []
    seen: set[Path] = set()

    for root in root_paths:
        root_path = Path(root)
        if not root_path.exists():
            logging.warning("Dataset search root does not exist: %s", root_path)
            continue

        candidates = [root_path]
        candidates.extend(path for path in root_path.rglob("*") if path.is_dir())
        for candidate in candidates:
            has_set_dir = any(
                child.is_dir() and child.name.startswith("set.") for child in candidate.iterdir()
            )
            if not has_set_dir:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            dataset_dirs.append(candidate)
            logging.info("Found DeepMD dataset: %s", candidate)

    return dataset_dirs


def load_multiple_datasets(dataset_paths: Sequence[Path]) -> DatasetBundle:
    """Load and concatenate DeepMD datasets that may have different atom counts."""

    if not dataset_paths:
        raise ValueError("No DeepMD dataset paths were provided")

    dpdata = _import_dpdata()
    coords_list: List[np.ndarray] = []
    cells_list: List[np.ndarray] = []
    energies_list: List[np.ndarray] = []
    n_atoms_list: List[int] = []
    atom_types_list: List[List[int]] = []
    type_symbols_list: List[List[str]] = []
    system_configs: List[str] = []
    source_paths: List[str] = []
    frame_to_dataset: List[int] = []

    for dataset_idx, path in enumerate(dataset_paths):
        try:
            system = dpdata.LabeledSystem(str(path), fmt="deepmd/npy")
            coords = np.asarray(system.data["coords"], dtype=np.float64)
            cells = np.asarray(system.data["cells"], dtype=np.float64)
            energies = np.asarray(system.data["energies"], dtype=np.float64)
            atom_types = list(map(int, system.get_atom_types()))
            type_symbols = list(system.get_atom_names())
        except Exception as exc:
            logging.error("Failed to load DeepMD dataset %s: %s", path, exc)
            continue

        loaded_idx = len(coords_list)
        config_name = get_system_config_name(system)
        n_frames = int(coords.shape[0])
        n_atoms = int(coords.shape[1])

        coords_list.append(coords)
        cells_list.append(cells)
        energies_list.append(energies)
        n_atoms_list.extend([n_atoms] * n_frames)
        atom_types_list.append(atom_types)
        type_symbols_list.append(type_symbols)
        system_configs.extend([config_name] * n_frames)
        source_paths.extend([str(path)] * n_frames)
        frame_to_dataset.extend([loaded_idx] * n_frames)

        logging.info(
            "Loaded %d frames from %s (%s, %d atoms)", n_frames, path, config_name, n_atoms
        )

    if not energies_list:
        raise ValueError("No DeepMD datasets were successfully loaded")

    return DatasetBundle(
        coords_list=coords_list,
        cells_list=cells_list,
        energies=np.concatenate(energies_list, axis=0),
        n_atoms_list=n_atoms_list,
        atom_types_list=atom_types_list,
        type_symbols_list=type_symbols_list,
        system_configs=system_configs,
        source_paths=source_paths,
        frame_to_dataset=frame_to_dataset,
    )


def build_model_type_indices(
    type_symbols: Sequence[str], atom_types: Sequence[int], model: Any
) -> List[int]:
    """Map dataset atom-type indices to the type ordering expected by a DeepPot model."""

    model_type_map = list(model.get_type_map())
    missing = sorted(set(type_symbols) - set(model_type_map))
    if missing:
        raise ValueError(f"Dataset elements are missing from model type map: {missing}")
    index_lookup = {symbol: model_type_map.index(symbol) for symbol in type_symbols}
    return [index_lookup[type_symbols[idx]] for idx in atom_types]


def batched(total_frames: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    """Yield ``(start, stop)`` frame ranges."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, total_frames, batch_size):
        yield start, min(total_frames, start + batch_size)


def evaluate_descriptors(dataset: DatasetBundle, model: Any, batch_size: int) -> np.ndarray:
    """Evaluate mean descriptor vectors for all frames in all loaded datasets."""

    descriptor_sets: List[np.ndarray] = []
    for dataset_idx, (coords, cells, atom_types, type_symbols) in enumerate(
        zip(
            dataset.coords_list,
            dataset.cells_list,
            dataset.atom_types_list,
            dataset.type_symbols_list,
        )
    ):
        mapped_types = build_model_type_indices(type_symbols, atom_types, model)
        n_frames = int(coords.shape[0])
        logging.info(
            "Evaluating dataset %d/%d with %d frames",
            dataset_idx + 1,
            len(dataset.coords_list),
            n_frames,
        )
        mean_batches: List[np.ndarray] = []

        for start, stop in batched(n_frames, batch_size):
            cells_batch = cells[start:stop]
            if cells_batch.ndim != 3 or cells_batch.shape[-2:] != (3, 3):
                cells_batch = cells_batch.reshape(-1, 3, 3)
            descriptor = model.eval_descriptor(coords[start:stop], cells_batch, mapped_types)
            mean_batches.append(np.asarray(descriptor).mean(axis=1))

        descriptors = np.concatenate(mean_batches, axis=0)
        descriptor_sets.append(descriptors)
        logging.info("Descriptor shape for dataset %d: %s", dataset_idx + 1, descriptors.shape)

    return np.concatenate(descriptor_sets, axis=0)


def compute_pca_embedding(data: np.ndarray, random_state: int) -> np.ndarray:
    """Reduce descriptor features to two PCA components."""

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("PCA embedding requires scikit-learn") from exc
    return PCA(n_components=2, random_state=random_state).fit_transform(data)


def preprocess_descriptor_features(
    data: np.ndarray,
    mode: str,
    pca_components: int,
    random_state: int,
) -> np.ndarray:
    """Optionally scale and PCA-compress descriptor features before embedding."""

    if mode == "none":
        return data
    if mode not in {"standardize", "pca", "standardize-pca"}:
        raise ValueError(
            "descriptor_preprocess must be one of: none, standardize, pca, standardize-pca"
        )
    if pca_components <= 0:
        raise ValueError("descriptor_pca_components must be positive")

    transformed = data
    if mode in {"standardize", "standardize-pca"}:
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Descriptor standardization requires scikit-learn") from exc
        transformed = StandardScaler().fit_transform(transformed)

    if mode in {"pca", "standardize-pca"}:
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Descriptor PCA preprocessing requires scikit-learn") from exc
        n_components = min(pca_components, transformed.shape[0] - 1, transformed.shape[1])
        transformed = PCA(n_components=n_components, random_state=random_state).fit_transform(
            transformed
        )

    return transformed


def compute_tsne_embedding(
    data: np.ndarray,
    random_state: int,
    perplexity: int = 30,
    early_exaggeration: float = 12.0,
    learning_rate: Optional[float] = None,
) -> np.ndarray:
    """Reduce descriptor features to two t-SNE components."""

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("t-SNE embedding requires scikit-learn") from exc

    kwargs: Dict[str, Any] = {
        "n_components": 2,
        "random_state": random_state,
        "perplexity": perplexity,
        "early_exaggeration": early_exaggeration,
        "verbose": 1,
    }
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    tsne_signature = inspect.signature(TSNE)
    if "max_iter" in tsne_signature.parameters:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    return TSNE(**kwargs).fit_transform(data)


def downsample_by_config(
    indices: np.ndarray,
    system_configs: Sequence[str],
    max_per_config: int = 200,
) -> np.ndarray:
    """Uniformly cap each system-configuration group to ``max_per_config`` frames."""

    if max_per_config <= 0:
        raise ValueError("max_per_config must be positive")

    config_to_indices: Dict[str, List[int]] = {}
    for idx in indices:
        config_to_indices.setdefault(system_configs[int(idx)], []).append(int(idx))

    selected: List[int] = []
    for config, config_indices in config_to_indices.items():
        if len(config_indices) <= max_per_config:
            selected.extend(config_indices)
            continue
        step = len(config_indices) / max_per_config
        sampled = [config_indices[int(i * step)] for i in range(max_per_config)]
        selected.extend(sampled)
        logging.info(
            "Downsampled %s from %d to %d frames", config, len(config_indices), len(sampled)
        )

    return np.asarray(selected, dtype=int)


def select_representative_per_config(
    embedding: np.ndarray,
    system_configs: Sequence[str],
) -> Dict[str, int]:
    """Select the frame nearest each configuration centroid in embedding space."""

    config_to_indices: Dict[str, List[int]] = {}
    for idx, config in enumerate(system_configs):
        config_to_indices.setdefault(config, []).append(idx)

    representatives: Dict[str, int] = {}
    for config, indices in config_to_indices.items():
        if len(indices) == 1:
            representatives[config] = indices[0]
            continue
        config_embedding = embedding[indices]
        centroid = config_embedding.mean(axis=0)
        distances = np.linalg.norm(config_embedding - centroid, axis=1)
        representatives[config] = int(indices[int(np.argmin(distances))])
    return representatives


def select_representative_frames(embedding: np.ndarray, count: int) -> List[Tuple[str, int]]:
    """Pick representative local frame indices from extreme embedding regions."""

    if count <= 0:
        return []
    selection = [
        ("pc1_min", int(np.argmin(embedding[:, 0]))),
        ("pc1_max", int(np.argmax(embedding[:, 0]))),
        ("pc2_min", int(np.argmin(embedding[:, 1]))),
        ("pc2_max", int(np.argmax(embedding[:, 1]))),
    ]
    remaining = max(count - len(selection), 0)
    if remaining:
        for idx, frame in enumerate(np.linspace(0, embedding.shape[0] - 1, remaining, dtype=int)):
            selection.append((f"extra{idx}", int(frame)))

    seen: set[int] = set()
    unique: List[Tuple[str, int]] = []
    for label, frame_idx in selection:
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        unique.append((label, frame_idx))
    return unique[:count]


def _frame_atoms(dataset: DatasetBundle, frame_idx: int) -> Any:
    Atoms, _write = _import_atoms_writer()
    coords, cell, atom_types, type_symbols = dataset.get_frame_data(frame_idx)
    symbols = [type_symbols[idx] for idx in atom_types]
    return Atoms(symbols=symbols, positions=coords, cell=cell, pbc=True)


def export_samples(
    dataset: DatasetBundle,
    frame_indices: Sequence[Tuple[str, int]],
    destination: Path,
) -> None:
    """Write representative frames to XYZ files."""

    _Atoms, write = _import_atoms_writer()
    destination.mkdir(parents=True, exist_ok=True)
    for label, frame_idx in frame_indices:
        file_path = destination / f"sample_{label}_{frame_idx:05d}.xyz"
        write(file_path, _frame_atoms(dataset, frame_idx))
        logging.info("Exported sample structure %s", file_path)


def export_config_representatives(
    dataset: DatasetBundle,
    config_to_frame: Mapping[str, int],
    destination: Path,
) -> None:
    """Export one representative extended XYZ per system configuration."""

    _Atoms, write = _import_atoms_writer()
    destination.mkdir(parents=True, exist_ok=True)
    for config_name, frame_idx in config_to_frame.items():
        atoms = _frame_atoms(dataset, frame_idx)
        atoms.info["config"] = config_name
        atoms.info["frame_index"] = int(frame_idx)
        atoms.info["source"] = dataset.source_paths[frame_idx]
        file_path = destination / f"representative_{config_name}.xyz"
        write(file_path, atoms, format="extxyz")
        logging.info("Exported representative structure %s", file_path)


def export_numbered_dataset_representatives(
    dataset: DatasetBundle,
    source_to_frame: Mapping[str, int],
    source_to_number: Mapping[str, int],
    destination: Path,
) -> None:
    """Export one representative extended XYZ per numbered source dataset."""

    _Atoms, write = _import_atoms_writer()
    destination.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        source_to_frame.items(),
        key=lambda item: int(source_to_number[item[0]]),
    )
    for source_path, frame_idx in rows:
        number = int(source_to_number[source_path])
        atoms = _frame_atoms(dataset, frame_idx)
        atoms.info["dataset_number"] = number
        atoms.info["dataset_path"] = source_path
        atoms.info["config"] = dataset.system_configs[frame_idx]
        atoms.info["frame_index"] = int(frame_idx)
        file_path = destination / f"representative_dataset_{number:03d}.xyz"
        write(file_path, atoms, format="extxyz")
        logging.info("Exported dataset representative structure %s", file_path)


def save_metadata(output_dir: Path, metadata: Mapping[str, object]) -> Path:
    """Write run metadata as JSON and return the output path."""

    metadata_path = output_dir / "descriptor_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def find_annotation_positions(
    embedding: np.ndarray,
    config_to_frame: Mapping[str, int],
) -> Dict[str, Tuple[float, float, str]]:
    """Choose simple non-overlapping label positions around representative points."""

    positions: Dict[str, Tuple[float, float, str]] = {}
    occupied: List[Tuple[float, float]] = []
    x_range = float(np.ptp(embedding[:, 0])) or 1.0
    y_range = float(np.ptp(embedding[:, 1])) or 1.0

    for config, frame_idx in config_to_frame.items():
        point_x, point_y = embedding[frame_idx]
        offsets = [
            (0.05 * x_range, 0.05 * y_range, "left"),
            (-0.05 * x_range, 0.05 * y_range, "right"),
            (0.05 * x_range, -0.05 * y_range, "left"),
            (-0.05 * x_range, -0.05 * y_range, "right"),
            (0.08 * x_range, 0.0, "left"),
            (-0.08 * x_range, 0.0, "right"),
        ]
        best_pos: Optional[Tuple[float, float, str]] = None
        best_dist = -1.0
        for dx, dy, align in offsets:
            text_x = float(point_x + dx)
            text_y = float(point_y + dy)
            min_dist = min(
                (float(np.hypot(text_x - occ_x, text_y - occ_y)) for occ_x, occ_y in occupied),
                default=float("inf"),
            )
            if min_dist > best_dist:
                best_dist = min_dist
                best_pos = (text_x, text_y, align)
        if best_pos is not None:
            positions[config] = best_pos
            occupied.append((best_pos[0], best_pos[1]))
    return positions


def render_structure_image(
    atoms: Any,
    output_path: Path,
    size: Tuple[int, int] = (300, 300),
    camera_pos: str = "xy",
) -> None:
    """Render an atomistic structure as a compact matplotlib 3D PNG."""

    plt = _import_pyplot()
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    centered = positions - positions.mean(axis=0)

    for pos, symbol in zip(centered, symbols):
        color = ELEMENT_COLORS.get(symbol, "#CCCCCC")
        radius = ELEMENT_RADII.get(symbol, 0.40)
        ax.scatter(
            pos[0],
            pos[1],
            pos[2],
            c=color,
            s=radius * 200,
            alpha=0.95,
            edgecolors="black",
            linewidth=0.3,
        )

    if camera_pos == "xy":
        ax.view_init(elev=20, azim=45)
    elif camera_pos == "xz":
        ax.view_init(elev=60, azim=45)
    else:
        ax.view_init(elev=30, azim=30)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    spans = np.ptp(centered, axis=0)
    max_range = float(max(spans.max() / 2.0, 1.0))
    mid_x, mid_y, mid_z = (centered.max(axis=0) + centered.min(axis=0)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="none")
    plt.close(fig)


def _composition_elements(config: str) -> frozenset[str]:
    return frozenset(match.group(2) for match in re.finditer(r"(\d+)([A-Z][a-z]?)", config))


def group_configs_by_chemistry(configs: Sequence[str]) -> Dict[str, int]:
    """Assign stable numeric labels grouped by element-set chemistry."""

    element_groups: Dict[Tuple[str, ...], List[str]] = {}
    for config in configs:
        key = tuple(sorted(_composition_elements(config)))
        element_groups.setdefault(key, []).append(config)

    config_to_number: Dict[str, int] = {}
    number = 1
    for key in sorted(element_groups):
        for config in sorted(element_groups[key]):
            config_to_number[config] = number
            number += 1
    return config_to_number


def create_sketch_map_with_numbers(
    embedding: np.ndarray,
    energies_per_atom: np.ndarray,
    config_to_frame: Mapping[str, int],
    output_path: Path,
    figsize: Tuple[float, float] = (12.0, 6.0),
    label_to_number: Optional[Mapping[str, int]] = None,
    mapping_filename: str = "number_to_config_mapping.json",
) -> Dict[int, str]:
    """Create a numbered sketch-map PNG and return ``number -> label`` values."""

    plt = _import_pyplot()
    if label_to_number is None:
        label_to_number = group_configs_by_chemistry(list(config_to_frame.keys()))
    missing_labels = sorted(set(config_to_frame) - set(label_to_number))
    if missing_labels:
        raise ValueError(f"Missing numeric labels for: {missing_labels}")

    active_label_to_number = {
        label: int(label_to_number[label]) for label in config_to_frame
    }
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=energies_per_atom,
        cmap="viridis",
        s=24,
        alpha=0.6,
        edgecolors="none",
    )
    annotation_positions = find_annotation_positions(embedding, config_to_frame)
    for config, frame_idx in config_to_frame.items():
        point_x, point_y = embedding[frame_idx]
        number = active_label_to_number[config]
        ax.scatter(
            [point_x],
            [point_y],
            s=38,
            facecolors="none",
            edgecolors="red",
            linewidths=1.4,
        )
        text_x, text_y, _align = annotation_positions.get(
            config,
            (float(point_x), float(point_y), "center"),
        )
        if text_x != float(point_x) or text_y != float(point_y):
            ax.plot(
                [point_x, text_x],
                [point_y, text_y],
                color="red",
                linewidth=0.6,
                alpha=0.65,
                zorder=3,
            )
        ax.text(
            text_x,
            text_y,
            str(number),
            fontsize=6.5,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            bbox={"boxstyle": "circle,pad=0.18", "facecolor": "red", "alpha": 0.85},
            zorder=4,
        )

    cbar = plt.colorbar(
        scatter, ax=ax, orientation="horizontal", pad=0.05, fraction=0.03, aspect=15
    )
    cbar.set_label("E/atom (eV)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_position([0.08, 0.04, 0.27, 0.015])
    _hide_axes(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, facecolor="none", bbox_inches="tight")
    plt.close(fig)

    number_to_config = {
        number: config for config, number in active_label_to_number.items()
    }
    mapping_path = output_path.parent / mapping_filename
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {str(k): v for k, v in sorted(number_to_config.items())},
            handle,
            indent=2,
        )
    logging.info("Saved numbered sketch map to %s", output_path)
    return number_to_config


def create_sketch_map_with_labels(
    embedding: np.ndarray,
    energies_per_atom: np.ndarray,
    config_to_frame: Mapping[str, int],
    output_path: Path,
    figsize: Tuple[float, float] = (12.0, 8.0),
) -> None:
    """Create a sketch-map PNG with composition labels and arrows."""

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=energies_per_atom,
        cmap="viridis",
        s=30,
        alpha=0.6,
        edgecolors="none",
    )
    annotation_positions = find_annotation_positions(embedding, config_to_frame)
    for config, frame_idx in config_to_frame.items():
        point_x, point_y = embedding[frame_idx]
        if config not in annotation_positions:
            continue
        text_x, text_y, align = annotation_positions[config]
        ax.scatter([point_x], [point_y], s=100, facecolors="none", edgecolors="red", linewidths=2)
        ax.annotate(
            config,
            xy=(point_x, point_y),
            xytext=(text_x, text_y),
            fontsize=12,
            ha=align,
            va="center",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
            arrowprops={
                "arrowstyle": "->",
                "connectionstyle": "arc3,rad=0.3",
                "color": "red",
                "lw": 2,
            },
        )

    cbar = plt.colorbar(
        scatter, ax=ax, orientation="horizontal", pad=0.05, fraction=0.03, aspect=15
    )
    cbar.set_label("E/atom (eV)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_position([0.08, 0.04, 0.27, 0.015])
    _hide_axes(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="none", bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved labeled sketch map to %s", output_path)


def _hide_axes(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def write_embedding_csv(
    path: Path,
    embedding: np.ndarray,
    energies_per_atom: np.ndarray,
    selected_indices: np.ndarray,
    system_configs: Sequence[str],
    source_paths: Optional[Sequence[str]] = None,
    dataset_numbers: Optional[Mapping[str, int]] = None,
) -> None:
    """Write embedding coordinates and frame metadata to CSV."""

    if source_paths is not None and len(source_paths) != len(system_configs):
        raise ValueError("source_paths must have the same length as system_configs")
    if dataset_numbers is not None and source_paths is None:
        raise ValueError("dataset_numbers requires source_paths")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pc1",
        "pc2",
        "energy_per_atom",
        "original_frame_index",
        "system_config",
    ]
    if source_paths is not None:
        fieldnames.append("source_path")
    if dataset_numbers is not None:
        fieldnames.append("dataset_number")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, config in enumerate(system_configs):
            row = {
                "pc1": float(embedding[row_idx, 0]),
                "pc2": float(embedding[row_idx, 1]),
                "energy_per_atom": float(energies_per_atom[row_idx]),
                "original_frame_index": int(selected_indices[row_idx]),
                "system_config": config,
            }
            if source_paths is not None:
                source_path = source_paths[row_idx]
                row["source_path"] = source_path
                if dataset_numbers is not None:
                    row["dataset_number"] = int(dataset_numbers[source_path])
            writer.writerow(row)


def write_subsystem_latex(
    output_path: Path,
    number_to_config: Mapping[int, str],
    config_counts: Mapping[str, int],
) -> None:
    """Write a compact LaTeX table listing numbered subsystem labels and frame counts."""

    rows = [
        (int(number), config, int(config_counts.get(config, 0)))
        for number, config in number_to_config.items()
    ]
    rows.sort(key=lambda item: item[0])
    total_frames = sum(frame_count for _number, _config, frame_count in rows)
    split = (len(rows) + 1) // 2

    def esc(text: str) -> str:
        return text.replace("_", r"\_")

    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\usepackage{caption}",
        r"\geometry{a4paper, margin=1in}",
        r"\begin{document}",
        r"\begin{table*}[htbp]",
        r"    \renewcommand{\arraystretch}{1.2}",
        (
            f"    \\caption{{\\textbf{{The details of the data sets.}} "
            f"The total number of frames is {total_frames}.}}"
        ),
        r"    \centering",
        r"    \begin{tabular}{ccc}",
        r"        \toprule",
        r"        index & subsystem & frame \\",
        r"        \midrule",
    ]
    for number, config, frame_count in rows:
        lines.append(f"        {number} & {esc(config)} & {frame_count} \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table*}",
            r"\newpage",
            r"\begin{table*}[htbp]",
            r"    \renewcommand{\arraystretch}{1.2}",
            r"    \caption{\textbf{The details of the data sets (Two-Column Layout).}}",
            r"    \centering",
            r"    \begin{tabular}{ccc|ccc}",
            r"        \toprule",
            r"        index & subsystem & frame & index & subsystem & frame \\",
            r"        \midrule",
        ]
    )
    for offset in range(split):
        left = rows[offset]
        right = rows[offset + split] if offset + split < len(rows) else None
        left_text = f"{left[0]} & {esc(left[1])} & {left[2]}"
        if right is None:
            right_text = " &  & "
        else:
            right_text = f"{right[0]} & {esc(right[1])} & {right[2]}"
        lines.append(f"        {left_text} & {right_text} \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table*}",
            r"\end{document}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_deepmd_dataset_sketch(config: DeepmdDatasetSketchConfig) -> DeepmdDatasetSketchResult:
    """Run the full DeepMD descriptor sketch-map workflow."""

    if config.method not in {"pca", "tsne"}:
        raise ValueError("method must be 'pca' or 'tsne'")

    configure_gpu_environment(config.gpu)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_cache = config.output_dir / "descriptor_frames.npy"

    logging.info("Searching for DeepMD datasets under %s", list(config.dataset_roots))
    dataset_paths = find_deepmd_datasets(config.dataset_roots)
    if not dataset_paths:
        raise ValueError("No DeepMD datasets found in the provided roots")

    dataset = load_multiple_datasets(dataset_paths)
    unique_configs = sorted(set(dataset.system_configs))
    config_counts = {name: dataset.system_configs.count(name) for name in unique_configs}
    logging.info(
        "Loaded %d total frames across %d configurations", dataset.n_frames, len(unique_configs)
    )

    DeepPot = _import_deep_pot()
    logging.info("Loading frozen Deep Potential model from %s", config.model)
    model = DeepPot(str(config.model))

    selected_indices = downsample_by_config(
        np.arange(dataset.n_frames),
        dataset.system_configs,
        max_per_config=config.max_per_config,
    )
    logging.info("Using %d frames after per-configuration downsampling", len(selected_indices))

    if descriptor_cache.exists() and not config.overwrite:
        logging.info("Reusing descriptor cache %s", descriptor_cache)
        mean_descriptors = np.load(descriptor_cache)
        if mean_descriptors.shape[0] != dataset.n_frames:
            raise ValueError(
                "Descriptor cache frame count does not match the loaded dataset; "
                "rerun with --overwrite"
            )
    else:
        mean_descriptors = evaluate_descriptors(dataset, model, batch_size=config.batch_size)
        np.save(descriptor_cache, mean_descriptors)
        logging.info(
            "Stored descriptor cache %s with shape %s", descriptor_cache, mean_descriptors.shape
        )

    selected_descriptors = mean_descriptors[selected_indices]
    selected_energies = dataset.energies[selected_indices]
    selected_n_atoms = np.asarray([dataset.n_atoms_list[int(idx)] for idx in selected_indices])
    selected_configs = [dataset.system_configs[int(idx)] for idx in selected_indices]
    selected_sources = [dataset.source_paths[int(idx)] for idx in selected_indices]
    energies_per_atom = selected_energies / selected_n_atoms

    embedding_input = preprocess_descriptor_features(
        selected_descriptors,
        mode=config.descriptor_preprocess,
        pca_components=config.descriptor_pca_components,
        random_state=config.random_state,
    )
    logging.info(
        "Embedding input shape after descriptor_preprocess=%s: %s",
        config.descriptor_preprocess,
        embedding_input.shape,
    )

    if config.method == "tsne":
        logging.info(
            "Computing t-SNE embedding with perplexity=%d, early_exaggeration=%s, "
            "learning_rate=%s",
            config.perplexity,
            config.tsne_early_exaggeration,
            config.tsne_learning_rate,
        )
        embedding = compute_tsne_embedding(
            embedding_input,
            config.random_state,
            config.perplexity,
            early_exaggeration=config.tsne_early_exaggeration,
            learning_rate=config.tsne_learning_rate,
        )
    else:
        logging.info("Computing PCA embedding")
        embedding = compute_pca_embedding(embedding_input, config.random_state)

    config_to_frame_local = select_representative_per_config(embedding, selected_configs)
    config_to_frame_global = {
        name: int(selected_indices[local_idx]) for name, local_idx in config_to_frame_local.items()
    }
    source_to_number = {str(path): idx + 1 for idx, path in enumerate(dataset_paths)}
    source_to_frame_local = select_representative_per_config(embedding, selected_sources)
    source_to_frame_global = {
        name: int(selected_indices[local_idx]) for name, local_idx in source_to_frame_local.items()
    }

    csv_path = config.output_dir / "sketch_map_points.csv"
    write_embedding_csv(
        csv_path,
        embedding,
        energies_per_atom,
        selected_indices,
        selected_configs,
        source_paths=selected_sources,
        dataset_numbers=source_to_number,
    )

    figure_path = config.output_dir / "sketch_map.png"
    create_sketch_map_with_labels(embedding, energies_per_atom, config_to_frame_local, figure_path)

    numbered_figure_path = config.output_dir / "sketch_map_num.png"
    number_to_config = create_sketch_map_with_numbers(
        embedding,
        energies_per_atom,
        config_to_frame_local,
        numbered_figure_path,
    )

    dataset_numbered_figure_path = config.output_dir / "sketch_map_dataset_num.png"
    number_to_dataset = create_sketch_map_with_numbers(
        embedding,
        energies_per_atom,
        source_to_frame_local,
        dataset_numbered_figure_path,
        figsize=(14.0, 7.0),
        label_to_number=source_to_number,
        mapping_filename="number_to_dataset_mapping.json",
    )

    if config.write_structure_images:
        image_dir = config.output_dir / "structure_images"
        for config_name, frame_idx in config_to_frame_global.items():
            render_structure_image(
                _frame_atoms(dataset, frame_idx), image_dir / f"structure_{config_name}.png"
            )

    export_config_representatives(
        dataset, config_to_frame_global, config.output_dir / "config_representatives"
    )
    export_numbered_dataset_representatives(
        dataset,
        source_to_frame_global,
        source_to_number,
        config.output_dir / "dataset_representatives",
    )

    local_samples = select_representative_frames(embedding, config.sample_count)
    global_samples = [
        (label, int(selected_indices[local_idx]))
        for label, local_idx in local_samples
    ]
    export_samples(dataset, global_samples, config.output_dir / "structures")

    with (config.output_dir / "selected_frames.json").open("w", encoding="utf-8") as handle:
        json.dump({label: frame_idx for label, frame_idx in global_samples}, handle, indent=2)

    if config.write_latex_table:
        write_subsystem_latex(
            config.output_dir / "subsystem_table.tex", number_to_config, config_counts
        )

    metadata_path = save_metadata(
        config.output_dir,
        {
            "dataset_paths": [str(path) for path in dataset_paths],
            "model": str(config.model.resolve()),
            "n_frames_total": dataset.n_frames,
            "n_frames_after_downsample": int(len(selected_indices)),
            "n_atoms_range": [int(min(dataset.n_atoms_list)), int(max(dataset.n_atoms_list))],
            "mean_descriptor_shape": list(map(int, mean_descriptors.shape)),
            "embedding_input_shape": list(map(int, embedding_input.shape)),
            "system_configurations": config_counts,
            "dataset_number_count": len(number_to_dataset),
            "config_number_count": len(number_to_config),
            "batch_size": config.batch_size,
            "sample_count": config.sample_count,
            "random_state": config.random_state,
            "method": config.method,
            "perplexity": config.perplexity if config.method == "tsne" else None,
            "tsne_early_exaggeration": (
                config.tsne_early_exaggeration if config.method == "tsne" else None
            ),
            "tsne_learning_rate": (
                config.tsne_learning_rate if config.method == "tsne" else None
            ),
            "descriptor_preprocess": config.descriptor_preprocess,
            "descriptor_pca_components": config.descriptor_pca_components,
            "max_per_config": config.max_per_config,
        },
    )

    return DeepmdDatasetSketchResult(
        output_dir=config.output_dir,
        dataset_count=len(dataset_paths),
        n_frames_total=dataset.n_frames,
        n_frames_after_downsample=int(len(selected_indices)),
        config_count=len(unique_configs),
        csv_path=csv_path,
        metadata_path=metadata_path,
        figure_path=figure_path,
        numbered_figure_path=numbered_figure_path,
        dataset_numbered_figure_path=dataset_numbered_figure_path,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build a standalone parser for the DeepMD sketch-map workflow."""

    parser = argparse.ArgumentParser(description="Visualize DeepMD datasets with model descriptors")
    parser.add_argument(
        "--dataset", type=Path, nargs="+", required=True, help="Dataset root(s) to scan"
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to frozen_model.pb")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--gpu", default="0", help="CUDA_VISIBLE_DEVICES value; use empty string for CPU"
    )
    parser.add_argument("--method", choices=("pca", "tsne"), default="tsne")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--tsne-early-exaggeration", type=float, default=12.0)
    parser.add_argument("--tsne-learning-rate", type=float)
    parser.add_argument(
        "--descriptor-preprocess",
        choices=("none", "standardize", "pca", "standardize-pca"),
        default="none",
    )
    parser.add_argument("--descriptor-pca-components", type=int, default=50)
    parser.add_argument("--max-per-config", type=int, default=200)
    parser.add_argument("--no-structure-images", action="store_true")
    parser.add_argument("--no-latex-table", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the standalone workflow."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_deepmd_dataset_sketch(
        DeepmdDatasetSketchConfig(
            dataset_roots=tuple(args.dataset),
            model=args.model,
            output_dir=args.output,
            batch_size=args.batch_size,
            sample_count=args.sample_count,
            random_state=args.random_state,
            overwrite=args.overwrite,
            gpu=args.gpu,
            method=args.method,
            perplexity=args.perplexity,
            tsne_early_exaggeration=args.tsne_early_exaggeration,
            tsne_learning_rate=args.tsne_learning_rate,
            descriptor_preprocess=args.descriptor_preprocess,
            descriptor_pca_components=args.descriptor_pca_components,
            max_per_config=args.max_per_config,
            write_structure_images=not args.no_structure_images,
            write_latex_table=not args.no_latex_table,
        )
    )
    print(result.output_dir)
    print(
        "datasets="
        f"{result.dataset_count} "
        f"frames={result.n_frames_total} "
        f"selected={result.n_frames_after_downsample}"
    )
    print(f"configs={result.config_count}")
    print(f"dataset_numbered_figure={result.dataset_numbered_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
