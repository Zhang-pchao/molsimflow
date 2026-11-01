"""Four-face chemical-junction contact adapters built on existing bubble geometry."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from molsimflow.io.lammps_dump import minimum_image_vectors, periodic_center
from molsimflow.postprocess.nanobubble_attachment import (
    SelectedFrame, analyze_frame, molecule_centers, largest_cluster,
)


def site_area_weights(xy: np.ndarray, lengths: np.ndarray, spacing: float) -> np.ndarray:
    """Assign periodic reference-plane grid-cell areas to nearest terminal sites."""
    if len(xy) == 0 or spacing <= 0:
        raise ValueError("Positive spacing and nonempty sites are required")
    shape = np.ceil(lengths[:2] / spacing).astype(int)
    dx = lengths[:2] / shape
    x, y = np.meshgrid((np.arange(shape[0]) + .5) * dx[0],
                       (np.arange(shape[1]) + .5) * dx[1], indexing='ij')
    grid = np.column_stack((x.ravel(), y.ravel()))
    nearest = cKDTree(xy % lengths[:2], boxsize=lengths[:2]).query(grid)[1]
    return np.bincount(nearest, minlength=len(xy)) * float(np.prod(dx))


def frame_metrics(frame, reference: dict, parameters: dict, source: Path) -> dict:
    """Compute one frame without imposing a temporal interpretation or continuity."""
    fields = {name: i for i, name in enumerate(frame.atom_fields)}
    required = {'id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz'}
    if not required <= fields.keys():
        raise ValueError(f"Missing frame columns: {required-fields.keys()}")
    arr = np.asarray(frame.atom_rows, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite trajectory values")
    ids = arr[:, fields['id']].astype(int)
    if not np.array_equal(ids, np.arange(1, len(reference['types']) + 1)):
        raise ValueError("Atom count/order/identity differs from the reference")
    if not np.array_equal(arr[:, fields['type']].astype(int), reference['types']):
        raise ValueError("Atom types differ from the reference")
    lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
    if not np.allclose(lengths, reference['lengths'], atol=1e-8, rtol=0):
        raise ValueError("Changing box requires a new analysis contract")
    if frame.box_header.split()[-3:] != ['pp', 'pp', 'pp']:
        raise ValueError("This adapter requires explicitly periodic XYZ boundaries")
    xyz = arr[:, [fields[v] for v in ['x', 'y', 'z']]]
    velocity = arr[:, [fields[v] for v in ['vx', 'vy', 'vz']]]
    nidx = reference['nitrogen_ids'] - 1
    sites = reference['site_ids'] - 1
    groups = reference['site_groups']
    nitrogen = xyz[nidx]
    centers = molecule_centers(nitrogen, frame.bounds)
    members = largest_cluster(centers, frame.bounds, parameters['cluster_cutoff_A'])
    center = periodic_center(centers[members], frame.bounds)
    top = sites[np.char.startswith(groups.astype(str), 'top_')]
    surface_z = float(np.median(xyz[top, 2]))
    rec = analyze_frame(SelectedFrame(source, frame.frame_index, frame.timestep,
                                      frame.bounds, xyz[sites], nitrogen),
                        surface_z=surface_z,
                        cluster_cutoff=parameters['cluster_cutoff_A'], contact_cutoff=4.)
    rec.pop('source_file'); rec.pop('source_frame')
    local = minimum_image_vectors(centers[members] - center, lengths)
    rec['main_x_width_q05_q95_A'] = float(np.diff(np.quantile(local[:, 0], [.05, .95]))[0])
    rec['main_y_width_q05_q95_A'] = float(np.diff(np.quantile(local[:, 1], [.05, .95]))[0])
    rec['center_internal_seam_signed_distance_A'] = float(
        minimum_image_vectors(center - np.array([parameters['internal_seam_x_A'],
                                                center[1], center[2]]), lengths)[0])
    main_atom_indices = np.column_stack((2 * members, 2 * members + 1)).ravel()
    main_xyz = nitrogen[main_atom_indices]
    origin = frame.bounds[:, 0]
    gas_tree = cKDTree((main_xyz-origin) % lengths, boxsize=lengths)
    site_gas_distance = gas_tree.query((xyz[sites]-origin) % lengths)[0]
    contact_cutoffs = parameters['contact_cutoffs_A']
    all_molecule_distances = []
    for group in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
        mask = groups == group
        if not mask.any():
            raise ValueError(f"Missing surface material/face: {group}")
        tree = cKDTree((xyz[sites[mask]]-origin) % lengths, boxsize=lengths)
        distances = tree.query((nitrogen-origin) % lengths)[0].reshape(-1, 2).min(axis=1)
        all_molecule_distances.append(distances)
        for cutoff in contact_cutoffs:
            label = f'{cutoff:.1f}'.replace('.', 'p')
            rec[f'{group}_contacts_{label}'] = int(np.sum(distances[members] <= cutoff))
            rec[f'{group}_site_area_{label}_A2'] = float(
                reference['site_weights_1A'][mask & (site_gas_distance <= cutoff)].sum())
        rec[f'{group}_site_area_4p0_grid2A_A2'] = float(
            reference['site_weights_2A'][mask & (site_gas_distance <= 4.)].sum())
        rec[f'{group}_minimum_main_gap_A'] = float(distances[members].min())
    distance = np.min(all_molecule_distances, axis=0)
    contacts = members[distance[members] <= 4.]
    if len(contacts):
        cx = minimum_image_vectors(centers[contacts] - center, lengths)[:, 0]
        rec['contact_rear_relative_A'], rec['contact_front_relative_A'] = map(
            float, np.quantile(cx, [.05, .95]))
    else:
        rec['contact_rear_relative_A'] = rec['contact_front_relative_A'] = None
    fixed = reference['fixed_ids'] - 1
    rec['fixed_max_displacement_A'] = float(np.linalg.norm(
        minimum_image_vectors(xyz[fixed]-reference['fixed_positions'], lengths), axis=1).max())
    rec['fixed_max_speed_A_ps'] = float(np.linalg.norm(velocity[fixed], axis=1).max())
    bond_lengths = np.linalg.norm(minimum_image_vectors(nitrogen[1::2]-nitrogen[::2], lengths), axis=1)
    rec['N2_bond_max_A'] = float(bond_lengths.max())
    rec['N2_bond_min_A'] = float(bond_lengths.min())
    if 'v_sicn' in fields:
        cn = arr[reference['silicon_ids']-1, fields['v_sicn']]
        rec['Si_coordination_min'] = float(cn.min())
        rec['Si_coordination_max'] = float(cn.max())
    else:
        rec['Si_coordination_min'] = rec['Si_coordination_max'] = None
    water = reference['waterO_ids'] - 1
    rec['waterO_mean_vx_A_ps'] = float(velocity[water, 0].mean())
    # This measures proximity to original sites, not insertion into a liquid film.
    water_tree = cKDTree((xyz[water]-origin) % lengths, boxsize=lengths)
    site_water = water_tree.query((xyz[sites]-origin) % lengths)[0] <= parameters['water_neighbor_cutoff_A']
    rec['gas_contact_site_count'] = int((site_gas_distance <= 4.).sum())
    rec['water_neighbor_site_count'] = int(site_water.sum())
    rec['_gas_bits'] = np.packbits(site_gas_distance <= 4.).tobytes().hex()
    rec['_water_bits'] = np.packbits(site_water).tobytes().hex()
    rec['_site_count'] = int(len(sites))
    rec['_members'] = members.tolist()
    rec['_bounds'] = frame.bounds.tolist()
    for key, value in list(rec.items()):
        if isinstance(value, np.generic):
            rec[key] = value.item()
        if isinstance(rec[key], float) and not np.isfinite(rec[key]):
            rec[key] = None
    return rec
