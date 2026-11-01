"""Assemble branch-aware motion and persistent geometric contact-state episodes."""
from __future__ import annotations
import copy
import numpy as np
from molsimflow.io.lammps_dump import minimum_image_vectors


def contact_state(row: dict, minimum_main_n2: int, gap_A: float) -> str:
    """A geometric screen only; no film or thermodynamic acceptance is implied."""
    if row['largest_cluster_n2_count'] < minimum_main_n2:
        return 'GAS_DEPLETED_OR_FRAGMENTED'
    top = sum(row[f'top_{d}_contacts_4p0'] for d in ['left', 'right'])
    bottom = sum(row[f'bottom_{d}_contacts_4p0'] for d in ['left', 'right'])
    if top and bottom:
        return 'MULTI_FACE_ATTACHED'
    if bottom:
        return 'BOTTOM_FACE_ATTACHED'
    left, right = (row[f'top_{d}_contacts_4p0'] for d in ['left', 'right'])
    if left and right:
        return 'SEAM_SPANNING_ATTACHED'
    if left:
        return 'LEFT_ATTACHED'
    if right:
        return 'RIGHT_ATTACHED'
    good = (row['fixed_max_displacement_A'] <= 1e-5
            and row['fixed_max_speed_A_ps'] <= 1e-8
            and row['N2_bond_max_A'] < 1.5
            and row['N2_bond_min_A'] > .8
            and row.get('Si_coordination_min') == 4
            and row.get('Si_coordination_max') == 4)
    if row['min_bubble_surface_distance_A'] >= gap_A and good:
        return 'DETACHMENT_CANDIDATE_GEOMETRY_ONLY'
    return 'NO_CONTACT_TRANSITION_OR_QA_HOLD'


def _bits(row: dict, name: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(row[name]), dtype=np.uint8))[:row['_site_count']].astype(bool)


def assemble_series(raw_rows: list[dict], parameters: dict) -> tuple[list[dict], list[dict]]:
    """Reset residence and velocity intervals at reseeds, missing frames and identity changes."""
    rows = copy.deepcopy(raw_rows)
    expected_stride = int(parameters['stride_steps'])
    dt = float(parameters['timestep_ps'])
    previous = None
    origin = None
    unwrapped = None
    epoch = -1
    episode = None
    events = []
    for row in rows:
        xyz = np.array([row[f'bubble_center_{axis}_A'] for axis in 'xyz'])
        bounds = np.asarray(row['_bounds']); lengths = bounds[:, 1]-bounds[:, 0]
        if previous is not None and row['step'] <= previous['step']:
            raise ValueError('Duplicate or unordered lineage steps')
        overlap = (len(set(row['_members']) & set(previous['_members'])) /
                   max(1, len(set(row['_members']) | set(previous['_members'])))) if previous else 1.
        gap = previous is None or row['step']-previous['step'] != expected_stride
        low_overlap = overlap < parameters['minimum_membership_overlap']
        reset = gap or row.get('reseed_boundary', False) or low_overlap
        if reset:
            epoch += 1
        if previous is None or gap or low_overlap:
            unwrapped = xyz.copy(); origin = xyz.copy()
            delta = None
        else:
            old = np.array([previous[f'bubble_center_{axis}_A'] for axis in 'xyz'])
            delta = minimum_image_vectors(xyz-old, lengths)
            unwrapped += delta
        row.update(global_ns=row['step']*dt/1000,
                   hold_elapsed_ns=(row['step']-parameters['hold_start_step'])*dt/1000,
                   continuity_epoch=epoch, membership_overlap=overlap,
                   temporal_interval_valid=not reset,
                   contact_state=contact_state(row, parameters['minimum_main_n2'], parameters['gap_screen_A']))
        for i, axis in enumerate('xyz'):
            row[f'center_{axis}_unwrapped_A'] = float(unwrapped[i])
            row[f'd{axis}_within_geometry_segment_A'] = float(unwrapped[i]-origin[i])
        row['bubble_vx_interval_A_ps'] = float(delta[0]/(expected_stride*dt)) if not reset else None
        row['bubble_minus_global_waterO_vx_A_ps'] = (row['bubble_vx_interval_A_ps']-row['waterO_mean_vx_A_ps']) if not reset else None
        row['lost_gas_contact_sites'] = None
        row['lost_sites_with_prior_water_neighbor'] = None
        row['lost_sites_with_current_water_neighbor'] = None
        if not reset:
            lost = _bits(previous, '_gas_bits') & ~_bits(row, '_gas_bits')
            row['lost_gas_contact_sites'] = int(lost.sum())
            row['lost_sites_with_prior_water_neighbor'] = int((lost & _bits(previous, '_water_bits')).sum())
            row['lost_sites_with_current_water_neighbor'] = int((lost & _bits(row, '_water_bits')).sum())
        if episode is None or reset or row['contact_state'] != episode['state']:
            if episode is not None:
                episode['end_reason'] = 'BOUNDARY_GAP_OR_IDENTITY_CHANGE' if reset else 'OBSERVED_STATE_CHANGE'
                events.append(episode)
            episode = dict(state=row['contact_state'], epoch=epoch, start_step=row['step'],
                           end_step=row['step'], frames=0, duration_ps=0.,
                           left_boundary=reset, persistent_screen=False)
        episode['end_step'] = row['step']; episode['frames'] += 1
        episode['duration_ps'] = (episode['end_step']-episode['start_step'])*dt
        episode['persistent_screen'] = episode['duration_ps'] >= parameters['persistence_ps']
        previous = row
    if episode is not None:
        episode['end_reason'] = 'ANALYSIS_WINDOW_END'; events.append(episode)
    return rows, events
