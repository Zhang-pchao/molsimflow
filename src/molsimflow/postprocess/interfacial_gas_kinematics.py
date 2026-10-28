"""Identity-preserving gas-layer exchange and rigid-motion diagnostics.

Callers own molecule pairing, periodic unwrapping, cluster membership and units.
An instantaneous best-fit rotation of a fluid is not a rolling mechanism.
"""
from __future__ import annotations
import numpy as np


def quantile_layer_labels(z, membership, fractions=(1 / 3, 2 / 3)):
    """Label lower/middle/upper gas layers 0/1/2; nonmembers are -1.

    Dynamic layers describe position inside the gas, not solid contact.
    Ordered molecular identities must be retained across calls.
    """
    z = np.asarray(z, dtype=float)
    member = np.asarray(membership, dtype=bool)
    if z.ndim != 1 or z.shape != member.shape or member.sum() < 3:
        raise ValueError("Matching one-dimensional arrays and three members required")
    if not np.isfinite(z).all() or not 0 < fractions[0] < fractions[1] < 1:
        raise ValueError("Finite coordinates and ordered quantile fractions required")
    low, high = np.quantile(z[member], fractions)
    if high <= low:
        raise ValueError("Degenerate layer boundaries")
    labels = np.full(z.shape, -1, dtype=np.int8)
    labels[member] = np.digitize(z[member], [low, high])
    return labels


def lagged_state_statistics(states, lag, origin_state=0):
    """Pool molecule/time-origin counts at a positive integer frame lag.

    Return intermittent return and uninterrupted *sampled* survival.
    Departed cluster members remain in the denominator. The caller must
    provide a contiguous uniform time grid within one identity epoch.
    """
    s = np.asarray(states)
    if s.ndim != 2 or s.dtype.kind not in "iu" or not np.isin(s, [-1, 0, 1, 2]).all():
        raise ValueError("States must be integer time-by-molecule labels -1,0,1,2")
    if not isinstance(lag, (int, np.integer)) or not 0 < lag < len(s):
        raise ValueError("Lag must be a positive integer smaller than frame count")
    if origin_state not in (0, 1, 2):
        raise ValueError("Origin state must be 0, 1 or 2")
    initial = s[:-lag] == origin_state
    denominator = int(initial.sum())
    if not denominator:
        raise ValueError("No initial population")
    terminal = s[lag:]
    stay = initial.copy()
    for offset in range(1, lag + 1):
        stay &= s[offset:len(s) - lag + offset] == origin_state
    counts = {int(k): int((initial & (terminal == k)).sum()) for k in (-1, 0, 1, 2)}
    return dict(origin_count=denominator, terminal_counts=counts,
                return_fraction=counts[origin_state] / denominator,
                sampled_survival_fraction=int(stay.sum()) / denominator,
                terminal_fractions={k: v / denominator for k, v in counts.items()})


def fit_rigid_motion(positions, velocities):
    """Fit v=translation+omega cross r with equal molecular weights.

    Inputs use a common locally unwrapped frame. Units of omega are velocity
    units divided by length units. Thermal/fluctuating velocities remain in
    the residual; block and coarse-field checks are the caller's responsibility.
    """
    x, v = np.asarray(positions, float), np.asarray(velocities, float)
    if x.ndim != 2 or x.shape[1:] != (3,) or v.shape != x.shape or len(x) < 3:
        raise ValueError("Matching N-by-3 coordinates/velocities required")
    if not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite coordinates or velocities")
    r, u = x - x.mean(axis=0), v - v.mean(axis=0)
    inertia = np.eye(3) * np.sum(r * r) - r.T @ r
    if np.linalg.matrix_rank(inertia) < 3:
        raise ValueError("Rotation is not identifiable in degenerate geometry")
    omega = np.linalg.solve(inertia, np.cross(r, u).sum(axis=0))
    rotation = np.cross(omega, r)
    total = float(np.sum(u * u))
    residual = float(np.sum((u - rotation) ** 2))
    return dict(translation=v.mean(axis=0), omega=omega,
                rotation_energy_fraction=1 - residual / total if total > 1e-24 else 0.,
                residual_rms=float(np.sqrt(residual / len(x))))


def coarse_velocity_field(local_positions, relative_velocities, x_edges, z_edges):
    """Accumulate an XZ field over equal-weight molecular/frame samples.

    Return counts, coordinate sums and velocity sums; do not silently fill
    empty cells. This is a Y-projected field rather than a central Y slice.
    """
    p, v = np.asarray(local_positions, float), np.asarray(relative_velocities, float)
    if p.ndim != 2 or p.shape[1:] != (3,) or p.shape != v.shape:
        raise ValueError("Matching sample-by-3 arrays required")
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError("Finite samples required")
    edges = [np.asarray(x_edges, float), np.asarray(z_edges, float)]
    if any(e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all()
           or np.any(np.diff(e) <= 0) for e in edges):
        raise ValueError("Finite increasing bin edges required")
    xy = p[:, [0, 2]]
    count = np.histogram2d(*xy.T, bins=edges)[0]
    position = np.stack([np.histogram2d(*xy.T, bins=edges, weights=p[:, j])[0]
                         for j in range(3)], axis=-1)
    velocity = np.stack([np.histogram2d(*xy.T, bins=edges, weights=v[:, j])[0]
                         for j in range(3)], axis=-1)
    return count, position, velocity


def planar_shape_tensor(local_positions):
    """Return equal-weight XY covariance and principal-axis shape descriptors.

    Supply one consistently unwrapped finite particle cluster. The caller must
    validate periodic extent. Isotropic clusters have no identifiable axis;
    collinear clusters have infinite aspect ratio. Neither is silently filled.
    """
    x = np.asarray(local_positions, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("At least three finite unwrapped planar positions required")
    x = x - x.mean(axis=0)
    tensor = x.T @ x / len(x)
    eigenvalues, vectors = np.linalg.eigh(tensor)
    eigenvalues = np.maximum(eigenvalues[::-1], 0.)
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("Coincident positions have no spatial extent")
    anisotropy = float((eigenvalues[0]-eigenvalues[1])/total)
    axis = vectors[:, -1] if anisotropy > 1e-12 else np.full(2, np.nan)
    aspect = float(np.sqrt(eigenvalues[0]/eigenvalues[1])) if eigenvalues[1] > 0 else np.inf
    return dict(tensor=tensor, eigenvalues=eigenvalues, principal_axis=axis,
                aspect_ratio=aspect, anisotropy=anisotropy,
                y_variance_fraction=float(tensor[1, 1]/total))


def periodic_cluster_winding(positions, bounds, cutoff):
    """Test periodic graph winding and return consistent integer image shifts.

    Neighbor edges use orthorhombic minimum-image distances. A nonzero cycle
    shift marks winding on that axis: a single finite unwrapped cluster chart
    then does not exist. Connectivity depends on the supplied physical cutoff.
    Image shifts remain a spanning-tree convention for winding components.
    """
    from scipy.spatial import cKDTree
    x, b = np.asarray(positions, float), np.asarray(bounds, float)
    if (x.ndim != 2 or x.shape[1] != 3 or not len(x) or b.shape != (3, 2)
            or not np.isfinite(x).all() or not np.isfinite(b).all()
            or not np.isfinite(cutoff)):
        raise ValueError("Finite nonempty positions and orthorhombic bounds required")
    lengths = b[:, 1]-b[:, 0]
    if np.any(lengths <= 0) or not 0 < cutoff < .5*lengths.min():
        raise ValueError("Cutoff must lie below every half-box length")
    x = (x-b[:, 0]) % lengths
    neighbors = [[] for _ in x]
    for i, j in sorted(cKDTree(x, boxsize=lengths).query_pairs(cutoff)):
        shift = -np.rint((x[j]-x[i])/lengths).astype(np.int64)
        neighbors[i].append((j, shift))
        neighbors[j].append((i, -shift))
    images = np.zeros((len(x), 3), dtype=np.int64)
    visited = np.zeros(len(x), dtype=bool)
    winding = np.zeros(3, dtype=bool)
    components = 0
    for root in range(len(x)):
        if visited[root]:
            continue
        components += 1
        visited[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for j, shift in neighbors[i]:
                candidate = images[i]+shift
                if visited[j]:
                    winding |= candidate != images[j]
                else:
                    images[j] = candidate
                    visited[j] = True
                    stack.append(j)
    return dict(winding_axes=winding, image_shifts=images, components=components)


def signed_planar_turning(
    positions, steps, epochs, membership, *,
    expected_step, axes=(0, 2), minimum_radius=3.0,
    maximum_increment=np.pi / 2,
):
    """Return supported signed angle increments around a supplied moving origin.

    Positions have shape (time, ordered identity, 3) in a caller-qualified
    finite local chart. The caller owns the center, PBC chart and membership.
    Every interval must share epoch and cadence, retain the identity, stay
    outside the minimum axis radius, and pass the sampled angular-increment
    bound. Missing intervals are NaN, never zero rotation.

    Positive angle in the default XZ plane is opposite positive omega_y.
    Subsampling can hide arbitrary intervening turns even if this bound passes.
    Net angle, nonzero vorticity, or a fitted rotation does not prove a closed
    spatial orbit, coherent circulation, or rolling. Report rejected support,
    cadence/radius sensitivity, signed cancellation and appropriate controls.
    """
    p = np.asarray(positions, dtype=float)
    step = np.asarray(steps)
    epoch = np.asarray(epochs)
    member = np.asarray(membership)
    if p.ndim != 3 or p.shape[2] != 3 or len(p) < 2:
        raise ValueError("At least two time-by-identity-by-three positions required")
    if step.shape != (len(p),) or epoch.shape != step.shape:
        raise ValueError("One step and epoch per frame required")
    if step.dtype.kind not in "iu" or epoch.dtype.kind not in "iu":
        raise ValueError("Integer steps and epochs required")
    if np.any(np.diff(step) <= 0) or np.any(epoch < 0):
        raise ValueError("Strictly increasing steps and nonnegative epochs required")
    if member.shape != p.shape[:2] or member.dtype.kind != "b":
        raise ValueError("Boolean time-by-identity membership required")
    if not np.isfinite(p[member]).all():
        raise ValueError("Present positions must be finite")
    if (len(axes) != 2 or axes[0] == axes[1]
            or any(a not in (0, 1, 2) for a in axes)):
        raise ValueError("Two distinct spatial axes required")
    if (not isinstance(expected_step, (int, np.integer)) or expected_step <= 0
            or not np.isfinite(minimum_radius) or minimum_radius <= 0
            or not np.isfinite(maximum_increment)
            or not 0 < maximum_increment < np.pi):
        raise ValueError("Positive cadence/radius and angular bound below pi required")
    xy = p[..., list(axes)]
    radius = np.linalg.norm(xy, axis=-1)
    a, b = xy[:-1], xy[1:]
    angle = np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
                      np.sum(a * b, axis=-1))
    time_ok = (np.diff(step) == expected_step) & (epoch[:-1] == epoch[1:])
    identity_ok = member[:-1] & member[1:]
    radius_ok = (radius[:-1] >= minimum_radius) & (radius[1:] >= minimum_radius)
    finite_ok = np.isfinite(angle)
    increment_ok = np.abs(angle) <= maximum_increment
    valid = time_ok[:, None] & identity_ok & radius_ok & finite_ok & increment_ok
    return dict(
        angle_increment=np.where(valid, angle, np.nan),
        valid_interval=valid,
        time_interval_valid=time_ok,
        member_interval_valid=identity_ok,
        radius_interval_valid=radius_ok,
        increment_interval_valid=increment_ok,
        planar_radius=radius,
    )
