"""Same-face connected gas contact for compact planar slabs under PPP or PPF."""
from dataclasses import dataclass
import numpy as np
from molsimflow.postprocess.gas_solid_distribution import gas_solid_distances
from molsimflow.postprocess.nanobubble_attachment import molecule_centers
from molsimflow.postprocess.constant_force_species_timeseries import _periodic_query
from molsimflow.postprocess.surface_contact_memory import _component_sizes

@dataclass(frozen=True)
class SlabContact:
    distance: np.ndarray
    main: np.ndarray
    contact: np.ndarray
    face: np.ndarray
    component_size: np.ndarray
    centers: np.ndarray
    scaffold_center_z: float
    supported: bool

def same_face_slab_contact(nitrogen: np.ndarray, solid: np.ndarray,
                          scaffold: np.ndarray, bounds: np.ndarray, *,
                          contact_cutoff: float = 4.,
                          cluster_cutoff: float = 5.5,
                          minimum_support: int = 3,
                          periodic_z: bool = True) -> SlabContact:
    """Count contacts without joining opposite faces or disconnected touches.

    Requires a compact planar slab normal to z in an orthorhombic box.
    XY are periodic; nonperiodic Z uses a padded box only for cluster helpers,
    while physical distances and face assignment retain open Z geometry.
    Face +1/-1 follows the signed z offset from the current scaffold center.
    Contact with inconsistent gas/nearest-solid faces has label 0 and cannot
    establish supported contact. The caller must audit such ambiguity.
    """
    nitrogen, solid, scaffold, bounds = map(
        lambda a: np.asarray(a, float), (nitrogen, solid, scaffold, bounds))
    if bounds.shape != (3, 2) or np.any(np.diff(bounds, axis=1) <= 0):
        raise ValueError("Positive orthorhombic box required")
    if any(a.ndim != 2 or a.shape[1] != 3 or not len(a) or
           not np.isfinite(a).all() for a in (nitrogen, solid, scaffold)):
        raise ValueError("Finite nonempty coordinates required")
    if min(contact_cutoff, cluster_cutoff) <= 0 or minimum_support < 1:
        raise ValueError("Positive contact parameters required")
    L = np.diff(bounds, axis=1).ravel()
    effective = bounds.copy()
    if periodic_z:
        angles = 2 * np.pi * (scaffold[:, 2] - bounds[2, 0]) / L[2]
        zmean = np.mean(np.exp(1j * angles))
        if abs(zmean) < 1e-8:
            raise ValueError("Undefined scaffold circular center")
        z0 = bounds[2, 0] + (np.angle(zmean) % (2*np.pi)) * L[2] / (2*np.pi)
    else:
        z = np.concatenate((nitrogen[:, 2], solid[:, 2], scaffold[:, 2]))
        if z.min() < bounds[2, 0] or z.max() > bounds[2, 1]:
            raise ValueError("Nonperiodic coordinates outside declared box")
        pad = max(float(np.ptp(z)) + 2 * cluster_cutoff, 100.)
        effective[2] = [z.min() - pad, z.max() + pad]
        z0 = float(scaffold[:, 2].mean())
    def dz(z):
        d = z-z0
        return d-np.rint(d/L[2])*L[2] if periodic_z else d
    sz = dz(solid[:, 2])
    if periodic_z and np.ptp(sz) >= L[2]/2:
        raise ValueError("Solid is not a compact single slab")
    distance, main = gas_solid_distances(nitrogen, solid, bounds,
                                        periodic_z=periodic_z, cutoff=cluster_cutoff)
    centers = molecule_centers(nitrogen, effective)
    nearest, atom_distance = _periodic_query(nitrogen, solid, bounds, periodic_z=periodic_z)
    pick = 2*np.arange(len(centers)) + np.argmin(atom_distance.reshape(-1, 2), axis=1)
    solid_face = np.sign(sz[nearest[pick]]).astype(np.int8)
    gas_face = np.sign(dz(centers[:, 2])).astype(np.int8)
    contact = main & (distance <= contact_cutoff)
    face = np.where(contact & (solid_face == gas_face), gas_face, 0).astype(np.int8)
    sizes = np.zeros(len(centers), int)
    for side in (-1, 1):
        mask = face == side
        sizes[mask] = _component_sizes(centers[mask], effective, cluster_cutoff)
    return SlabContact(distance, main, contact, face, sizes, centers, float(z0),
                       bool(np.any(sizes >= minimum_support)))
