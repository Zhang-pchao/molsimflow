"""Classical kinetic partitions and native-image translation diagnostics.

Units are LAMMPS metal: amu, Angstrom/ps, eV, K. Molecular pairs must be
supplied in immutable pair order by the caller; no bond reassignment is done.
"""
from __future__ import annotations
import numpy as np
from molsimflow.io.lammps_dump import minimum_image_vectors

MVV2E = 1.0364269e-4
KB = 8.617343e-5

def kinetic_partition(masses: np.ndarray, velocities: np.ndarray, *,
                      remove_com: bool, degrees_of_freedom: int | None = None) -> dict:
    """Return lab/COM/internal energies and an explicitly defined temperature."""
    m, v = np.asarray(masses, float), np.asarray(velocities, float)
    if m.ndim != 1 or v.shape != (len(m), 3) or len(m) < 1:
        raise ValueError("Expected N masses and N x 3 velocities")
    if not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(m <= 0):
        raise ValueError("Masses must be positive; all inputs must be finite")
    dof = 3 * len(m) - 3 if degrees_of_freedom is None else degrees_of_freedom
    if dof <= 0:
        raise ValueError("Positive degrees of freedom required")
    vc = (m[:, None] * v).sum(axis=0) / m.sum()
    lab = .5 * MVV2E * np.sum(m[:, None] * v * v)
    com = .5 * MVV2E * m.sum() * np.dot(vc, vc)
    internal = .5 * MVV2E * np.sum(m[:, None] * (v - vc)**2)
    return dict(temperature_K=float(2 * (internal if remove_com else lab) / (dof * KB)),
                lab_KE_eV=float(lab), COM_KE_eV=float(com),
                internal_KE_eV=float(internal), COM_velocity_A_ps=vc,
                closure_eV=float(lab-com-internal))

def diatomic_modes(positions: np.ndarray, velocities: np.ndarray,
                   masses: np.ndarray, lengths: np.ndarray) -> dict:
    """Split fixed diatomics into gas COM, translation, rotation and radial KE."""
    p, v, m, L = map(lambda a: np.asarray(a, float), (positions, velocities, masses, lengths))
    if p.ndim != 3 or p.shape[1:] != (2, 3) or v.shape != p.shape or m.shape != p.shape[:2]:
        raise ValueError("Expected paired positions/velocities (M,2,3) and masses (M,2)")
    if len(p) < 2 or L.shape != (3,) or np.any(L <= 0) or not np.isfinite(L).all():
        raise ValueError("At least two molecules and positive finite box lengths required")
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite positions")
    total = kinetic_partition(m.ravel(), v.reshape(-1, 3), remove_com=True)
    d = minimum_image_vectors(p[:, 1] - p[:, 0], L)
    bond = np.linalg.norm(d, axis=1)
    if np.any(bond <= 0):
        raise ValueError("Zero molecular bond")
    u, dv = d/bond[:, None], v[:, 1]-v[:, 0]
    vr = np.sum(dv*u, axis=1)
    tangential = dv - vr[:, None]*u
    reduced, molecular_mass = m.prod(axis=1)/m.sum(axis=1), m.sum(axis=1)
    radial = .5 * MVV2E * np.sum(reduced*vr**2)
    rotational = .5 * MVV2E * np.sum(reduced[:, None]*tangential**2)
    vc = (m[:, :, None]*v).sum(axis=1)/molecular_mass[:, None]
    molecular = kinetic_partition(molecular_mass, vc, remove_com=True)
    trans = molecular["internal_KE_eV"]
    n = len(p)
    return dict(translation_K=2*trans/((3*n-3)*KB),
                rotation_K=2*rotational/(2*n*KB), radial_K=2*radial/(n*KB),
                translation_eV=trans, rotation_eV=float(rotational), radial_eV=float(radial),
                COM_eV=total["COM_KE_eV"], lab_eV=total["lab_KE_eV"],
                closure_eV=total["lab_KE_eV"]-total["COM_KE_eV"]-trans-rotational-radial,
                min_bond_A=float(bond.min()), max_bond_A=float(bond.max()))

def native_image_translation(positions: np.ndarray, images: np.ndarray,
                             reference_unwrapped: np.ndarray,
                             lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit translation only to explicitly unwrapped PPP scaffold positions."""
    p, im, ref, L = map(lambda a: np.asarray(a, float),
                       (positions, images, reference_unwrapped, lengths))
    if p.ndim != 2 or p.shape[1] != 3 or im.shape != p.shape or ref.shape != p.shape or not len(p):
        raise ValueError("Expected nonempty matching N x 3 scaffold arrays")
    if L.shape != (3,) or np.any(L <= 0) or not all(np.isfinite(a).all() for a in (p,im,ref,L)):
        raise ValueError("Invalid native-image inputs")
    if not np.array_equal(im, np.rint(im)):
        raise ValueError("Image flags must be integers")
    displacement = p+im*L-ref
    shift = np.median(displacement, axis=0)
    return shift, displacement-shift
