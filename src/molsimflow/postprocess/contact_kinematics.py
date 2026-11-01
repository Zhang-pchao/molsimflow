"""Exact finite-step kinematics for equal-mass diatomic contact distances.

These are geometric decompositions, not force, energy, or causal attributions.
A fixed molecular cohort is a mathematical reference, not a cluster detector.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from molsimflow.io.lammps_dump import minimum_image_vectors

COMPONENTS = ("cohort", "molecule", "rotation", "stretch", "site")

@dataclass(frozen=True)
class DiatomicSnapshot:
    centers: np.ndarray
    bonds: np.ndarray
    site: np.ndarray
    reference: np.ndarray

class ImageAlignmentError(ValueError):
    """The two contact vectors cannot share a minimum-image lattice offset."""

def diatomic_snapshot(atoms: np.ndarray, site: np.ndarray, reference: np.ndarray,
                      box: np.ndarray, maximum_bond_length: float) -> DiatomicSnapshot:
    """Reconstruct equal-mass pairs from unwrapped first atoms and short bonds.

    The first-atom trajectories must already be temporally unwrapped.
    The fixed ordering in each pair defines the bond direction.
    """
    atoms = np.asarray(atoms, float)
    box = np.asarray(box, float)
    site, reference = np.asarray(site, float), np.asarray(reference, float)
    if (atoms.ndim != 3 or atoms.shape[1:] != (2, 3) or len(atoms) == 0
            or box.shape != (3,) or np.any(box <= 0)
            or site.shape != (3,) or reference.shape != (3,)
            or not all(np.isfinite(x).all() for x in (atoms, box, site, reference))
            or not np.isfinite(maximum_bond_length) or maximum_bond_length <= 0):
        raise ValueError("Invalid pair geometry or box")
    bonds = minimum_image_vectors(atoms[:, 1] - atoms[:, 0], box)
    lengths = np.linalg.norm(bonds, axis=1)
    if np.any(lengths <= 0) or np.any(lengths >= maximum_bond_length):
        raise ValueError("Bond outside the declared reconstruction range")
    return DiatomicSnapshot(atoms[:, 0] + 0.5 * bonds, bonds,
                            site.copy(), reference.copy())

@dataclass(frozen=True)
class ContactDecomposition:
    vectors: np.ndarray
    projected: np.ndarray
    secant: np.ndarray
    r0: np.ndarray
    r1: np.ndarray
    distances: np.ndarray
    closure_error: float

def decompose_contact(previous: DiatomicSnapshot, current: DiatomicSnapshot,
                      molecule: int, atom_in_pair: int, box: np.ndarray,
                      tolerance: float = 1e-9) -> ContactDecomposition:
    """Split a fixed-atom distance increment into five exactly closing terms."""
    if previous.centers.shape != current.centers.shape or atom_in_pair not in (0, 1):
        raise ValueError("Molecular identities or atom index do not match")
    if not 0 <= molecule < len(previous.centers):
        raise ValueError("Molecule index out of bounds")
    sign = -1 if atom_in_pair == 0 else 1
    m0, m1 = previous.centers[molecule], current.centers[molecule]
    b0, b1 = previous.bonds[molecule], current.bonds[molecule]
    a0, a1 = m0 + sign * b0 / 2, m1 + sign * b1 / 2
    raw0, raw1 = a0 - previous.site, a1 - current.site
    r0 = minimum_image_vectors(raw0, box)
    r1 = raw1 - (raw0 - r0)
    if np.max(np.abs(r1 - minimum_image_vectors(raw1, box))) > tolerance:
        raise ImageAlignmentError("Contact endpoints require different lattice images")
    d0, d1 = np.linalg.norm(r0), np.linalg.norm(r1)
    if d0 + d1 <= tolerance:
        raise ValueError("Undefined distance secant for coincident endpoints")
    secant = (r0 + r1) / (d0 + d1)
    c0, c1 = previous.centers.mean(0), current.centers.mean(0)
    l0, l1 = np.linalg.norm(b0), np.linalg.norm(b1)
    u0, u1 = b0 / l0, b1 / l1
    vectors = np.array([
        (c1-current.reference) - (c0-previous.reference),
        (m1-c1) - (m0-c0),
        sign * (l1+l0) / 4 * (u1-u0),
        sign * (l1-l0) / 4 * (u1+u0),
        -(current.site-current.reference) + (previous.site-previous.reference),
    ])
    projected = vectors @ secant
    error = abs(float(projected.sum() - (d1-d0)))
    if np.max(np.abs(vectors.sum(0) - (r1-r0))) > tolerance or error > tolerance:
        raise ValueError("Kinematic decomposition failed to close")
    return ContactDecomposition(vectors, projected, secant, r0, r1,
                                np.array([d0, d1]), error)
