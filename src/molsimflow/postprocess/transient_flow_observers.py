"""Reference-frame and profile observers for conditional transient responses.

These are descriptive observers, not bias coordinates or equilibrium estimators.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from molsimflow.io.lammps_dump import minimum_image_vectors

MVV2E = 1.0364269e-4
KB = 8.617343e-5

def weighted_quantiles(x: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Midpoint-CDF weighted quantiles; return NaN if support is absent."""
    x, weights = np.asarray(x, float), np.asarray(weights, float)
    keep = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.full(len(qs), np.nan)
    order = np.argsort(x[keep], kind="stable")
    xx, ww = x[keep][order], weights[keep][order]
    return np.interp(qs, (np.cumsum(ww)-0.5*ww)/ww.sum(), xx)

def periodic_density(x: np.ndarray, length: float, spacing: float,
                     sigma: float = 1.) -> np.ndarray:
    """Normalized periodic cloud-in-cell density, smoothed in physical units."""
    n = max(8, int(round(length / spacing)))
    u = (np.asarray(x) % length) * n / length
    i = np.floor(u).astype(int)
    t = u - i
    h = np.bincount(i % n, 1-t, minlength=n)
    h += np.bincount((i+1) % n, t, minlength=n)
    h = gaussian_filter1d(h, sigma*n/length, mode="wrap")
    return h / h.sum() if h.sum() else h

def density_shift(reference: np.ndarray, current: np.ndarray, length: float) -> dict:
    """Align current to reference; positive shift means positive physical motion.

    Alternative-peak ratio excludes a 2 A neighborhood around the best shift.
    This diagnostic is not a statistical confidence or unique-translation proof.
    """
    a = np.asarray(reference)-np.mean(reference)
    b = np.asarray(current)-np.mean(current)
    den = np.linalg.norm(a)*np.linalg.norm(b)
    if den < 1e-15:
        return dict(shift_A=np.nan, correlation=np.nan, alternative_ratio=np.nan)
    corr = np.fft.ifft(np.conj(np.fft.fft(a))*np.fft.fft(b)).real / den
    n = len(corr); k = int(np.argmax(corr))
    left, peak, right = corr[(k-1)%n], corr[k], corr[(k+1)%n]
    curvature = left - 2*peak + right
    offset = .5*(left-right)/curvature if curvature < -1e-15 else 0.
    offset = np.clip(offset, -.5, .5)
    shift = ((k+offset+n/2)%n-n/2)*length/n
    dist = np.abs((np.arange(n)-k+n/2)%n-n/2)*length/n
    other = corr[dist >= 2.]
    ratio = other.max()/peak if len(other) and peak > 0 else np.nan
    return dict(shift_A=float(shift), correlation=float(peak),
                alternative_ratio=float(ratio))

def velocity_moments(masses: np.ndarray, velocities: np.ndarray) -> dict:
    """Mass flow and kinetic temperature after removing three COM DOF."""
    m, v = np.asarray(masses), np.asarray(velocities)
    n = len(m)
    if n == 0:
        return dict(atoms=0, mass_amu=0., vx_A_ps=np.nan, vy_A_ps=np.nan,
                    vz_A_ps=np.nan, kinetic_T_K=np.nan, kinetic_dof=0)
    mean = np.sum(m[:,None]*v, axis=0)/m.sum()
    dof = max(0, 3*n-3)
    temp = MVV2E*np.sum(m[:,None]*(v-mean)**2)/(KB*dof) if dof else np.nan
    return dict(atoms=n, mass_amu=float(m.sum()), vx_A_ps=float(mean[0]),
                vy_A_ps=float(mean[1]), vz_A_ps=float(mean[2]),
                kinetic_T_K=float(temp), kinetic_dof=dof)

def contact_envelope(x: np.ndarray, distances: np.ndarray,
                     midpoint: float, softness: float=.25) -> dict:
    """Soft contact-weighted 10/90% X envelope, never a thermodynamic contact line."""
    w = 1/(1+np.exp(np.clip((distances-midpoint)/softness,-60,60)))
    left,right = weighted_quantiles(x,w,np.array([.1,.9]))
    return dict(left_A=float(left),right_A=float(right),
                midpoint_A=float((left+right)/2),halfwidth_A=float((right-left)/2),
                weight=float(w.sum()),effective_members=float(w.sum()**2/(w@w)))

def coherent_center(coords: np.ndarray, previous: np.ndarray,
                    lengths: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Unwrap a compact nonwinding cluster about the previous center."""
    delta = minimum_image_vectors(coords-previous,lengths)
    center = previous+delta.mean(axis=0)
    return center, minimum_image_vectors(coords-center,lengths)
