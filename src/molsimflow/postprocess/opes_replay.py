"""Offline OPES_METAD kernel replay for nonperiodic collective variables.

Uses recorded centers, widths, heights and log weights in deposition order.
Callers must qualify state timing against native bias observations. Periodic
CVs, OPES_METAD_EXPLORE and neighbor-list approximations are not handled.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class NonperiodicOPESReplay:
    """Replay recursive compression; kernel columns: centers, widths, height."""

    kernels: np.ndarray
    sum_weights: float
    biasfactor: float
    epsilon: float
    cutoff: float
    threshold: float
    kbt: float
    recursive: bool = True

    def __post_init__(self) -> None:
        self.kernels = np.array(self.kernels, dtype=float, copy=True)
        if self.kernels.ndim != 2 or self.kernels.shape[1] < 3:
            raise ValueError("Expected a two-dimensional kernel array")
        if self.kernels.shape[1] % 2 != 1:
            raise ValueError("Kernel columns must be centers, widths, height")
        self.ndim = (self.kernels.shape[1] - 1) // 2
        constants = np.array([self.sum_weights, self.biasfactor, self.epsilon,
                             self.cutoff, self.threshold, self.kbt])
        if not np.isfinite(constants).all() or np.any(constants[[0, 2, 3, 5]] <= 0):
            raise ValueError("Normalization, epsilon, cutoff and kBT must be positive")
        if self.biasfactor <= 1 or self.threshold < 0:
            raise ValueError("Require biasfactor > 1 and threshold >= 0")
        self._check_kernel(self.kernels)

    def _check_kernel(self, kernel: np.ndarray) -> None:
        if not np.isfinite(kernel).all() or np.any(kernel[..., self.ndim:] <= 0):
            raise ValueError("Kernel widths and heights must be finite and positive")

    def _nearest(self, center: np.ndarray, exclude: int | None = None) -> int | None:
        if not len(self.kernels) or self.threshold == 0:
            return None
        d = (self.kernels[:, :self.ndim] - center) / self.kernels[:, self.ndim:-1]
        norm2 = np.sum(d * d, axis=1)
        if exclude is not None:
            norm2[exclude] = np.inf
        index = int(np.argmin(norm2))
        return index if norm2[index] < self.threshold**2 else None

    def _merge(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        h = left[-1] + right[-1]
        w1, w2 = left[-1] / h, right[-1] / h
        delta = right[:self.ndim] - left[:self.ndim]
        center = left[:self.ndim] + w2 * delta
        variance = (w1 * left[self.ndim:-1]**2 + w2 * right[self.ndim:-1]**2
                    + w1 * w2 * delta**2)
        return np.concatenate((center, np.sqrt(variance), [h]))

    def add(self, center: np.ndarray, sigma: np.ndarray, height: float,
            log_weight: float) -> None:
        """Add a recorded kernel, conserving mass and moments in each merge."""
        center, sigma = np.asarray(center, float), np.asarray(sigma, float)
        if center.shape != (self.ndim,) or sigma.shape != (self.ndim,):
            raise ValueError("Wrong CV dimension")
        new = np.concatenate((center, sigma, [height]))
        self._check_kernel(new)
        weight = np.exp(log_weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Invalid deposition weight")
        index = self._nearest(center)
        if index is None:
            self.kernels = np.vstack((self.kernels, new))
        else:
            self.kernels[index] = self._merge(self.kernels[index], new)
            if self.recursive:
                while True:
                    other = self._nearest(self.kernels[index, :self.ndim], exclude=index)
                    if other is None:
                        break
                    keep, remove = sorted((index, other))
                    self.kernels[keep] = self._merge(self.kernels[keep], self.kernels[remove])
                    self.kernels = np.delete(self.kernels, remove, axis=0)
                    index = keep
        self.sum_weights += float(weight)

    def _density(self, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cv = np.asarray(cv, dtype=float)
        if cv.shape[-1:] != (self.ndim,) or not np.isfinite(cv).all():
            raise ValueError("Invalid CV point")
        d = (cv[..., None, :] - self.kernels[:, :self.ndim]) / self.kernels[:, self.ndim:-1]
        r2 = np.sum(d*d, axis=-1)
        active = r2 < self.cutoff**2
        gaussian = self.kernels[:, -1] * np.exp(-0.5*r2)
        shifted = np.where(active, gaussian-self.kernels[:, -1]*np.exp(-0.5*self.cutoff**2), 0)
        derivative = np.where(active[..., None],
                              -gaussian[..., None]*d/self.kernels[:, self.ndim:-1], 0)
        return shifted.sum(-1)/self.sum_weights, derivative.sum(-2)/self.sum_weights

    def normalization(self) -> float:
        """Full-kernel Z estimate at compressed kernel centers."""
        if not len(self.kernels):
            return 1.0
        return float(self._density(self.kernels[:, :self.ndim])[0].mean())

    def evaluate(self, cv: np.ndarray, zed: float | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Bias energy and CV gradient in the units of kBT."""
        if zed is None:
            zed = self.normalization()
        if not np.isfinite(zed) or zed <= 0:
            raise ValueError("Require a positive finite Z estimate")
        p, dp = self._density(cv)
        prefactor = self.kbt * (1 - 1/self.biasfactor)
        energy = prefactor * np.log(p/zed + self.epsilon)
        gradient = prefactor * dp / (p + zed*self.epsilon)[..., None]
        return energy, gradient
