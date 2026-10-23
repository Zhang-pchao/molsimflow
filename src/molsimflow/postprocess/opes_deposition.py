"""Compatibility imports for the scoped PIMD diagnostics module."""
from .pimd.opes_deposition import (
    OpesKernelBatch as OpesKernelBatch,
    OpesDepositionComparison as OpesDepositionComparison,
    read_opes_kernel_batches as read_opes_kernel_batches,
    compare_opes_deposition_weights as compare_opes_deposition_weights,
)
