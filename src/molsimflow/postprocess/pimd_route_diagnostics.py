"""Compatibility imports for the scoped PIMD diagnostics module."""
from .pimd.route_diagnostics import (
    DistanceFrameMetrics as DistanceFrameMetrics,
    distance_frame_metrics as distance_frame_metrics,
    contiguous_episodes as contiguous_episodes,
    regional_weight_support as regional_weight_support,
    read_plumed_selected as read_plumed_selected,
    nearest_oxygen_proxy as nearest_oxygen_proxy,
)
