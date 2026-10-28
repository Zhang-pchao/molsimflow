# Sampled collective contact fields

The caller provides an ordered fixed surface chart, exact terminal contact
occupancies, native integer steps, and explicit continuity segments.

- `sampled_window_means` censors incomplete native windows before decimation.
  It returns admitted row indices, means, sample counts and mean sample-time
  offsets. A cadence sensitivity must include all sampling phases.
- `periodic_contact_field` rasterizes the supplied site occupancies using the
  existing fixed ownership chart and performs periodic XY Gaussian convolution.
  Leading dimensions are never smoothed. Sigma is in chart length units.
- `periodic_mask_labels` retains all four-neighbor periodic components,
  including satellites and holes. Zero is background.

Thresholds, spatial scales, time windows and physical interpretation belong to
the consuming analysis contract. A smoothed envelope is not a thermodynamic
contact line, continuous molecular lifetime, adsorption energy or causal test.
