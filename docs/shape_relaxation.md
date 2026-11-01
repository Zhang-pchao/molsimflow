# Mean shape relaxation helpers

The shape_relaxation module contains small observers and descriptive model fits:

- moment_axes(points) returns moment-equivalent uniform-ellipsoid axes and a volume proxy from unwrapped coordinates.
- constant_volume_radius(c, a0, c0) evaluates the conditional fixed-volume geometric relation. Measured c remains an input.
- relaxation_curve and fit_relaxation implement a three-parameter single exponential with declared numerical bounds and boundary flags.

These functions do not identify free energy, friction, equilibrium, independent observations or a mechanism. A moment volume can change when a non-ellipsoidal shape changes at constant physical volume. Data selection, identity, periodic unwrapping, time averaging and calibration-versus-prediction separation remain caller responsibilities.

Analytic tests cover gas-side circle-angle convention, exact relaxation parameter recovery, moment covariance under volume-preserving deformation, and nonpositive-axis rejection.
