# Curved local density-front observer
This module extends local_density_front and reuses its radial crossing extraction. It fits local quadratic patches of r(phi,z)^2, not a common sphere or circle. Full azimuthal and vertical derivatives enter the gas-side normal, evaluated at a supplied solid reference plane. Extrapolation remains a model assumption that must be qualified on physical data.

The original direct quadratic-radius prototype failed the known-ellipsoid angle test. Squared radius corrects that analytic truncation bias without selecting MD parameters. Source and failed tests are preserved in the task receipts. Synthetic tests now cover ellipsoid variation, missing columns/heights, high residuals, complete water-density extraction, reference translation, and open-contour motion.

Missing columns or heights are not filled; >=80% finite crossings, sufficient rank, local RMS radial residual <=1 A, and a reference radius inside the sampled domain are required. These numerical gates do not establish molecular contact.

partial_normal_displacements intersects only adjacent finite target edges. It never closes a missing contour interval, and returns NaN when no local intersection exists. A fine polygon is needed when movements are comparable with chord discretization. Analytic tests include the second-order translation term rather than confusing its difference from a first-order projection with an error.

Callers must independently establish real contact association, material-frame registration, topology, water identity, spatial/time sensitivity and uncertainty before interpreting any angle or speed physically. A failed observer is not a falsification of capillary theory. Dense rays and sectors are correlated observations, not independent samples.
