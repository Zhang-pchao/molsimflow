# Open-population gas centroid budget

`moving_membership_centroid_budget` accepts two N-by-D position arrays and two boolean membership masks with identical ordered, equal-mass particle identities. It returns retained displacement, joined and departed contributions, the exact centroid change, and count/centroid closure. Empty populations remain explicitly undefined.

For final population N1 and initial center C0:
delta C = sum_retained(x1-x0)/N1 + sum_joined(x1-C0)/N1 - sum_departed(x0-C0)/N1.

The caller must validate a common periodic chart, native joins, identity, and temporal support. Sum native-step terms to describe a trajectory; a long endpoint budget generally gives different component allocations. Report cadence sensitivity. Finite precision in cached coordinates limits comparison with centers stored at higher precision.

This is an observer decomposition. The retained term is not the total force-driven translation of a closed body; membership change is not automatically evaporation or condensation. No force, causal effect, interface energy, or new clustering cutoff is implied.
