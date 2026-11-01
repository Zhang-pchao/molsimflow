# Site-resolved contributions

Aggregate molecule observations by exact site identity and frame. Retain the
number of eligible molecules and divide by the full parent population in that
frame. Binary site occupancy cannot replace this weight when reconstructing
a molecule-based parent statistic. Contradictory states for the same site and
frame are rejected. Classify sites by the two environments observed during the
specified window; one observed type does not imply permanent specialization.

Count changes only between adjacent resolved saved observations at the expected
cadence and in the same source segment. Missing, unresolved and cadence/source
breaks reset the run. Two changes within one run establish a sampled reversal,
not continuous-time kinetics, a metastable state or causal control. Two types
seen only across gaps establish repeated site identity but not a resolved change.
