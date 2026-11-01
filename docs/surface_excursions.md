# Sampled exterior visits and returns

`molsimflow.postprocess.surface_excursions.observed_exterior_return` classifies
one aligned, complete, source-contiguous path starting in a contact region A.
An independent exterior B must be disjoint from A. The helper returns the first
observed B sample, first subsequent A sample, sampled transition brackets, and
explicit right censoring when no return is observed by the window end.

No visit observed does not exclude an inter-frame visit. Thinning can hide B or
A samples; the helper does not estimate continuous first-passage rates. The
caller must freeze origin selection, molecular identities, geometry and horizon,
and split at source boundaries. Invalid states, gaps and overlapping A/B fail.
Origin connectivity is a caller-side criterion; individual return does not
require additional molecules to return simultaneously. Overlapping windows,
patches and repeated molecules are not independent statistical observations.
