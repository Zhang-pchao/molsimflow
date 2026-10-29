# Directional local structure descriptors

`interfacial_structure_descriptors.py` measures spatial arrangements of scalar
marks on a periodic material chart. A mark may be a chemical label or a
previously calculated occupation fraction. The caller owns site identity,
surface selection, time windows, units, and the physical meaning of each mark.

`periodic_site_edges(xy, lengths, cutoff)` supplies unique unordered neighboring
pairs with orthorhombic minimum-image planar vectors. The cutoff must be below
half of both box lengths; coincident periodic sites are rejected.

`directional_mark_contrast(marks, edges, vectors, axis=(1,0), weights=None)`
uses squared increments d_ij = (z_i-z_j)^2. With a_ij = cos(theta_ij)^2:

- parallel = sum(w*a*d)/sum(w*a)
- transverse = sum(w*(1-a)*d)/sum(w*(1-a))
- contrast = parallel - transverse
- isotropic = sum(w*d)/sum(w)

The optional weights are edge weights. Returned directional weight sums expose
support. Unsupported values are NaN; a uniform mark with support gives zero.
The final input dimension indexes sites; leading dimensions support batches.

A positive contrast means stronger mark variation along the chosen axis.
For binary chemical marks, isotropic is the unlike-edge fraction. Neither
quantity is a force, barrier, bond strength, or a unique reaction coordinate.
The axis is unoriented: reversing it preserves the result. These descriptors
cannot by themselves distinguish forward from backward driving.

Compare different marks only with explicit normalization and common support.
For composition control, a consuming study may permute mark vectors jointly
among the same selected sites. Joint permutation preserves mark means and
cross-mark covariance while removing spatial arrangement. Retain zero-variance
nulls. Such permutations are descriptive controls, not experimental causal
interventions. Prediction validation must exclude overlapping input/response
windows and test added value above geometric and mean-occupation baselines.
