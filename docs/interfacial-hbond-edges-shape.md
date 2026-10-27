# Identity-resolved interfacial hydrogen-bond graphs and planar shape

The existing hydrogen-bond distance and donor-angle convention is retained.
`interfacial_contact_renewal.water_surface_hbond_edges` returns water index,
surface-site index and donor kind (0 water, 1 surface). Both directions may
exist for the same O pair; the rows are directed, not unique pair counts.
`shared_water_site_bridges` projects distinct shared-water identities to site
pairs, collapsing duplicate directions. `interfacial_water_hbond.hydrogen_bond_pair_edges`
returns unique unordered pairs for eligible water-water or surface-surface O.
These APIs do not assign chemical eligibility: callers supply current H
ownership and retain original identity maps, exclusions and ambiguous states.

A one-water bridge, a two-water path, and direct surface OH-OH bonding are
different structures. Proximity of original oxygen sites alone establishes
none of them. Connected components describe the supplied finite graph; a
large component is not a thermodynamic percolation transition or a barrier.
Endpoint recurrence of an atom pair is not continuous bond lifetime.

`interfacial_gas_kinematics.planar_shape_tensor` measures XY covariance on a
consistently unwrapped molecular cluster. It reports aspect ratio, principal
axis, anisotropy and Y variance fraction. PBC unwrapping and compactness belong
to the caller. Near isotropy makes orientation unstable; do not claim a
preferred direction merely from its returned eigenvector. Contact footprint
shape and gas-body covariance must be kept separate.

Units, spatial regions, thresholds, temporal support and comparison controls
belong in consuming-project contracts. No HB count measures bond energy, and
no gas-site contact count measures adsorption energy or detachment free energy.

A periodic neighbor graph must also be checked for winding cycles.
The periodic_cluster_winding helper distinguishes ordinary boundary crossing
from nonzero periodic cycle shifts. In a winding cluster, finite-cluster
shape moments must be withheld; the effect is cutoff-dependent, not proof
of a macroscopic phase transition.
