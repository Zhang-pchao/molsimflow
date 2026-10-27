# Fixed-site contact budgets

`spatial_contact_budget` extends interfacial_contact_renewal without changing old APIs.
For before/after masks on fixed reference sites, area change is gained minus lost
area. Let C0 be the old weighted centroid and A1 the new area. The centroid
change is exactly [sum_gain w(x-C0) - sum_loss w(x-C0)]/A1.
It separates forward contact gain from rear contact loss; the latter can shift
the centroid without creating forward contact. It does not assign causality.

Use identical ordered site identities, weights and a common unwrapped coordinate
chart. This reference-site centroid deliberately excludes actual site motion.
Empty endpoint populations retain the area budget but return NaN centroids.
Caller must retain this missing status. Gain/loss depends on sampling interval;
endpoint differences do not count unresolved recrossings or continuous lifetimes.

`contact_graph.components_with_isolated_nodes` adds explicit-node connectivity,
including isolated sites omitted by edge-only summaries. Proximity cutoffs and
chemical labels belong to consuming studies. A proximity edge is not a bond;
finite-cell connectivity alone does not establish macroscopic percolation.
