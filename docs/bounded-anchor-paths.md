# Bounded paths to explicit graph anchors

The contact_graph.bounded_anchor_hops helper returns shortest undirected
distances, parents and source anchors for a supplied finite graph, retaining
isolated nodes and deterministic paths. Supply node ordinals, explicit edges
and seed nodes. The maximum edge count is inclusive; zero returns only seeds.

For water-layer analysis, seeds can be water molecules directly connected to
an eligible surface OH. A distance of one or two describes one or two
water-water edges before the directly bound water, respectively. Paths may
leave a target cohort if the graph population permits it. Specify graph-layer
height, target selection, spatial boundaries, H ownership, bond geometry,
persistent identities and sampling intervals in the consuming contract.

A path is snapshot geometric connectivity. It does not establish continuous
bond survival, binding energy, transmitted force or a causal migration barrier.
Seed absence, graph truncation and chemical ineligibility must be reported.
