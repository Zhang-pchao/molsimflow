# Interfacial state support
`match_caliper_states` first maximizes eligible pair cardinality, then minimizes squared feature differences normalized by the declared calipers. It uses no replacement. Every feature must satisfy its caliper. Empty sets and unmatched states remain explicit; ties can depend on input row order.

The caller supplies physical feature definitions, pre-outcome strata, continuity and time thinning. Do not insert outcomes into matching features or loosen tolerances after seeing a desired contrast. Report unmatched support, feature balance and sensitivity. Matched saved frames are not independent replicas; measured geometric overlap does not establish causal exchangeability or shared microstates.
