# Nearest-solid chemical identity

`nearest_solid_identity` returns the nearest solid atom ID/type for each
consecutive N2 pair in an orthorhombic PPP box. It preserves the exact minimum
distance. Numerical ties within 1e-12 A select the smallest solid atom ID;
ties between nitrogen partners then use nitrogen order. Tie flags are retained.

An element-resolved CDF contribution uses the complete main-cluster count as
its denominator, including when that element has no selected molecules.
The sum of element contributions must equal the original distance CDF.
Do not normalize each element curve to unity. Nearest oxygen is not automatically
hydroxyl oxygen; that assignment requires current bonding information.

This adds chemical contact identity to a geometric observation. It does not
measure interaction strength, water causality, friction or sliding barriers.
