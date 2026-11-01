# Current oxygen hydrogen identity

This helper reuses the established geometric hydrogen owner definition: nearest
O/C within 1.3 A, periodic minimum-image distances, with O preferred at an exact
O/C tie. Supply each H once by atom ID; do not append surface H to an all-H array.
All current surface and solution O atoms compete with current surface C atoms.

The nominal oxygen H count and its uncertainty are separate outputs. An H is
ambiguous if at least two O/C candidates are within the assignment cutoff and
within 0.1 A of its closest candidate. Every O in that competing set is marked,
including O atoms that did not win the nominal assignment. This extends the
prior O/O near-degeneracy convention to the actual O/C competitor set.

Category precedence is ambiguous, then nominal H count 0, 1, or >=2. With a
whole-main-gas denominator, these mutually exclusive contributions must recover
the accepted O-nearest distance CDF. The count is geometric, not an electronic
bond order. H0 does not prove bridging oxygen; H1 does not establish a hydroxyl
mechanism, adsorption strength, interposed water or sliding dynamics.

For PPF boundaries, pass periodic_z=False. The nonperiodic Z span is expanded with an empty seam wider than twice the ownership cutoff; no accepted owner or ambiguity candidate crosses this seam. Default PPP behavior is preserved.
