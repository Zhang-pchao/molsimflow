# Sampled material-location survival

sampled_region_survival reuses region_recontact_curve risk sets and validation.
Each initially supported location must be supported in every intervening saved frame.
The result includes endpoint recontact and its nonnegative difference from sampled
uninterrupted support, with identical origin weights and denominators. Source seams,
gaps and invalid frames censor entire pairs; repeated labels do not reconnect epochs.
All-origin statistics are conditional on nonempty initial support, not event-start
residence distributions. Overlapping probes and time origins are not independent.
Sampling cannot resolve inter-frame interruptions; threshold crossings need not mean
physical evacuation, detachment, or displacement. No dwell filter is applied.
