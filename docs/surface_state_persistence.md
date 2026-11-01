# Sampled state confirmation

`confirm_core_states` provides causal, identity-resolved confirmation of caller-defined
state cores. The input codes are 0/1 (observed cores), -1 (neutral band) and -2
(missing or ambiguous). State changes require a continuous saved-time span at least
as long as the requested dwell. Five time units at unit cadence require six frames.

Neutral observations retain an established state, but interrupt a pending change.
Missing data, cadence gaps and source seams erase memory; initialization is unknown.
Transitions are reported only when confirmed, never backdated.

Keep raw geometric populations alongside confirmed states. A remembered inside state
can persist while coordinates are outside the region, and must not be interpreted
as physical water occupancy. Likewise, confirmed departure of original identities
does not imply net loss if incoming molecules replace them. This helper neither
sets physical contact cutoffs nor establishes continuous-time lifetimes, independent
event rates, chemical speciation or gas replacement.
