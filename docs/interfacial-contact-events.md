Saved contact events and periodic exterior
=========================================

interfacial_contact_events supplies gas-only saved-state labels and four-neighbor periodic void classification. A winding void is distinguished from a finite cavity merely crossing a periodic boundary. The supplied occupied mask is unchanged. Consumers must distinguish gas projection from surface contact.

saved_contact_states uses k saved configurations before and after each center: [t-k,t) and [t,t+k). Gain/loss require uniformly absent/present sampled contact; stable dry/wet and mixed states remain explicit. Five samples at 5 ps span 20 ps, with a 5 ps gap between the last pre and first post sample. Unsaved recrossings are unresolved. Missing native steps invalidate a window.

complete_window_centers separately admits symmetric antecedent/follow-up windows. Define spatial eligibility from pre-event gas geometry before inspecting water/H-bond variables. Retain unsuccessful and mixed cases and record support. Neighboring sites and overlapping windows are not independent replicas. Covariate overlap and temporal ordering do not establish causation.

Persistent-state intervals
-------------------------

persistent_state_intervals accepts saved binary states plus unknown (-1), native steps, and optional segment identities. Uniform runs must meet the requested persistence. Missing samples or explicit boundaries reset history. Transitions retain the last old-state sample, first new-state sample, and separate confirmation sample; the intervening uncertainty is never discarded for being long. Same-state qualified runs refresh the last evidence. All qualified runs are returned, including censored endpoints. Comparing two processes requires retaining overlapping brackets as unresolved and multiple matches as ambiguous. This interface does not estimate continuous residence or causality.
