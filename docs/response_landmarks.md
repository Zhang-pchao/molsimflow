# Descriptive response landmarks

The helpers normalize a response between declared endpoint values, keeping overshoot
and supporting decreasing observables. First upward crossings use linear interpolation
between adjacent finite observations. Initial exceedance, missing-data intervals, and
unreached levels have explicit statuses. These functions do not infer equilibration,
causality, independent replicates, or confidence intervals. Endpoint spread describes
variation within the chosen endpoint bins. Retrospective peak selection and averaging
scales must be recorded by the calling analysis.
