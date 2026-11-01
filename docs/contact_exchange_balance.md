# Contact exchange balance
contact_exchange_balance.py provides general, pure array helpers. States are outside-main, main-far, main-near/noncontacting, and main-contacting. Input contact masks remain authoritative. contact_exchange returns the full transition matrix and asserts both endpoint populations and exact contact closure.

The identity Delta Nc = G - L + I - O separates contact changes among retained main members from contact-bearing membership entry/exit. All counts depend on the sampling interval and the caller's fixed contact definition. Repeated threshold crossings are not independent reactions.

fit_contact_probabilities estimates two empirical per-sample probabilities from supplied training transitions. predict_contact_increment uses observed current populations and excludes membership flux. It is a conditional one-step predictor; summing it does not create a physically constrained free-running trajectory. Constant probabilities require external validation. No continuous-time rates, Markov property, barrier, mechanical pinning or chemical causality is implied.

The caller owns branch boundaries, contiguous blocked train/gap/test splits, sampling interval, uncertainty and baseline comparisons. Tests check exact channel separation, manual fitting, missing exposure and invalid membership. Case-specific paths and split policy stay outside this module.
