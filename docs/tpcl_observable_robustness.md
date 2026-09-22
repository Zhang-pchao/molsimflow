# TPCL observable robustness

`molsimflow.postprocess.tpcl_observable_robustness` is the final bounded
existing-trajectory audit after Stage A observable qualification.

It performs two pre-registered checks:

1. extend synthetic detector injections to `4-8 A` and `2-10 ps`, while
   retaining the smaller Stage A grid and all five motion modes; and
2. repeat the 20 ps high-cadence region analysis with one-at-a-time changes to
   contact height (`4 A`, `6 A`), edge-tail fraction (`0.20`), and O-O cutoff
   (`3.2 A`).

The accepted `5 A / 0.10 / 3.5 A` Stage A result is consumed as the primary
configuration and is never rewritten. The new output is also immutable.

```bash
PYTHONPATH=/path/to/code_snapshot \
python -m molsimflow.postprocess.tpcl_observable_robustness \
  --package-root /path/to/four_interface_tpcl_high_frequency_force_on_v1_r3 \
  --reference-analysis /path/to/high_frequency_tpcl_mechanism_v3 \
  --primary-analysis /path/to/tpcl_observable_qualification_v1 \
  --output-dir /path/to/fresh/tpcl_observable_robustness_v1
```

The audit records detector recovery, membership-confound fractions, paired
metric sign sensitivity, and 5 ps block stability. Parameter robustness cannot
turn a single parent history into independent evidence or establish causality.
