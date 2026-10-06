# Research roadmap

[Overview](../README.md) · [Completed pilot](QWEN_PILOT_RESULTS.md)

## Completed

- Seven-arm LLM implementation, mathematical checks, same-shape causality checks,
  exact tiny-model resume tests, and A100 preflight.
- Three Qwen/GSM8K pipeline replications with disjoint A/M/B/D partitions.
- Locked development comparator and all 126 official-test evaluations.
- Archived negative/mixed results and migration from SynLess to SYNAPSE.

## Diagnostics completed; fresh-pair replication implemented

The corrected [bounded generator audit](GENERATOR_AUDIT.md) has run on A100.
Its bitwise-repeatable outputs show substantial bf16 meta/ordinary-path
differences and much smaller fp32 discrepancies, but no convincing fixed-set
generator fit. This does not establish the cause of every pilot outcome.

The next [fp32 learnability check](LEARNABILITY_CHECK.md) is implemented with
offline correctness/resume tests and a separate A100 notebook. It compares free
shared/per-pair logits with the original feature scorer and scans coarse mixtures.
Its A100 run is complete: weights moved, but all six mean real-loss reductions
were below their diagnostic numerical reference scales. Neither diagnostic
alters the completed pilot.

The [fixed-mixture strength check](STRENGTH_CHECK.md) is the next bounded test:
vary alpha while measuring actual and simulated updates, with native/uniform
controls and no generator fitting. Its A100 run is complete: alpha 1 produces
measurable loss changes, but mean improvements are dominated by one pair and
do not establish a mixture advantage.

The [fresh-pair replication](FRESH_STRENGTH_CHECK.md) excludes both inspected
pairs and repeats fixed native/uniform/single-candidate comparisons on 16 pairs.
It reports medians, sign counts and leave-one-out means, with no generator fitting
or D/test evaluation. Its A100 outcomes remain pending.

- Audit near-uniform generator fitting: feature variation, parameter changes,
  gradient scales, and sensitivity of the functional update.
- Compare controlled precision and batch-layout conditions on training/meta data.
- Test exact-uniform versus near-uniform update repeatability at matched states.
- Define a new evaluation plan before any tuning or follow-up utility claim.

No promise that these diagnostics will turn the negative result into a positive
one. They should discriminate implementation/numerical limitations from weak
learning signals and an ineffective intervention.

## Deferred hypotheses

- State-dependent on/off or strength control; the current generator only mixes.
- Multiple layer taps, genuine learned OT, and multiple capabilities.
- LESS/BIDS selection over a frozen recipe bank with separate selection data.
- Longer continuations and equal-compute controls.
- Llama method replication or frozen-function transfer. The current development
  gate failed; neither is automatically authorized or launched.
