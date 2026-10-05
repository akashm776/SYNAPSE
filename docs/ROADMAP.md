# Research roadmap

[Overview](../README.md) · [Completed pilot](QWEN_PILOT_RESULTS.md)

## Completed

- Seven-arm LLM implementation, mathematical checks, same-shape causality checks,
  exact tiny-model resume tests, and A100 preflight.
- Three Qwen/GSM8K pipeline replications with disjoint A/M/B/D partitions.
- Locked development comparator and all 126 official-test evaluations.
- Archived negative/mixed results and migration from SynLess to SYNAPSE.

## Next diagnostics — proposed, not run

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
