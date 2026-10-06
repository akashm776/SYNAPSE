# Strength replication on diagnostic-fresh A/M pairs

[Overview](../README.md) · [Preceding two-pair scan](STRENGTH_CHECK.md)

## Question

Does the stronger-auxiliary response persist beyond the two previously inspected
A/M pairs, or is the mean driven by an unusually favorable pair?

The preceding A100 scan completed in 490.4 seconds. At alpha 1, uniform mean M-loss
deltas were −0.000284 and −0.000702 at states 100/500, much larger than its recorded
numerical discrepancies. However, pair 0 drove most of the improvement; pair 1
was harmed at state 500. Some single-candidate controls beat uniform in the mean.
This motivates replication, not selecting alpha 1 as a successful training recipe.

## Fixed default design

- Same saved pipeline seed 789 and teacher states 100/500; **16 new A/M pairs**.
- Continue the source A/M no-replacement permutations with seed 61005 after the
  first two pairs. Exclude both A and M component IDs from those pairs. Fail on
  duplicates, overlap, or insufficient eligible data; never wrap into another epoch.
  Manifest records every included/excluded ID. Selection does not use outcomes.
- **Fresh means new to these diagnostics**, not historically unseen data: examples
  may have appeared in original A teacher training or M generator meta-training.
  This is not a new validation/test split and does not establish generalization.
- Alpha **0, 0.01, 0.1, 1.0**: zero control plus the three positive strengths
  retained after inspection of the prior scan. No additional alpha search, adaptive
  pair selection, stopping on outcomes, or automatic winning-recipe selection.
- Native, uniform, and all four fixed single-candidate vertices. Candidate slots
  are the original deterministic order, not hardness ranks. No edge mixtures or
  learned generator. Every trial performs one restored learner update; no teacher
  continuation accumulates. Same examples are reused across states/strengths.
- Same fp32 arithmetic on bf16-rounded frozen parameters, fp32 adapters/moments,
  preserved model buffers, TF32 off, native clipping and Adam settings.

Only A/M rows reach the model. Full source data is read for its integrity check;
B/D/test are not sampled or evaluated. No decoding, Llama run, generator training,
or original-pilot report rerun. The original engine and previous diagnostic modules
are unchanged and included in the resume identity.

## Measurements and interpretation

Each state/pair/alpha group has eight trials: native twice, uniform, four vertices,
and uniform again. Default **128 groups / 1,024 trials**. Exact zero-strength/native
equivalence, repeated controls, auxiliary/native gradient ratios, clipping, actual
update contrasts, and both virtual/real parity paths are retained.

Report per-state/per-alpha:

- Uniform versus native: mean, median, simulated mean, helpful/harmful/zero pair
  counts, and the range of means obtained by leaving out one pair at a time.
- Uniform versus the **average of four single-candidate post-update losses**.
  These are four separate fixed updates, not a combined update, prediction
  ensemble, or best-candidate oracle. Negative favors uniform. This contrast
  helps separate mixing from simply applying auxiliary supervision.
- All four individual vertex-versus-native contrasts in JSON. Every contrast
  retains its per-pair real/virtual deltas and numerical reference scales.

Per-pair reference = maximum virtual/real loss gap, repeat drift, or eight fp32
epsilons times loss scale. Beyond-reference counts are descriptive heuristics,
not error bounds, confidence intervals or statistical significance. Raw sign
counts include tiny changes. Pair counts are not independent pipeline replications;
states share pairs. Leave-one-out ranges diagnose concentration, not uncertainty
coverage. Interpret clipping and virtual/real agreement alongside the summaries.

A consistent one-step response would justify further study, not a fine-tuning
gain or a learned-generator claim. No selected alpha/recipe is exported. A later
efficacy protocol must be defined separately from these already inspected data.

## Run and resume

[Open the fresh-strength A100 notebook](https://colab.research.google.com/github/akashm776/SYNAPSE/blob/main/colabs/SYNAPSE_Fresh_Strength_A100.ipynb).

Read-only source: `/content/drive/MyDrive/SynLess/runs/llm_qwen_v2`.
Separate output: `/content/drive/MyDrive/SYNAPSE/diagnostics/qwen_fresh_strength_v1`.
No previous diagnostic folder is required; exclusions are reconstructed from the
original data and fixed sampling rule. The complete original pilot is required,
including the provenance-hashed generator checkpoint (not used for inference).
Only load trusted checkpoints.

```bash
python -m synapse.fresh_strength --source /path/to/complete/llm_qwen_v2 \
  --output /path/to/separate/fresh_strength_v1 \
  --config configs/fresh_strength_qwen_a100.json
```

Installed alias: `synapse-fresh-strength`. A100 runtime is **unmeasured**. This
replication is larger than the two-pair scan and may need multiple budget windows.
The 30-minute per-invocation budget is checked between groups, not a hard timeout.
Rerun after `paused`; completed groups are skipped, an interrupted group replays.
Restore the same recorded Git revision/config/environment on runtime restart.
Changed inputs reject resume. A complete run returns without loading a model.
No silent precision/OOM fallback. Never run two Colab VMs into one output folder.

Share `FRESH_STRENGTH_REPORT.md`, `fresh_strength_summary.json`,
`fresh_strength_manifest.json`, and `model_buffers.json` when complete. Per-group
JSON files are the resume state; there are no new model checkpoints. Console
output is also saved. Preserve all original and prior diagnostic artifacts.

CPU tests cover deterministic exclusions, no duplicate components, insufficient
data rejection, outlier-sensitive statistics, the average-single contrast,
source immutability, only fresh A/M model inputs, zero/repeat controls, fp32,
and exact group-level pause/resume. Tiny results are implementation evidence only.
