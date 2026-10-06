# Generator-learning and numerical-stability audit v1

[Overview](../README.md) · [Original pilot results](QWEN_PILOT_RESULTS.md)

This is a bounded diagnostic, not a new efficacy benchmark. The completed pilot
and its scientific code are unchanged. The audit lives in `synapse/audit.py`,
outside the source-hashed historical `synapse/llm/` engine.

## Questions

1. Do repeated identical updates reproduce gradients, parameter changes, and M loss?
2. How do learned and slightly nonuniform weights differ from exactly uniform mixing?
3. How sensitive are these quantities to bf16 versus fp32 arithmetic?
4. Can a freshly initialized generator reduce its virtual-update loss on a tiny
   fixed A/M set? This deliberately tests fitting, not generalization.

## Fixed default design

- One pipeline seed (learner ID 789), saved **teacher** states 100 and 500.
- Two fixed, distinct A examples paired with two fixed, distinct M examples;
  deterministic selection seed 61005, selected without inspecting outcomes.
- Native, exactly uniform, saved learned barycenter, and two fixed perturbations:
  `[.25 + epsilon, .25 - epsilon, .25, .25]`, epsilon 0.0001 and 0.001.
  Candidate positions are the original deterministic order, not hardness ranks.
- Two restored-state repeats per condition, in bf16 and fp32: 80 real one-step
  updates total. Each update is discarded; teacher weights and AdamW state reset.
- Four fresh-generator fitting jobs (two states × two precisions), 20 steps each,
  cycling those same two A/M pairs. Original one-step meta-objective, AdamW outer
  LR .001, gradient clipping 1. No hyperparameter search or adaptive stopping.
- A 1,800-second budget **per invocation**, checked between work units. It is not
  a hard preemptive timeout: setup or an in-progress group/assessment may overrun.
  A repeatability unit includes both precisions and all conditions for one pair.

Only A/M records reach the model. No B continuation, D/test evaluation, answer
generation, or Llama launch occurs. The source `data.json` is read and its full
hash checked for integrity; B/D/test contents are not sampled, evaluated, or
exported. The A/M set is intentionally reused for before/after fitting diagnostics.

## Precision interpretation

Both arms use the **same bf16-rounded frozen base weights**. The fp32 arm promotes
those values to fp32; adapters and moments remain fp32 in both. This isolates
arithmetic precision without substituting a different underlying pretrained base.
It does not recover the original full-precision weights or implement a fp64
reference. Eager attention, TF32 off, and deterministic algorithms are recorded.
Same-shape future-token causality is checked at each state/pair/precision.

The source manifest's environment is retained; the audit records its own runtime
and locks it on resume. Differences from the source environment are not evidence
of exact replay. Repeatability is assessed within the audit's recorded runtime.
The source protocol and six-file scientific-engine hash must match the pilot.

## Outputs and interpretation

`audit_summary.json` and `AUDIT_REPORT.md` summarize:

- Gradient/update norms and absolute, relative, cosine, and bitwise comparisons.
- Repeated-trial differences; learned/perturbed versus uniform; fp32 versus bf16.
- Actual post-update correct-answer M loss and differences versus controls.
- Per-example weights and candidate-feature variation. The shared q–p feature
  is constant across candidates by construction; zero variation there is expected.
- Fixed-set virtual M loss before/after fitting; per-step meta-gradients, actual
  generator parameter changes, and weight trajectories.

The weighted auxiliary gradient is measured as total minus native gradient on
the same restored state. It may itself be sensitive to floating-point subtraction;
native trials supply a zero-auxiliary control. Relative differences are unstable
when the reference norm is tiny, so use the absolute norms too. Cross-precision
M-loss differences include precision effects on evaluation, not just the update.
Interpret within-precision treatment-minus-native comparisons alongside them.

No automatic pass/fail threshold declares a scientific explanation. A lower
fixed-set loss is an optimization diagnostic, not transfer. Twenty steps without
improvement do not prove the generator cannot learn. Nonzero gradients do not
establish useful learning. The original inspected test set must not be treated
as untouched validation for recipes chosen from these diagnostics.

## Run on Colab

[Open the audit notebook](https://colab.research.google.com/github/akashm776/SYNAPSE/blob/main/colabs/SYNAPSE_Generator_Audit_A100.ipynb).
It reads your complete original Drive run at
`/content/drive/MyDrive/SynLess/runs/llm_qwen_v2` and writes separately to
`/content/drive/MyDrive/SYNAPSE/audits/qwen_generator_v1`.
Do not rerun the original pilot preflight, training, or report.

```bash
synapse-audit --source /path/to/complete/llm_qwen_v2 \
  --output /path/to/separate/audit_v1 --config configs/audit_qwen_a100.json
```

Equivalent: `python -m synapse.audit ...`. Requires the full source `data.json`,
manifest/protocol/preflight, selected teacher checkpoints, and saved learned
barycenter—not merely the public result archive. Only load trusted checkpoints.

After a pause/restart, use the same audit code/config/runtime/output and rerun.
Completed pair groups are skipped; fitting saves generator/optimizer state every
step. An interrupted pair group or unsaved fit step replays. `--max-units N`
provides a deterministic pause for testing. Source changes or audit identity
changes reject resume. Output cannot equal, contain, or lie inside the source
run. A local process lock prevents same-machine concurrent writers; do not run
two Colab sessions against one audit folder.

The script does not silently lower precision/batch size after an OOM. Preserve
the failed audit and revise the diagnostic configuration in a new output folder.
Real A100 audit feasibility/runtime remain to be measured; the old pilot's
preflight was not a resource check of this new fp32 audit.

## Offline smoke

First create the tiny source with the normal `llm_smoke.json` preflight/run/report,
then use `configs/audit_smoke.json` against that source. Tests cover source
immutability, reporting exclusion, corrupted-data rejection, exact uniform
identity, precision promotion, and interrupted fitting resume.
