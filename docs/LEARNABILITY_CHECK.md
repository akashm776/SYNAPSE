# Fixed-set fp32 learnability check

[Overview](../README.md) · [Preceding numerical audit](GENERATOR_AUDIT.md)

## Question and scope

Can directly optimized mixture weights lower the fixed A/M meta-objective more
clearly than the current feature scorer? This is a small optimization diagnostic,
not another instruction-tuning benchmark. It does not establish generalization.

The completed user-supplied audit v2 was inspected locally: 40/40 repeated
gradient/update/M-loss comparisons were bitwise identical. The bf16 differentiable
versus ordinary gradient discrepancy was 2.36–5.36%; fp32 reduced it to about
0.000125–0.000254%. Fixed-set fp32 fitting changed mean M loss by only −4.17e-7
and −6.26e-7 at states 100/500, respectively. Those tiny changes do not establish
meaningful fitting. These are diagnostic observations from one seed/two pairs,
not proof that numerical effects explain all original pilot outcomes.

## Locked default design

- Pipeline learner seed 789 identifies the saved teacher; states 100 and 500.
- The same two deterministic A/M pairs as audit v2: selection seed 61005.
- Frozen base parameters retain bf16 rounding and are promoted to fp32.
  Adapters, Adam moments, generator parameters, and arithmetic use fp32.
  Model buffers retain their loaded dtype and values. TF32 is off.
- Restore teacher parameters, moments, and RNG before every virtual or real
  learner step. No teacher training is accumulated. Use the original alpha,
  layer, candidate set, inner optimizer, and one-step meta-objective unchanged.
- Three arms start exactly uniform. All get 50 **full-fixed-set** outer steps,
  each averaging the two pair losses before updating. AdamW: LR 0.01, epsilon
  1e-8, zero weight decay, gradient clipping 1. No LR search or best-step selection.
  These settings differ from audit v2's 20 alternating-pair steps at LR 0.001;
  direct comparisons between audits are not matched-budget comparisons.

| Arm | Trainable parameters | Meaning |
|---|---:|---|
| shared_logits | 4 | One softmax over raw candidate slots shared across pairs |
| pair_logits | 4 per pair (8 total) | Independent softmax for each fixed pair; intentionally allows memorization |
| feature_scorer | 193 | Original shared 4→32→1 tanh scorer; zero output head |

Candidate slots are the original deterministically shuffled wrong answers, **not
hardness ranks**. The shared-logit control therefore differs from the original
`global_rank_mixture`. Pair logits are an overfitting diagnostic with greater
pair-specific freedom, not a model that can be deployed on unseen examples.
Softmax has a redundant additive-logit degree of freedom. Equal steps/settings
do not equalize model capacity or optimize the learning rate for each arm.

## Measurements and interpretation

Before fitting, scan uniform, four simplex vertices, and six edge midpoints on
each pair. Repeat uniform once. This coarse scan tests substantial weight changes
without depending on the outer optimizer; it is not exhaustive simplex search.
The per-pair minimum is selected on M itself, not held-out validation.

For each scanned mixture and each arm's initial/final weights, record:

- Actual post-AdamW M loss and full differentiable-path virtual M loss.
- Virtual/real gradient and parameter-update discrepancies using the corrected
  audit's `same_gradient` and `meta_path` checks.
- Weights, candidate-feature variation, auxiliary/native gradient ratios.

During fitting, log per-pair loss/weights, meta-gradient norm, and actual parameter
movement at every step. Assess final weights on the same A/M pairs. Report both
per-pair and equally averaged losses; pairs/states/steps are not seed replications.

The report uses a **descriptive numerical reference scale**: the maximum of
observed virtual/real M-loss gaps, repeated-uniform real-loss drift, and eight
fp32 epsilons times the largest absolute loss (at least one). A real reduction
exceeding that scale is a screening flag, **not** an error bound, significance
test, convergence certificate, or generalization claim. The factor eight is a
documented heuristic. A tiny negative delta alone is insufficient evidence.

If pair logits improve clearly while the scorer does not, investigate scorer
features/parameterization and optimizer dynamics. If coarse mixtures improve but
learned weights do not, investigate optimization and derivative reliability.
If none improves, the conclusion is limited to these states/pairs and this
intervention—not that synthetic supervision is universally ineffective.

## Run on Colab

[Open the learnability A100 notebook](https://colab.research.google.com/github/akashm776/SYNAPSE/blob/main/colabs/SYNAPSE_Learnability_A100.ipynb).

Read-only source: `/content/drive/MyDrive/SynLess/runs/llm_qwen_v2`.
New output: `/content/drive/MyDrive/SYNAPSE/diagnostics/qwen_learnability_v1`.
It does **not** need the audit v2 output directory. It does need the full original
pilot data/metadata, teacher checkpoints, and saved barycenter checkpoint (the
shared audit loader checks its provenance, though it is not used for fitting).
Only load trusted checkpoints.

```bash
python -m synapse.learnability --source /path/to/complete/llm_qwen_v2 \
  --output /path/to/separate/learnability_v1 \
  --config configs/learnability_qwen_a100.json
```

Equivalent installed entry point: `synapse-learnability`.
Do not rerun original pilot preflight, run, or report. No B/D/test rows reach the
model; the complete prepared data file is read only to verify its integrity and
extract A/M. No answer decoding, new benchmark, or Llama launch occurs.

The per-invocation 30-minute budget is checked between work units, not a hard
timeout. A scan of one pair, a full-set outer step, or an assessment can overrun
the boundary. This new diagnostic has **not yet run on A100**; the prior audit's
3.9-minute duration is not a timing guarantee. No silent OOM precision fallback.

Resume with the same recorded Git revision, configuration, environment, source,
and output. Completed scans and arm reports are skipped; fitting saves optimizer
and generator state after every full-set step, including a step-zero checkpoint.
An interrupted unsaved unit replays. A completed run returns without reloading
the model. Changed inputs/code/environment reject resume. `--max-units` supports
deterministic pause testing. Output must be separate from—not inside or above—the
original source run. A local process lock does not coordinate separate Colab VMs;
do not run two sessions into one output folder.

Share `LEARNABILITY_REPORT.md`, `learnability_summary.json`,
`learnability_manifest.json`, and `model_buffers.json`. Keep `state_*/fit_*.pt`
for resume. The summary includes the scans and per-step histories; checkpoints
need not be shared. Download after the manifest says `complete`.

## Verification

Offline tests cover live second derivatives to direct logits, raw-slot/pair
semantics, full-set step accounting, exact interrupted-versus-uninterrupted
results, model precision, virtual/real parity, source immutability, B/D/test
exclusion, buffer guards, and resume identity checks. The original six-file LLM
engine and archived pilot artifacts remain unchanged.
