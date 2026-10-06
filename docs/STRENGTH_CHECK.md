# Fixed-mixture auxiliary-strength sensitivity

[Overview](../README.md) · [Previous learnability diagnostic](LEARNABILITY_CHECK.md)

## Question and design

Does increasing the auxiliary coefficient make mixture composition have a clearer
effect on an actual learner update and its A/M meta-loss? The completed learnability
check moved weights away from uniform, but its six mean loss reductions remained
below the numerical reference scales. This does not prove the auxiliary is too weak.

- Read the completed original pilot, without altering it. Use the same pipeline
  seed 789, teacher states 100/500, two A/M pairs and selection seed 61005.
- Alpha grid: **0, 0.001, 0.01, 0.1, 1.0**. Original Qwen alpha is 0.01.
  Alpha 1 is a stress condition, not a recommended training setting. The CLI
  requires zero and the source alpha, caps the grid at one, and rejects duplicates.
- Eleven fixed recipes: uniform, four vertices (single candidates), and all six
  edge midpoints (equal pairs). These use raw deterministic candidate slots, not
  hardness ranks. No fitted weights or selected prior winners are used.
- Restore teacher LoRA parameters, Adam moments and RNG before every trial.
  Apply one ordinary clipped AdamW update on A; measure native correct-answer loss
  on M. Updates never accumulate. Layer, candidates, inner LR and clipping stay fixed.
- Frozen parameters retain bf16 rounding and are promoted to fp32. Model buffers
  retain their original dtype/values; adapters/moments are fp32 and TF32 is off.
  The original engine and previous diagnostic code remain unchanged.

Alpha changes only in a copied in-memory config, not the source protocol. No
generator fitting, B continuation, D/test evaluation, answer decoding, or Llama
launch occurs. The full prepared data is read for its integrity check, but only
A/M is sampled or evaluated. The provenance loader also hashes the saved
barycenter checkpoint; its weights are not used. Only load trusted checkpoints.

## Measurements

Each state/pair/alpha group has **14 trials**: native twice, eleven mixtures, and
uniform once more. Default: 20 groups, 280 trials. Alpha-zero recipes exercise the
actual learned-arm early exit and must exactly match native loss/update vectors.
No measured weights/features are fabricated when the auxiliary is skipped.

Each trial records actual M loss and deltas versus native and same-alpha uniform;
differentiable virtual M loss and corresponding deltas; virtual/real parity for
the same-gradient and meta-gradient paths; unclipped auxiliary/native gradient
ratio; total gradient norm, clipping factor, actual update norm and vector
contrasts. Native/uniform repeats record gradient/update and loss differences.

The report averages each **fixed recipe across pairs first**, then displays the
range across recipes. It does not average per-pair winners. All per-pair losses
and eleven recipe aggregates remain in JSON. Native is recomputed in every group.
Relative auxiliary-gradient errors against a zero/tiny reference are ill-conditioned;
use absolute differences too.

The numerical reference is the maximum observed virtual/real loss gap, repeat
drift, or eight fp32 epsilons times loss scale (at least one). It is a descriptive
heuristic, **not an error bound or statistical test**, and does not bound a
treatment-minus-control difference. Interpret loss changes alongside clipping
and actual updates. More auxiliary gradient need not mean useful supervision.

Only one seed/two reused A/M pairs are tested. Sensitivity does not establish
generator learnability, generalization, or fine-tuning gains. **No winning alpha
or recipe is selected.** A later efficacy experiment needs a separate protocol.

## Run and resume

[Open the strength A100 notebook](https://colab.research.google.com/github/akashm776/SYNAPSE/blob/main/colabs/SYNAPSE_Strength_A100.ipynb).

Source: `/content/drive/MyDrive/SynLess/runs/llm_qwen_v2`.
New output: `/content/drive/MyDrive/SYNAPSE/diagnostics/qwen_strength_v1`.
No previous diagnostic output is needed. Do not rerun old training/reporting.

```bash
python -m synapse.strength --source /path/to/complete/llm_qwen_v2 \
  --output /path/to/separate/strength_v1 --config configs/strength_qwen_a100.json
```

Installed alias: `synapse-strength`. The user-supplied A100 run completed in
**490.4 seconds (8.2 minutes)**; this is not a timing guarantee. It found much
larger effects at alpha 1, but one pair dominated the mean improvement. See the
[fresh-pair replication](FRESH_STRENGTH_CHECK.md) for the next bounded test.
The 30-minute budget is per invocation, checked between groups, not a hard timeout.
Completed groups are skipped on resume; an interrupted group replays. Record and
restore the exact Git revision/environment. Changed inputs/code/settings reject
resume; completed runs return without loading a model. Output must be separate
from the source. Do not run two Colab VMs into one output folder. No silent OOM
precision fallback. `--max-units` provides deterministic pause testing.

Share `STRENGTH_REPORT.md`, `strength_summary.json`, `strength_manifest.json`,
and `model_buffers.json` after status becomes `complete`. The notebook saves
`console.log` too. Per-group JSON files are the resumable work; there are no new
model checkpoints. Preserve the original pilot and previous diagnostic folders.

Offline tests cover settings, zero-strength/native equivalence, repeated controls,
auxiliary-gradient scaling, teacher restoration, source immutability, fp32/A/M-only
inputs, exact resume and original-strength inclusion. Tiny results are implementation
checks only.
