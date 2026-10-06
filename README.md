# SYNAPSE

**State-Dependent Synthetic Supervision for Language Models**

[![CPU science checks](https://github.com/akashm776/SYNAPSE/actions/workflows/tests.yml/badge.svg)](https://github.com/akashm776/SYNAPSE/actions/workflows/tests.yml)

[Results](docs/QWEN_PILOT_RESULTS.md) · [Run guide](docs/LLM_RUN.md) · [Protocol](docs/LLM_GENERATOR_PROTOCOL.md) · [Migration](docs/MIGRATION.md) · [Roadmap](docs/ROADMAP.md)

When do synthetic hidden-state negatives help language-model fine-tuning?
SYNAPSE learns barycentric weighting functions through a differentiable learner
update, then tests the frozen functions on separate learners and disjoint data.
It separates **constructing a negative** from **demonstrating that learning from
it helps**.

This is the language-model investigation of the broader
[SCSS research program](https://github.com/akashm776/SCSS), extracted from
[SynLess](https://github.com/akashm776/SynLess). SynLess remains the home of the
CLIP/CUB LESS/BIDS selection pilot. This repository contains only the LLM workflow.

## Current evidence: completed Qwen pilot

Qwen2.5-1.5B **base**, answer-only GSM8K, layer 14, three pipeline seeds,
starting states 100/500, seven arms, continuation horizons 1/5/20.
All 126 official-test evaluations are complete (1,319 questions each).

Primary results at 20 updates, averaged equally over states within each seed:

| Method | Loss delta vs native ↓ | Exact-match delta ↑ |
|---|---:|---:|
| Uniform barycenter | −0.001116 | −0.329 pp |
| Learned loss reweighting | +0.000136 | −0.341 pp |
| Hardest existing | +0.001003 | −0.291 pp |
| Similarity barycenter | +0.001115 | −0.190 pp |
| Global rank mixture | +0.001648 | −0.240 pp |
| Learned barycenter | +0.002876 | −0.392 pp |

**No overall fine-tuning gain is established.** Uniform mixing slightly improves
mean loss but reduces answer accuracy. Learned barycenters worsen loss in all
six seed–state combinations at the primary horizon. The development gate to
Llama failed; no Llama run has been launched.

Uniform mixing's early-help/later-harm mean loss pattern is suggestive, not a
robust cross-domain replication: early test loss improves in only one of three
seeds. The learned weights remain nearly uniform during fitting and reuse.
See [interpretation, seed variability, and numerical caveats](docs/QWEN_PILOT_RESULTS.md)
and the [unaltered result archive](results/qwen_gsm8k_pilot_v1/README.md).

## What is generated?

A weighted mixture of four wrong-answer **hidden representations**, used in an
auxiliary loss alongside correct-answer SFT. The candidates are deterministic
numeric perturbations, not nearest-neighbor answers or generated rationales.
The mixture is not decoded text, an inserted token embedding, or a genuine OT map.
The generator learns mixture weights; it does not learn an on/off gate or strength.

The experiment separates teacher training (**A**), generator feedback (**M**),
fresh-learner training (**B**), development (**D**), and official test reporting.
The base is frozen; LoRA parameters are the learner's trainable weights.
This is not a LESS/BIDS reproduction or a multi-capability benchmark.

## Run

[Open SYNAPSE on Colab A100](https://colab.research.google.com/github/akashm776/SYNAPSE/blob/main/colabs/SYNAPSE_A100.ipynb)

For a new experiment:

```bash
python -m pip install -e '.[llm,test]'
python -m pytest -q
synapse preflight --config configs/llm_qwen_a100.json --output runs/qwen_v1
synapse run --config configs/llm_qwen_a100.json --output runs/qwen_v1
synapse report --config configs/llm_qwen_a100.json --output runs/qwen_v1
```

Use `configs/llm_smoke.json` with a separate output folder for offline CPU checks.
Tiny-model output is implementation evidence, not scientific evidence.
The CLI is also available as `python -m synapse.llm.cli`.

**Compute warning:** the original A100 run took about 14 hours for training plus
development; completed test evaluations consumed another 21.51 hours. Evaluation
is one question at a time. A successful preflight is not an end-to-end runtime estimate.
Completed checkpoint evaluations resume; unfinished evaluations replay in full.

For an existing `SynLess/runs/llm_qwen_v2` run, use the original pinned SynLess
revision and environment, **not this renamed notebook**. See [migration and resume](docs/MIGRATION.md).

## Repository map

- `synapse/llm/`: generators, LoRA, data, functional AdamW, resumable experiments.
- `configs/`: A100 pilot and offline smoke settings.
- `colabs/SYNAPSE_A100.ipynb`: new-run setup and staged execution.
- `tests/`: mathematical, causality, checkpoint, and resume checks.
- `results/qwen_gsm8k_pilot_v1/`: recorded metrics and provenance, no model weights.
- `scripts/check_research_docs.py`: links, notebook syntax, archive arithmetic, checksums.

Next work is numerical and optimization diagnosis, not an automatic larger-model
experiment. Test outcomes are final for this protocol; future experiments need
explicitly separate protocols and evaluation plans.

## Numerical audit and next learnability check

The [A/M-only diagnostic](docs/GENERATOR_AUDIT.md) is implemented separately from
the unchanged pilot engine. It reuses saved teachers to measure repeatability,
bf16/fp32 arithmetic sensitivity, near-uniform perturbations, and tiny fixed-set
generator fitting. No D/test decoding or full retraining is needed.

[Open the audit A100 notebook](https://colab.research.google.com/github/akashm776/SYNAPSE/blob/main/colabs/SYNAPSE_Generator_Audit_A100.ipynb).
Use **audit v2** and a new output folder: v1 ran on A100 but inadvertently cast
Qwen's positional-frequency buffer. V2 preserves buffers and adds explicit
virtual/real update comparisons. The user-supplied v2 A100 run is complete:
repeatability is exact on its 40 comparisons, but bf16 meta/ordinary gradients
differ materially. Fp32 greatly reduces that discrepancy; its tiny fixed-set
fitting changes still do not establish meaningful learning. The original pilot
and prior audit outputs remain unchanged.

The next [fp32 learnability diagnostic](docs/LEARNABILITY_CHECK.md) compares
shared direct logits, per-pair direct logits, and the feature scorer on the same
fixed A/M pairs, with coarse mixture scans and actual-update checks. It is
implemented and CPU-tested, **not yet run on A100**, and does not evaluate D/test.

[Open the learnability A100 notebook](https://colab.research.google.com/github/akashm776/SYNAPSE/blob/main/colabs/SYNAPSE_Learnability_A100.ipynb).
