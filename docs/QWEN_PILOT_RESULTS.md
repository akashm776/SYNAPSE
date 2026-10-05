# Qwen/GSM8K pilot: completed result

[Overview](../README.md) · [Recorded artifacts](../results/qwen_gsm8k_pilot_v1/README.md)

This is the completed answer-only Qwen2.5-1.5B pilot, originally run in SynLess.
It is not standard chain-of-thought GSM8K benchmarking or evidence about all LLMs.
Three seeds are the replication units; two states and multiple horizons are
repeated measurements, not additional independent seeds.

## Primary held-out result

At 20 continuation updates, averaged over starting states 100/500 within seed:

| Arm | Mean loss delta | Seed SD | Accuracy delta (pp) |
|---|---:|---:|---:|
| Uniform barycenter | −0.001116 | 0.003033 | −0.329 |
| Similarity barycenter | +0.001115 | 0.005776 | −0.190 |
| Hardest existing | +0.001003 | 0.002050 | −0.291 |
| Global rank mixture | +0.001648 | 0.002090 | −0.240 |
| Learned loss reweighting | +0.000136 | 0.002552 | −0.341 |
| Learned barycenter | +0.002876 | 0.000725 | −0.392 |

Deltas are treatment minus matched native. Native exact match averages 11.524%;
uniform averages 11.195%, learned barycenter 11.132%. No arm improves mean primary
exact match. Learned barycenters worsen loss in all six seed–state combinations
and underperform loss reweighting in each seed after averaging states.
These descriptive results do not establish population significance.

## State and horizon

Uniform mixing's mean loss delta is −0.003130 at state 100 and +0.000897 at state
500. However, early test loss improves in only one seed, unlike development's
three-seed early improvement. Mean test accuracy falls at both states
(−0.278 and −0.379 pp). This is a tentative directional resemblance to SCSS,
not a robust early-state win or a proven shared mechanism.

Learned barycenters lower loss after one update at state 100 in all three seeds,
but worsen it after twenty in all three. At state 500 they lower loss after five
updates in all three seeds, while accuracy declines on average; after twenty,
loss worsens in all three. Short-horizon loss improvement does not establish
sustained utility. These are secondary observations, not a new selected endpoint.

## Generator diagnostics

All nine learned-arm fits completed 200 episodes. Learned-barycenter maximum
candidate weights, averaged over each seed's last 20 fitting episodes, were
25.047%, 25.024%, and 25.006% (seeds 789/2026/31415); uniform is 25%.
The generator was already nearly uniform during fitting, not only on fresh B
learners. Median gradient norms across 600 episodes were approximately 1.93e−8
for learned barycenters and 1.60e−8 for loss reweighting. Nonzero gradients and
small magnitudes alone neither prove useful optimization nor diagnose its cause.

Logged cross-layout prompt relative-L2 drift reached 0.314 during fitting.
Same-layout suffix-perturbation causality tests passed in preflight. The drift is
not itself proof of leakage, but motivates controlled numerical/repeatability
checks before attributing near-uniform versus uniform trajectory differences to
learned geometry. Generator histories are published as JSON, without weights.

## Integrity, compute, and next decisions

All 126 checkpoints matched the original development lock during local archive
audit. All 126 test records match the summary and paired arithmetic; each has
1,319 questions and zero token-cap truncations. Development used 747 questions.
Completed development evaluation times sum to 12.18 hours; test times sum to
21.51 hours. Recorded `run` wall time is 50,406 seconds (14.00 hours).
Interrupted report command logs do not account for every session, so their sum
must not be described as complete end-to-end reporting wall time.

The development comparator was loss reweighting, selected only among the
designated non-generating controls. The development gate to Llama was false and
remains false. No larger-model run or changed protocol is justified automatically.
Preserve these outcomes; future diagnostic tuning must not treat this inspected
test split as untouched validation.
