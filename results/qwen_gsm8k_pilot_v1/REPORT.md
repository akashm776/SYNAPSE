# LLM generator pilot report

Real-model pilot.

Loss deltas are treatment minus matched native; negative is helpful.
Primary horizon: 20; equal mean over states [100, 500] within each seed.

| Arm | Mean loss delta | Seed SD |
|---|---:|---:|
| native | +0 | 0 |
| uniform_barycenter | -0.0011164819 | 0.0030333599 |
| similarity_barycenter | +0.0011145716 | 0.0057757533 |
| hardest_existing | +0.0010031774 | 0.0020500503 |
| global_rank_mixture | +0.0016478717 | 0.0020900784 |
| learned_loss_reweighting | +0.00013633048 | 0.0025520268 |
| learned_barycenter | +0.0028757661 | 0.00072538763 |

See summary.json for every seed/state/horizon and exact-match/numeric-NLL outcomes.
Development chooses the non-generating comparator before this report.
Comparator: learned_loss_reweighting. Development Llama advancement gate: False.
No automatic Llama launch. No OT-map, multi-capability balance, or compute-efficiency claim.
Meta-training and repeated evaluation costs are recorded separately in checkpoints and metric artifacts.
