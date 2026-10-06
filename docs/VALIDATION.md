# Validation

The migrated LLM engine is byte-identical to SynLess commit
`34f8060c40f5842910ef0dda3d5281ceab0a1f4a`; its namespace is now `synapse.llm`.
The archive inventory checks source-file hashes as well as recorded result bytes.

The 14 inherited LLM tests cover numeric parsing, masks and splits, deterministic
sampling, generator permutation behavior and live gradients, functional AdamW
parity, finite differences, virtual-step non-mutation, optimizer restore,
interrupted meta/continuation resume, development locking, and prompt causality.
Tiny fp64 reference tests replace Qwen operations that internally downcast to
fp32 only in the fixture. Production code is unchanged. Mixed-precision
connectivity checks are not proofs of numerical accuracy or utility.

Run `python -m pytest -q` and `python scripts/check_research_docs.py`. CI additionally
runs the complete offline preflight/run/report workflow with no downloaded model
or data. Scientific results come from the separately archived real A100 pilot,
not the smoke run. See [results](QWEN_PILOT_RESULTS.md).

## Separate generator audit

The audit adds six tests for output-path/budget safety, exact-uniform identity
and live gradients, controlled precision promotion, and end-to-end resume with
source immutability and reporting exclusion. Its smoke command is also exercised
in CI. V2 regression checks preserve buffer dtype/values through repeated
precision switches, bound diagnostic cosines, and compare direct virtual/real
updates. The v1 A100 run exposed numerical sensitivity but also a buffer-casting
bug in the audit. Corrected v2 A100 outputs have now been inspected: 40/40 exact
repeat comparisons and much smaller fp32 virtual/real discrepancies, but tiny
fixed-set fitting changes. The original six-file engine and archive are unchanged.

## Separate fp32 learnability check

Five additional tests cover settings, raw-slot/per-pair semantics, finite-difference
verification of live second derivatives to direct logits, full-set gradient
averaging before the outer optimizer step, and exact interrupted fitting resume.
End-to-end guards check source immutability, A/M-only model inputs, fp32 parameters,
virtual/real parity, changed-identity rejection, and completed-run no-op resume.
The independent CLI smoke also runs in CI. These are implementation checks;
the completed A100 run showed weight movement but no mean reduction exceeding
its numerical reference scales.

## Separate auxiliary-strength check

Strength checks cover alpha-grid safety, fixed recipes, native/uniform repeats,
actual zero-alpha learned-arm equivalence, linear scaling of unclipped auxiliary
gradients, A/M-only fp32 model inputs, source immutability and exact group-level
resume. The independent CLI smoke runs in CI. The new strength diagnostic's
A100 outcomes remain pending; no alpha or recipe is automatically selected.
