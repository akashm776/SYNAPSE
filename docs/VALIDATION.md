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
