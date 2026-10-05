# Recorded Qwen/GSM8K pilot

This archive records the completed `llm_qwen_v2` run, produced with
[SynLess commit 34f8060](https://github.com/akashm776/SynLess/tree/34f8060c40f5842910ef0dda3d5281ceab0a1f4a),
before the SYNAPSE migration. Original reports, summaries, manifest, protocol,
preflight, development lock, and command history are copied unchanged.

[Original report](REPORT.md) · [All test results](summary.json) ·
[Development results](development_summary.json) · [Interpretation](../../docs/QWEN_PILOT_RESULTS.md)

`archive_inventory.json` records SHA256 hashes of published original artifacts,
their source revision, migrated source files, and the checkpoint audit.
`generator_histories/` contains JSON exports of all nine fits: episode diagnostics,
not parameters, optimizer states, or a substitute for resumable checkpoints.

All 126 test evaluations and all 126 development evaluations are included in
the summaries. All 126 checkpoint hashes were verified against the downloaded
weights during migration; weights are deliberately not published here. Neither
raw questions/answers, tokenized data, credentials, nor model files are included.
Keep the original complete Drive run for resume and deeper numerical analysis.

The inspected official test split is not a fresh tuning set. Three seeds and a
single answer-only task cannot establish general synthetic-supervision benefits.
