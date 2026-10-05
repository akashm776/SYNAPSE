# Migration from SynLess

SYNAPSE is the active home of language-model work as of October 5, 2026.
SynLess retains CLIP/CUB and the historical LLM snapshot for reproducibility.
The new repository starts with an attributed source snapshot rather than
rewriting the original repository's Git history.

## Source lineage

The six files in `synapse/llm/` were copied byte-for-byte from `synless/llm/` at
[SynLess commit 34f8060](https://github.com/akashm776/SynLess/tree/34f8060c40f5842910ef0dda3d5281ceab0a1f4a).
Relative imports and scientific calculations are unchanged. Tests import the new
namespace. Package metadata, notebook paths, documentation, and the CLI name are
updated. The archived manifest intentionally retains its original repository
commit and runtime identity. It does not claim the pilot ran under SYNAPSE.

| Item | Historical pilot | New SYNAPSE runs |
|---|---|---|
| Repository | `akashm776/SynLess` | `akashm776/SYNAPSE` |
| Python CLI | `python -m synless.llm.cli` | `python -m synapse.llm.cli` |
| Installed command | `synless-llm` | `synapse` |
| Colab checkout | `/content/SynLessLLM` | `/content/SYNAPSE` |
| Default Drive root | `MyDrive/SynLess/runs/llm_qwen_v2` | `MyDrive/SYNAPSE/runs/qwen_v1` |

## Resume the existing pilot

Do not rename or move its Drive files. Use the historical notebook from
[the exact original revision](https://colab.research.google.com/github/akashm776/SynLess/blob/34f8060c40f5842910ef0dda3d5281ceab0a1f4a/colabs/SynLess_LLM_A100.ipynb).
Set its `CODE_REF` explicitly to `34f8060c40f5842910ef0dda3d5281ceab0a1f4a`:
the notebook's historical default otherwise clones current main. Set
`OUTPUT = Path('/content/drive/MyDrive/SynLess/runs/llm_qwen_v2')`.
Recreate the original environment from the manifest, mount Drive, and rerun the
needed command. Do not rerun completed preflight/training just to resume reporting.
Do not bypass an identity mismatch or replace a recorded manifest.

`report` loads `seed_*/branches/step_*/*/horizon_*.pt` and skips existing
`test_*.json` files. There is no within-evaluation per-question checkpoint.
Keep the full Drive folder: the public metric archive is not a resumable run.

## New work

Use the SYNAPSE notebook, record its printed commit, and choose a new output
directory whenever code/config/environment changes. The migration introduces
no new GPU experiment and does not modify the completed pilot or its gate.
