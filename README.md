# Inkling underwriting evaluation

Fifteen three-turn cases test whether a model updates an underwriting finding as evidence changes. Cases use pinned Texas and Massachusetts manuals, an Allianz proposal form and OpenFEMA documentation. Each conversation has deterministic manual search and passage lookup. The scored runner preserves the candidate’s actual answers and tool results across turns.

| Candidate | Cases passed | Pass@4 |
|---|---:|---:|
| Inkling | 9/15 | 60% |
| Inkling 2 Small | 6/15 | 40% |
| Inkling Small | 3/15 | 20% |

[Download the versioned benchmark data and result files on Hugging Face](https://huggingface.co/datasets/mohammed8284/inkling-underwriting-eval/tree/32eaacb180f18351128f91d3e35f22fb139f0364). The results are automated judgments on authored cases. Independent expert agreement and probability calibration were not measured.

## Setup

Use Python 3.12, Git and uv. From a clean checkout:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
uv pip install --no-deps 'git+https://github.com/thinking-machines-lab/tinker-cookbook.git@1991897cba59ea9d9b0030d183d1867b28eba9d9'
python fetch_data.py
python -m unittest discover -s benchmark -p 'test_*.py'
python evaluate.py runs/dry-check
```

Use the platform-specific PyTorch wheel if your platform lacks a CPU-index wheel. The download needs no Hub credentials. It fetches the frozen evidence, tasks and historical summaries. The tests and dry check make no candidate or judge calls.

## Reproduce pass@4

Export TINKER_API_KEY and OPENAI_API_KEY in your shell. The accounts need sampling access to the selected Inkling checkpoint and OpenAI chat-completions access to gpt-6-luna. Select Inkling, Inkling-2-Small or Inkling-Small.

```bash
python evaluate.py runs/full-inkling --model Inkling --execute
```

To use Modal after setup, authenticate the Modal CLI and run:

```bash
uv pip install modal==1.6.4
modal run --detach run_modal.py --model Inkling
```

The launcher prints a trial directory. Retrieve its summary with `modal volume get inkling-underwriting-trials TRIAL_DIRECTORY/summary.json`. For local runs read `runs/full-inkling/summary.json`. Each attempt saves turns, cases and call records below its trial directory. Use a new output directory per paid run. A fresh ledger permits up to $40 in model reservations for full Inkling or $28 for a small model. Hosting is additional; the runner uses frozen cost rates.

Four seeds (20261117–20261120), temperature 0.6, 256K checkpoint variant, 65,536-token candidate input cap and 8,192-token output cap. Each attempt has three user turns. LLM-as-judge Luna evaluates decision, evidence/action and grounding; every required criterion on every turn must pass in the same attempt. A case passes if any of its four attempts passes. The denominator is 15. Incomplete or ambiguous results can withhold the point estimate.

## Harbor task format

After downloading, open `benchmark/harbor/uw-01/task.toml` and its three ordered steps. Each case contains an environment, manual tools and a separate verifier. Native exports use the HTTP rubric verifier configured through GRADER_URL, GRADER_MODEL and GRADER_API_KEY. Published pass@4 results use the custom Tinker BenchmarkBuilder and direct Luna verifier in the reproduction commands above. The native verifier’s mean step reward is a different metric. [Harbor multi-step task format](https://docs.harborframework.com/news/multi-step-tasks).
