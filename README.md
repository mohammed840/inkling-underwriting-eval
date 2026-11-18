# Inkling underwriting evaluation

Fifteen three-turn cases test whether a model updates an underwriting finding as evidence changes. Cases use pinned Texas and Massachusetts manuals, an Allianz proposal form and OpenFEMA documentation. Each conversation has deterministic manual search and passage lookup. **Tinker ran the evaluation on Modal. LLM-as-judge Luna graded each turn.** The runner carried the candidate’s actual answers and tool results into the next turn.

| Candidate | Cases passed | Pass@4 |
|---|---:|---:|
| Inkling | 9/15 | 60% |
| Inkling 2 Small | 6/15 | 40% |
| Inkling Small | 3/15 | 20% |

[Download the versioned benchmark data and result files on Hugging Face](https://huggingface.co/datasets/mohammed8284/inkling-underwriting-eval/tree/32eaacb180f18351128f91d3e35f22fb139f0364). The results are automated judgments on authored cases. Independent expert agreement and probability calibration were not measured.

## Browse the 15 Harbor cases

Each row is one conversation. Read the prompts from left to right to follow the evidence as it changes.

| Case | Applicant prompts |
|---|---|
| [01 · Obtain the missing qualifying declination](benchmark/harbor/uw-01) | [Turn 1](benchmark/harbor/uw-01/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-01/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-01/steps/turn-3/instruction.md) |
| [02 · Two letters from the same holding company](benchmark/harbor/uw-02) | [Turn 1](benchmark/harbor/uw-02/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-02/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-02/steps/turn-3/instruction.md) |
| [03 · Outstanding taxes with a documented exception](benchmark/harbor/uw-03) | [Turn 1](benchmark/harbor/uw-03/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-03/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-03/steps/turn-3/instruction.md) |
| [04 · Renovation is a separate eligibility issue](benchmark/harbor/uw-04) | [Turn 1](benchmark/harbor/uw-04/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-04/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-04/steps/turn-3/instruction.md) |
| [05 · Assemble a complete quote submission](benchmark/harbor/uw-05) | [Turn 1](benchmark/harbor/uw-05/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-05/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-05/steps/turn-3/instruction.md) |
| [06 · Apply the limit per building](benchmark/harbor/uw-06) | [Turn 1](benchmark/harbor/uw-06/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-06/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-06/steps/turn-3/instruction.md) |
| [07 · Correct the property's jurisdiction](benchmark/harbor/uw-07) | [Turn 1](benchmark/harbor/uw-07/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-07/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-07/steps/turn-3/instruction.md) |
| [08 · Rehabilitation and the unoccupancy exception](benchmark/harbor/uw-08) | [Turn 1](benchmark/harbor/uw-08/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-08/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-08/steps/turn-3/instruction.md) |
| [09 · Read the tax-delinquency threshold precisely](benchmark/harbor/uw-09) | [Turn 1](benchmark/harbor/uw-09/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-09/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-09/steps/turn-3/instruction.md) |
| [10 · Photos do not replace the required interior inspection](benchmark/harbor/uw-10) | [Turn 1](benchmark/harbor/uw-10/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-10/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-10/steps/turn-3/instruction.md) |
| [11 · Request the referenced engineering report](benchmark/harbor/uw-11) | [Turn 1](benchmark/harbor/uw-11/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-11/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-11/steps/turn-3/instruction.md) |
| [12 · Correct a conflicting flooding disclosure](benchmark/harbor/uw-12) | [Turn 1](benchmark/harbor/uw-12/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-12/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-12/steps/turn-3/instruction.md) |
| [13 · Separate rebuilding values by premises](benchmark/harbor/uw-13) | [Turn 1](benchmark/harbor/uw-13/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-13/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-13/steps/turn-3/instruction.md) |
| [14 · Decode a FEMA property record without inventing a rule](benchmark/harbor/uw-14) | [Turn 1](benchmark/harbor/uw-14/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-14/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-14/steps/turn-3/instruction.md) |
| [15 · Decide whether the public dataset supplies training labels](benchmark/harbor/uw-15) | [Turn 1](benchmark/harbor/uw-15/steps/turn-1/instruction.md) · [Turn 2](benchmark/harbor/uw-15/steps/turn-2/instruction.md) · [Turn 3](benchmark/harbor/uw-15/steps/turn-3/instruction.md) |

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

## Harbor task format

Open the [first task configuration](benchmark/harbor/uw-01/task.toml) to see how Harbor orders the three steps. Each case contains an environment, manual tools and a separate verifier. Native exports use the HTTP rubric verifier configured through GRADER_URL, GRADER_MODEL and GRADER_API_KEY. Published pass@4 results use the custom Tinker BenchmarkBuilder and direct Luna verifier in the reproduction commands above. The native verifier’s mean step reward is a different metric. [Harbor multi-step task format](https://docs.harborframework.com/news/multi-step-tasks).
