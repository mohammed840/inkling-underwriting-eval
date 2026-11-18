# Inkling underwriting benchmark

This collection contains 15 native Harbor tasks. Each task has three ordered applicant prompts, a shared environment with source lookup tools, and a verifier for each step.

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

## Verifier

The task configurations pass `GRADER_URL`, `GRADER_MODEL` and `GRADER_API_KEY` to the HTTP rubric verifier. See the [first task configuration](benchmark/harbor/uw-01/task.toml) and the [Harbor multi-step task format](https://docs.harborframework.com/news/multi-step-tasks).
