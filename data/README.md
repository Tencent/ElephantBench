# Dataset

`elephantbench.jsonl` contains 1,094 records in JSON Lines format, with one question per line.
The same release is mirrored at
[Tencent/ElephantBench](https://huggingface.co/datasets/Tencent/ElephantBench).

Load the hosted test split with:

```python
from datasets import load_dataset

dataset = load_dataset("Tencent/ElephantBench", split="test")
```

## Example

The text below is shortened for readability, but its keys and nesting match the dataset exactly.

```json
{
  "benchmark_id": "4382fd6d-ec5d-5ec8-a3f2-8abc706fa010",
  "item_group_id": "c9ef1c87-3e50-5190-8e8e-37a82ec25634",
  "eval": {
    "question": "What birth date is reported for Mother Teresa ...?",
    "gold_answers": [
      {"value": "August 26, 1910"},
      {"value": "August 27, 1910"}
    ],
    "preferred_answer": "Sources report August 26 and August 27, 1910."
  }
}
```

## Record structure

| Field | Description |
| --- | --- |
| `benchmark_id` | Canonical UUID identifying one question formulation. |
| `item_group_id` | Canonical UUID grouping formulations derived from the same reviewed conflict. Every group contains at least two records, but group sizes are not otherwise fixed. |
| `eval.question` | Question sent to the target model. |
| `eval.gold_answers[].value` | Verified accounts used by the judge; complete recall requires covering all of them. |
| `eval.preferred_answer` | Concise human-readable reference response. |

The target model receives only `eval.question`. Gold answers are supplied to the judge after the response has been saved. Complete, partial, and failed recall are assigned mechanically from per-gold coverage and material contradictions.

The release contains 1,094 unique normalized question texts. Every record has at least two verified answers. Source pages and review traces are not distributed with the benchmark and are never passed to evaluated models. The dataset is intended for research on factual recall and memory completeness; scores should not be interpreted as a general measure of model quality or source credibility.
