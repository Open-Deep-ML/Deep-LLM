# Deep-LLM

**A language model built by everyone and no one.**

Deep-LLM is a tiny GPT whose every component — tokenizer, embeddings, attention,
feed-forward, normalization, architecture, optimizer, LR schedule, train step,
config — is an open **slot** anyone can improve on
[Deep-ML Research](https://deep-ml.com/research/tiny-llm). Fork a slot, beat the
current model on a hidden evaluation, and your code is **merged automatically**.
No maintainers, no review queue: the metric is the maintainer.

`main` is always the current best model. Every experiment anyone ever tried —
merged or rejected — lives in [`experiments/`](experiments/), and every merge is
a commit. The git history *is* the research log.

## Current model

| | |
|---|---|
| **Generation** | 2 |
| **bits per byte** | **2.4878** (hidden test set, lower is better) |
| **Experiments tried** | 7 (2 merged) |
| **Training budget** | 2000 steps / 240s on a T4, ≤15M params |
| **Data** | FineWeb-Edu (educational web text), scored in bits per byte |

## Run it

```bash
pip install torch numpy
python train.py --data your_text_file.txt
```

`train.py` is regenerated from the canonical slots on every merge — this file
IS the model. It trains from scratch (there are no stored weights, by design)
and reports held-out bits per byte.

## The slots

| Slot | Version | Author |
|---|---|---|
| `config` | v0 | Deep-ML |
| `tokenizer` | v1 | moe chabot |
| `embeddings` | v0 | Deep-ML |
| `attention` | v0 | Deep-ML |
| `ffn` | v2 | Nick Grebe |
| `norm` | v0 | Deep-ML |
| `architecture` | v0 | Deep-ML |
| `optimizer` | v0 | Deep-ML |
| `lr_schedule` | v0 | Deep-ML |
| `train_step` | v0 | Deep-ML |

Beat the model and your name replaces one of these rows.

## How merging works (and why you can't cheat it)

Every submission trains in a network-blocked sandbox that physically does not
contain the evaluation data. A separate trusted process — running **zero** user
code — scores the frozen model on hidden splits and speaks the only metric the
platform believes. Beat the canonical by the promotion threshold and a
compare-and-set transaction makes your code the new `main`. The full harness is
published in [`harness/`](harness/) — auditing the referee is encouraged.

## Contribute

→ **[deep-ml.com/research/tiny-llm](https://deep-ml.com/research/tiny-llm)** — fork a slot in the browser,
test in seconds, and submit. Connect GitHub there and merged experiments are
committed here **as you**.

See [`RESEARCH_LOG.md`](RESEARCH_LOG.md) for every improvement so far.
