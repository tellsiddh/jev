# open-jev

Natural language inference with [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev)
(a Qwen3.5-4B cross-encoder), plus a zero-shot model router built on top of it.

Two pieces:

- `open_jev.py` loads the model and scores premise/hypothesis pairs as
  contradiction / entailment / neutral.
- `router.py` uses those scores to pick which LLM should handle an incoming
  request, with no training data and no labelled examples.

## Setup

```bash
pip install -r requirements.txt
```

The weights (~9 GB) download from the Hugging Face Hub on first run and land in
your HF cache. Everything runs locally after that.

## NLI

```bash
python open_jev.py            # demo, then an interactive prompt
```

```python
from open_jev import nli, rank

nli("The bird is 0.05 below the centre of the gap.",
    "The bird is below the centre of the gap.")
# [{'contradiction': 0.0045, 'entailment': 0.9848, 'neutral': 0.0106}]
```

The model has exactly three trained classes and it does not answer questions. To
choose among candidate answers, score each candidate as its own hypothesis and
compare entailment:

```python
rank("There is a demon to your right and a demon to your left.",
     ["You go left.", "You go right.", "You go straight ahead."])
# [('You go right.', 0.064), ('You go left.', 0.045), ('You go straight ahead.', 0.034)]
```

All three scores are near zero, correctly: nothing in that premise implies a
direction of travel. Uniformly low entailment is the model declining to pick, and
worth checking for before you trust the ranking.

## Router

```bash
python router.py "my python script throws a KeyError on line 40"
python router.py                 # demo across seven request types, then a prompt
```

```python
from router import route

d = route("What is 17% of 2340?")
d.model           # 'deepseek-math'
d.confidence      # 0.618
d.used_fallback   # False
d.scores          # every route's score, descending
```

Each route is one sentence describing the requests it handles. Adding a route
means appending a `Route(model=..., hypothesis=...)` to `ROUTES`. Phrase
hypotheses as declarative statements ("This is a request about X"); questions and
bare keywords both measure worse.

Measured on the built-in demo set:

| request | routed to | confidence |
| --- | --- | --- |
| `My python script throws a KeyError on line 40, can you fix it?` | qwen3-coder-480b | 0.864 |
| `What is 17% of 2340?` | deepseek-math | 0.618 |
| `Should we migrate our monolith to microservices? Walk me through the tradeoffs.` | claude-opus-4 | 0.947 |
| `Write me a short poem about rain on a tin roof.` | gpt-4o-creative | 0.945 |
| `What's in this screenshot I attached?` | qwen3-vl | 0.649 |
| `hey there` | llama-3.1-8b | 0.955 |
| `asdkjh qwe` | llama-3.1-70b (fallback) | 0.390 |

### How scoring works

The user's message becomes the premise and every route's hypothesis is scored in
a single batched forward pass. Two normalisation steps follow, both load-bearing:

1. **Per route, softmax over entailment and contradiction only.** Neutral is the
   dominant class for unrelated pairs, often above 0.99, so leaving it in
   compresses every route toward zero and destroys the ranking.
2. **Across routes, softmax over those scores.** Each route comes from an
   independent forward pass, so raw scores are not on a comparable scale.

A winner that does not clear `THRESHOLD` (0.5) is treated as no match and goes to
`FALLBACK`.

### Before using this in production

**Latency is 0.6-0.9 s per decision** on an M-series Mac with bf16 on CPU/MPS,
measured. That is paid before the chosen model starts working, and it grows with
the number of routes since they share one batch. A small embedding model with
cosine similarity, or a fine-tuned DistilBERT classifier, gets the same job done
in single-digit milliseconds. A 4B NLI model earns its cost only while you have
no labelled routing data.

**The 0.5 threshold is fitted to seven hand-written examples.** It separates them
cleanly, but seven examples is not a validation set. Log `Decision.scores` against
real traffic and re-fit.

## Why `openjev_config/`

Loading the model straight from the Hub fails:

```
score.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([3, 2560]) vs model: torch.Size([2, 2560])
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
```

The upstream config sets `id2label` with three NLI labels at the top level only.
`AutoModelForSequenceClassification` hands the model `config.text_config` rather
than the top-level config
([auto_factory.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/auto_factory.py),
the `config.sub_configs.get("text_config")` branch), and that sub-config has no
`id2label`, so it falls back to the default 2 labels. The head is built as
`[2, 2560]` while the checkpoint ships `[3, 2560]`.

`openjev_config/config.json` is the upstream config with `id2label` and
`label2id` copied into `text_config`, so the head is built at the right size and
the trained classifier weights actually load. Passing
`ignore_mismatched_sizes=True` also silences the error, but it reinitialises the
head to random weights and every prediction becomes noise.

Consequence of vendoring the config: it is pinned to the upstream revision it was
copied from. Refresh it by hand if upstream changes.

The `model.visual.*` keys reported as UNEXPECTED on load are expected. The
checkpoint retains the vision tower from the base VLM and the text-only
classification path does not use it.

## Credits

Model: [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev), MIT
licensed, fine-tuned from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
`openjev_config/config.json` is derived from that repo.
