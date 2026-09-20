"""Natural language inference with the openjev Qwen3.5-4B classifier.

The model scores a premise/hypothesis pair across three classes: contradiction,
entailment, and neutral. Those three are fixed by training. `id2label` in the
config is a display map only, so renaming the classes there changes the printed
names and nothing else.

An NLI model does not answer questions. To choose among candidate answers, score
each candidate as its own hypothesis against a shared premise and compare the
entailment probabilities, which is what `rank` does.

Run this file directly for a demo plus an interactive prompt, or import `nli` and
`rank`. The weights load on first use, not on import.
"""

import sys
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

REPO = "AlexWortega/openjev"
SUBFOLDER = "qwen3.5-4b-nli-v2"

# Local copy of the upstream config with id2label/label2id also written into
# `text_config`. AutoModelForSequenceClassification hands the model the text
# sub-config rather than the top-level one, and upstream only sets the labels at
# the top level, so the classification head gets built with the default 2 labels
# while the checkpoint ships 3. Without this the load fails on a size mismatch
# for `score.weight`. See README for the full story.
CONFIG_DIR = Path(__file__).parent / "openjev_config"

CONFIG = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code=True)
NLI_TEMPLATE = CONFIG.nli_template
ID2LABEL = {int(k): v for k, v in CONFIG.id2label.items()}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

ENTAILMENT = LABEL2ID["entailment"]
CONTRADICTION = LABEL2ID["contradiction"]
NEUTRAL = LABEL2ID["neutral"]


@lru_cache(maxsize=1)
def load():
    """Load tokenizer and model once, reusing them on later calls."""
    tokenizer = AutoTokenizer.from_pretrained(
        REPO, subfolder=SUBFOLDER, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = CONFIG.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        REPO,
        subfolder=SUBFOLDER,
        config=CONFIG,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    return tokenizer, model


@torch.no_grad()
def nli_logits(premise, hypotheses):
    """Return [len(hypotheses), 3] logits for one premise against several hypotheses.

    All hypotheses go through as a single padded batch. The classification head
    pools the last non-pad token, so padding does not shift the result.
    """
    tokenizer, model = load()
    texts = [NLI_TEMPLATE.format(premise=premise, hypothesis=h) for h in hypotheses]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    return model(**inputs).logits.float()


def nli(premise, hypotheses):
    """Score a premise against one hypothesis or many.

    Returns one {label: probability} dict per hypothesis.
    """
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]

    probs = nli_logits(premise, hypotheses).softmax(dim=-1)
    return [{ID2LABEL[i]: float(row[i]) for i in range(len(row))} for row in probs]


def rank(premise, choices):
    """Rank candidate answers by entailment probability, highest first.

    Returns a list of (choice, probability) tuples. Low scores across the board
    mean the premise does not imply any of the choices, which is a meaningful
    answer in itself.
    """
    scored = [
        (choice, scores[ID2LABEL[ENTAILMENT]])
        for choice, scores in zip(choices, nli(premise, choices))
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def demo():
    premise = "The bird is 0.05 below the centre of the gap."
    print(f"  premise: {premise}")
    for hypothesis in [
        "The bird is below the centre of the gap.",
        "The bird is above the centre of the gap.",
        "The bird is blue.",
    ]:
        (scores,) = nli(premise, hypothesis)
        best = max(scores, key=scores.get)
        print(f"     {best:13s} {scores[best]:.3f}  {hypothesis}")

    premise = "There is a demon to your right and a demon to your left."
    print(f"\n  premise: {premise}")
    for choice, score in rank(
        premise, ["You go left.", "You go right.", "You go straight ahead."]
    ):
        print(f"     entailment    {score:.3f}  {choice}")


def repl():
    print("\nEnter a premise, then a hypothesis. Blank premise or Ctrl-D quits.")
    print("Separate several hypotheses with '|' to rank them by entailment.\n")

    while True:
        try:
            premise = input("premise>    ").strip()
            if not premise:
                return
            raw = input("hypothesis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not raw:
            continue

        choices = [part.strip() for part in raw.split("|") if part.strip()]

        if len(choices) == 1:
            (scores,) = nli(premise, choices[0])
            best = max(scores, key=scores.get)
            print(f"  {scores}")
            print(f"  prediction: {best}\n")
        else:
            for choice, score in rank(premise, choices):
                print(f"  entailment {score:.3f}  {choice}")
            print()


if __name__ == "__main__":
    demo()
    if sys.stdin.isatty():
        repl()
