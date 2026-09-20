"""Zero-shot model router built on the openjev NLI classifier.

Each route carries a hypothesis describing the kind of request it handles. A user
message becomes the premise, and the route whose hypothesis is most entailed by
the message wins. No training data and no labelled examples required, which is
the whole appeal: you add a route by writing one sentence.

Scores need two normalisation steps before they can be compared:

1. Per route, softmax over entailment and contradiction only. Neutral dominates
   on unrelated pairs, so leaving it in flattens every route toward zero and
   destroys the ranking.
2. Across routes, softmax over those per-route scores. Each route is scored by an
   independent forward pass, so the raw numbers are not on a shared scale.

A winner below `THRESHOLD` is treated as no match and goes to `FALLBACK`.

This module picks a model name. Wiring names to actual providers is left to the
caller.
"""

import sys
from dataclasses import dataclass

import torch

from open_jev import CONTRADICTION, ENTAILMENT, nli_logits


@dataclass(frozen=True)
class Route:
    model: str
    hypothesis: str


# Hypotheses read as declarative statements about the request. Questions and bare
# keywords both measure noticeably worse.
ROUTES = [
    Route(
        model="qwen3-coder-480b",
        hypothesis="This is a request about writing, debugging, or explaining source code.",
    ),
    Route(
        model="deepseek-math",
        hypothesis="This is a mathematics, arithmetic, or numerical calculation problem.",
    ),
    Route(
        model="claude-opus-4",
        hypothesis="This request needs careful multi-step reasoning, planning, or in-depth analysis.",
    ),
    Route(
        model="gpt-4o-creative",
        hypothesis="This is a request for creative writing such as a story, poem, or song.",
    ),
    Route(
        model="qwen3-vl",
        hypothesis="This is a request about an image, photo, screenshot, or diagram.",
    ),
    Route(
        model="llama-3.1-8b",
        hypothesis="This is small talk, a greeting, or a simple factual lookup.",
    ),
]

FALLBACK = "llama-3.1-70b"

# Cross-route probability the winner must clear. Fitted by hand against the
# examples in `demo()`, where clear requests land at 0.62-0.96 and gibberish tops
# out near 0.39. Seven examples is not a validation set: log `Decision.scores` on
# real traffic and re-fit.
THRESHOLD = 0.5

PREMISE_TEMPLATE = 'A user sent this request: "{question}"'


@dataclass
class Decision:
    model: str
    confidence: float
    used_fallback: bool
    scores: dict  # model name -> cross-route probability, descending


def route(question, routes=ROUTES, threshold=THRESHOLD, fallback=FALLBACK):
    """Choose a model for `question`. One batched forward pass over all routes."""
    premise = PREMISE_TEMPLATE.format(question=question)
    logits = nli_logits(premise, [r.hypothesis for r in routes])

    entailment = logits[:, [CONTRADICTION, ENTAILMENT]].softmax(dim=-1)[:, 1]
    across_routes = torch.log(entailment.clamp_min(1e-9)).softmax(dim=-1)

    scores = sorted(
        ((r.model, float(p)) for r, p in zip(routes, across_routes)),
        key=lambda item: item[1],
        reverse=True,
    )
    top_model, top_score = scores[0]
    no_clear_winner = top_score < threshold

    return Decision(
        model=fallback if no_clear_winner else top_model,
        confidence=top_score,
        used_fallback=no_clear_winner,
        scores=dict(scores),
    )


def explain(question, **kwargs):
    """Route `question` and print the full score breakdown."""
    decision = route(question, **kwargs)
    tag = ", fallback" if decision.used_fallback else ""
    print(f"\n  {question!r}")
    print(f"  -> {decision.model}   ({decision.confidence:.3f}{tag})")
    for model, score in decision.scores.items():
        marker = "*" if model == decision.model else " "
        print(f"     {marker} {score:.3f}  {model}")
    return decision


def demo():
    for question in [
        "My python script throws a KeyError on line 40, can you fix it?",
        "What is 17% of 2340?",
        "Should we migrate our monolith to microservices? Walk me through the tradeoffs.",
        "Write me a short poem about rain on a tin roof.",
        "What's in this screenshot I attached?",
        "hey there",
        "asdkjh qwe",
    ]:
        explain(question)


def repl():
    print("\nType a request to see where it routes. Blank line or Ctrl-D quits.")
    while True:
        try:
            question = input("\nrequest> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        explain(question)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        explain(" ".join(sys.argv[1:]))
    else:
        demo()
        if sys.stdin.isatty():
            repl()
