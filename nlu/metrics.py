"""The three MASSIVE metrics: intent accuracy, slot micro-F1 (seqeval, span level) and exact match."""
from collections import Counter, defaultdict

from seqeval.metrics import f1_score


def score(gold_rows, pred_intents, pred_tags):
    gi = [r["intent"] for r in gold_rows]
    gt = [r["tags"] for r in gold_rows]
    n = len(gold_rows)
    intent_acc = sum(a == b for a, b in zip(gi, pred_intents)) / n
    exact = sum(a == b and x == y for a, b, x, y in zip(gi, pred_intents, gt, pred_tags)) / n
    return {"n": n, "intent_acc": intent_acc, "slot_f1": f1_score(gt, pred_tags), "exact_match": exact}


def per_intent(gold_rows, pred_intents):
    """Accuracy per gold intent and the most common confusion, for error analysis."""
    total, right, confused = Counter(), Counter(), defaultdict(Counter)
    for r, p in zip(gold_rows, pred_intents):
        total[r["intent"]] += 1
        if p == r["intent"]:
            right[r["intent"]] += 1
        else:
            confused[r["intent"]][p] += 1
    return {k: {"n": total[k], "acc": right[k] / total[k],
                "top_confusion": (confused[k].most_common(1) or [(None, 0)])[0][0]} for k in sorted(total)}
