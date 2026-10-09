"""MASSIVE loading, slot-annotation parsing and subword label alignment.

MASSIVE marks slots inline: "wake me up at [time : five am] [date : this week]".
`parse_annotation` turns that into whitespace tokens plus BIO tags, which is the unit the slot metric is scored on.
"""
import json
import re
from pathlib import Path

SLOT_RE = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")
IGNORE = -100


def parse_annotation(annot_utt):
    """'set [time : five am]' -> (['set', 'five', 'am'], ['O', 'B-time', 'I-time'])."""
    tokens, tags, pos = [], [], 0
    for m in SLOT_RE.finditer(annot_utt):
        for w in annot_utt[pos:m.start()].split():
            tokens.append(w)
            tags.append("O")
        slot, value = m.group(1).strip(), m.group(2).split()
        for i, w in enumerate(value):
            tokens.append(w)
            tags.append(("B-" if i == 0 else "I-") + slot)
        pos = m.end()
    for w in annot_utt[pos:].split():
        tokens.append(w)
        tags.append("O")
    return tokens, tags


def load_locale(raw_dir, locale):
    """All rows of one locale as dicts with id, partition, intent, tokens, tags."""
    rows = []
    with open(Path(raw_dir) / f"{locale}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tokens, tags = parse_annotation(r["annot_utt"])
            rows.append({"id": f"{locale}:{r['id']}", "locale": locale, "partition": r["partition"],
                         "intent": r["intent"], "text": r["utt"], "tokens": tokens, "tags": tags})
    return rows


def label_maps(rows):
    """Sorted, deterministic label vocabularies built from training rows."""
    intents = sorted({r["intent"] for r in rows})
    slots = sorted({t[2:] for r in rows for t in r["tags"] if t != "O"})
    tags = ["O"] + [f"{p}-{s}" for s in slots for p in ("B", "I")]
    return intents, tags


def align_labels(word_ids, word_tags, tag2id):
    """Label the first subword of each word; later subwords and special tokens get IGNORE.

    `word_ids` is the tokenizer's mapping from subword position to word index (None for special tokens).
    """
    out, prev = [], None
    for w in word_ids:
        if w is None or w == prev:
            out.append(IGNORE)
        else:
            out.append(tag2id[word_tags[w]])
        prev = w
    return out


def decode_word_tags(word_ids, subword_pred, id2tag, n_words):
    """Inverse of align_labels: take the prediction at each word's first subword."""
    tags, prev = ["O"] * n_words, None
    for pos, w in enumerate(word_ids):
        if w is not None and w != prev and w < n_words:
            tags[w] = id2tag[int(subword_pred[pos])]
        prev = w
    return tags


def spans(tokens, tags):
    """BIO tags -> {slot: value} (first occurrence wins); a stray I- starts a new span."""
    out, cur, words = {}, None, []
    for tok, tag in list(zip(tokens, tags)) + [("", "O")]:
        if tag.startswith("I-") and cur == tag[2:]:
            words.append(tok)
            continue
        if cur and cur not in out:
            out[cur] = " ".join(words)
        cur, words = (tag[2:], [tok]) if tag != "O" else (None, [])
    return out
