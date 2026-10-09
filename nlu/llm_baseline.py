"""Zero-shot LLM intent baseline (Gemini) on a fixed random subset of the test set.

The key is read from GEMINI_API_KEY and never written anywhere. The subset is fixed by a seed so the
comparison with the fine-tuned model is on identical utterances.
  GEMINI_API_KEY=... python -m nlu.llm_baseline --n 300
"""
import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from nlu.data import load_locale

ROOT = Path(__file__).resolve().parent.parent
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def ask(prompt, model, key):
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}).encode()
    req = urllib.request.Request(API.format(model=model), data=body,
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
    return json.JSONDecoder(strict=False).raw_decode(text[text.find("{"):])[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="utterances per locale")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--raw", default=str(ROOT / "data/raw/1.1/data"))
    args = ap.parse_args(argv)
    key = os.environ["GEMINI_API_KEY"]
    out = {"model": args.model, "n_per_locale": args.n, "seed": args.seed, "locales": {}}
    for loc in ("en-US", "ur-PK"):
        rows = [r for r in load_locale(args.raw, loc) if r["partition"] == "test"]
        intents = sorted({r["intent"] for r in rows})
        sample = random.Random(args.seed).sample(rows, args.n)
        preds = {}
        for s in range(0, len(sample), args.batch):
            chunk = sample[s:s + args.batch]
            prompt = ("Classify each voice-assistant request into exactly one intent from this list:\n"
                      + ", ".join(intents)
                      + "\n\nRequests (JSON, id -> text):\n"
                      + json.dumps({r["id"]: r["text"] for r in chunk}, ensure_ascii=False)
                      + '\n\nReturn JSON {"<id>": "<intent>"} with one entry per request and nothing else.')
            for attempt in range(3):
                try:
                    preds.update(ask(prompt, args.model, key))
                    break
                except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as e:
                    print(f"  retry {attempt + 1}: {type(e).__name__}", flush=True)
                    time.sleep(10)
            time.sleep(5)
        ok = [preds.get(r["id"]) == r["intent"] for r in sample]
        out["locales"][loc] = {"intent_acc": sum(ok) / len(ok), "answered": sum(r["id"] in preds for r in sample),
                               "ids": [r["id"] for r in sample]}
        print(loc, out["locales"][loc]["intent_acc"], flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/baseline_llm.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
