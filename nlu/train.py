"""Fine-tune the joint model on MASSIVE locales, select on dev, score test once.

  python -m nlu.train --name en_ur --train-locales en-US,ur-PK
  python -m nlu.train --name en_only --train-locales en-US          # cross-lingual: still evaluated on ur-PK
  python -m nlu.train --name gate --train-locales ur-PK --max-rows 64 --overfit --epochs 40   # sanity gate
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from nlu import model as M
from nlu.data import IGNORE, align_labels, decode_word_tags, label_maps, load_locale
from nlu.metrics import per_intent, score

ROOT = Path(__file__).resolve().parent.parent


def encode(rows, tok, intent2id, tag2id, max_len):
    enc = tok([r["tokens"] for r in rows], is_split_into_words=True, truncation=True, max_length=max_len)
    feats = []
    for i, r in enumerate(rows):
        wid = enc.word_ids(i)
        feats.append({"input_ids": enc["input_ids"][i], "word_ids": wid,
                      "intent": intent2id[r["intent"]], "slots": align_labels(wid, r["tags"], tag2id)})
    return feats


def batches(feats, size, pad_id, shuffle, rng):
    idx = list(range(len(feats)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), size):
        chunk = [feats[i] for i in idx[s:s + size]]
        n = max(len(f["input_ids"]) for f in chunk)
        ids = torch.full((len(chunk), n), pad_id)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        slots = torch.full((len(chunk), n), IGNORE)
        for j, f in enumerate(chunk):
            L = len(f["input_ids"])
            ids[j, :L] = torch.tensor(f["input_ids"])
            mask[j, :L] = 1
            slots[j, :L] = torch.tensor(f["slots"])
        yield chunk, ids, mask, torch.tensor([f["intent"] for f in chunk]), slots


@torch.no_grad()
def predict(model, feats, rows, meta, pad_id, device, size=128):
    model.eval()
    intents, tags = [], []
    for chunk, ids, mask, _, _ in batches(feats, size, pad_id, False, None):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            il, sl = model(ids.to(device), mask.to(device))
        il, sl = il.float().argmax(-1).cpu(), sl.float().argmax(-1).cpu()
        for j, f in enumerate(chunk):
            intents.append(meta["intents"][int(il[j])])
            n_words = max((w for w in f["word_ids"] if w is not None), default=-1) + 1
            tags.append(decode_word_tags(f["word_ids"], sl[j].tolist(), dict(enumerate(meta["tags"])), n_words))
    # words cut off by truncation are scored as 'O'
    tags = [t + ["O"] * (len(r["tokens"]) - len(t)) for t, r in zip(tags, rows)]
    return intents, tags


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-locales", default="en-US,ur-PK")
    ap.add_argument("--eval-locales", default="en-US,ur-PK")
    ap.add_argument("--select-locales", default="",
                    help="dev locales used to pick the checkpoint (default: the training locales, so a zero-shot "
                         "run never sees target-language labels)")
    ap.add_argument("--encoder", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--raw", default=str(ROOT / "data/raw/1.1/data"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-rows", type=int, default=0, help="cap training rows (sanity runs)")
    ap.add_argument("--overfit", action="store_true", help="evaluate on the training rows themselves")
    args = ap.parse_args(argv)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)

    data = {loc: load_locale(args.raw, loc) for loc in set(args.train_locales.split(",") + args.eval_locales.split(","))}
    train = [r for loc in args.train_locales.split(",") for r in data[loc] if r["partition"] == "train"]
    if args.max_rows:
        train = rng.sample(train, args.max_rows)
    # label vocabularies from ALL training rows of every locale, so every run shares one label space
    intents, tags = label_maps([r for rows in data.values() for r in rows if r["partition"] == "train"])
    meta = {"encoder": args.encoder, "intents": intents, "tags": tags}
    i2id, t2id = {k: i for i, k in enumerate(intents)}, {k: i for i, k in enumerate(tags)}

    tok = AutoTokenizer.from_pretrained(args.encoder)
    tr_feats = encode(train, tok, i2id, t2id, args.max_len)
    split = lambda loc, part: [r for r in data[loc] if r["partition"] == part]  # noqa: E731
    dev = {loc: (train if args.overfit else split(loc, "dev")) for loc in args.eval_locales.split(",")}
    dev_feats = {loc: encode(rows, tok, i2id, t2id, args.max_len) for loc, rows in dev.items()}
    select = [loc for loc in (args.select_locales or args.train_locales).split(",") if loc in dev]
    assert select, "no selection locale among the eval locales"

    model = M.JointNLU(args.encoder, len(intents), len(tags)).to(device)
    heads = list(model.intent_head.parameters()) + list(model.slot_head.parameters())
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                             {"params": heads, "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr_feats) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    history, best, t0 = [], -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for _, ids, mask, y_int, y_slot in batches(tr_feats, args.batch, tok.pad_token_id, True, rng):
            ids, mask, y_int, y_slot = ids.to(device), mask.to(device), y_int.to(device), y_slot.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                il, sl = model(ids, mask)
            loss = ce(il.float(), y_int) + ce(sl.float().reshape(-1, len(tags)), y_slot.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(y_int)
        ev = {loc: score(dev[loc], *predict(model, dev_feats[loc], dev[loc], meta, tok.pad_token_id, device))
              for loc in dev}
        sel = float(np.mean([ev[loc]["exact_match"] for loc in select]))
        history.append({"epoch": ep, "train_loss": tot / len(tr_feats), "dev": ev, "select": sel})
        print(f"epoch {ep} loss {tot / len(tr_feats):.4f} " + " ".join(
            f"{loc}: int {m['intent_acc']:.4f} slot {m['slot_f1']:.4f} em {m['exact_match']:.4f}"
            for loc, m in ev.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        if sel > best:
            best = sel
            M.save(model, out / "best", args.encoder, intents, tags)

    result = {"args": vars(args), "history": history, "best_dev_select": best, "train_rows": len(train),
              "seconds": round(time.time() - t0)}
    if not args.overfit:  # test set: scored once, with the dev-selected checkpoint
        model, meta = M.load(out / "best", device)
        result["test"], result["per_intent"] = {}, {}
        for loc in args.eval_locales.split(","):
            rows = split(loc, "test")
            pi, pt = predict(model, encode(rows, tok, i2id, t2id, args.max_len), rows, meta, tok.pad_token_id, device)
            result["test"][loc] = score(rows, pi, pt)
            result["per_intent"][loc] = per_intent(rows, pi)
            print(f"TEST {loc}: {result['test'][loc]}", flush=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    log_mlflow(args, result, out)


def log_mlflow(args, result, out):
    """Optional experiment tracking to a local file store (runs/mlruns); skipped if mlflow is absent."""
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(f"file://{Path(args.out).resolve() / 'mlruns'}")
    mlflow.set_experiment("massive-en-ur")
    with mlflow.start_run(run_name=args.name):
        mlflow.log_params({k: v for k, v in vars(args).items() if k not in ("raw", "out")})
        for h in result["history"]:
            mlflow.log_metric("train_loss", h["train_loss"], step=h["epoch"])
            for loc, m in h["dev"].items():
                for k in ("intent_acc", "slot_f1", "exact_match"):
                    mlflow.log_metric(f"dev_{loc}_{k}", m[k], step=h["epoch"])
        for loc, m in result.get("test", {}).items():
            for k in ("intent_acc", "slot_f1", "exact_match"):
                mlflow.log_metric(f"test_{loc}_{k}", m[k])
        mlflow.log_artifact(str(out / "metrics.json"))


if __name__ == "__main__":
    main()
