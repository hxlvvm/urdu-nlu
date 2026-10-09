"""Export the dev-selected checkpoint to ONNX, quantize it to int8, and check both against PyTorch.

  python -m nlu.export_onnx --run runs/en_ur/best --out artifacts

Writes artifacts/{model.fp32.onnx, model.int8.onnx, tokenizer.json, labels.json, export_report.json}.
The report holds test-set metrics of the int8 model through the serving code path, its agreement with
PyTorch, and single-request CPU latency for PyTorch fp32, ONNX fp32 and ONNX int8.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from nlu import model as M
from nlu.data import load_locale
from nlu.metrics import score
from nlu.predictor import Predictor
from nlu.train import ROOT, encode, predict


def latency(fn, reps=200, warmup=20):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/en_ur/best"))
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    ap.add_argument("--raw", default=str(ROOT / "data/raw/1.1/data"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, meta = M.load(run)
    tok = AutoTokenizer.from_pretrained(meta["encoder"])
    tok.save_pretrained(out / "_tok")
    shutil.copy(out / "_tok" / "tokenizer.json", out / "tokenizer.json")
    shutil.rmtree(out / "_tok")
    shutil.copy(run / "labels.json", out / "labels.json")

    fp32, int8 = out / "model.fp32.onnx", out / "model.int8.onnx"
    dummy = tok([["set", "an", "alarm"]], is_split_into_words=True, return_tensors="pt")
    torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
                      input_names=["input_ids", "attention_mask"], output_names=["intent_logits", "slot_logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "intent_logits": {0: "batch"}, "slot_logits": {0: "batch", 1: "seq"}},
                      opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    report = {"size_mb": {p.name: round(p.stat().st_size / 2**20, 1) for p in (fp32, int8)}, "test": {}}
    i2id = {k: i for i, k in enumerate(meta["intents"])}
    t2id = {k: i for i, k in enumerate(meta["tags"])}
    for loc in ("en-US", "ur-PK"):
        rows = [r for r in load_locale(args.raw, loc) if r["partition"] == "test"]
        ti, tt = predict(model, encode(rows, tok, i2id, t2id, 64), rows, meta, tok.pad_token_id, torch.device("cpu"))
        entry = {"pytorch_fp32": score(rows, ti, tt)}
        for name in ("model.fp32.onnx", "model.int8.onnx"):
            pred = Predictor(out, name, threads=args.threads)
            res = [p for s in range(0, len(rows), 64) for p in pred.predict_words([r["tokens"] for r in rows[s:s + 64]])]
            oi, ot = [p["intent"] for p in res], [p["tags"] for p in res]
            entry[name] = {**score(rows, oi, ot),
                           "intent_agreement_with_pytorch": float(np.mean([a == b for a, b in zip(oi, ti)])),
                           "tag_agreement_with_pytorch": float(np.mean([a == b for a, b in zip(ot, tt)]))}
        llm = ROOT / "runs/baseline_llm.json"
        if llm.exists():  # same utterances the LLM saw, for a like-for-like comparison
            ids = set(json.loads(llm.read_text())["locales"][loc]["ids"])
            sub = [(p, r) for p, r in zip(oi, rows) if r["id"] in ids]
            entry["model.int8.onnx"]["intent_acc_on_llm_subset"] = sum(p == r["intent"] for p, r in sub) / len(sub)
        report["test"][loc] = entry
        print(loc, json.dumps(entry, indent=1), flush=True)

    words = "مجھے کل صبح سات بجے جگا دیں".split()
    enc = tok([words], is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        report["latency_single_request"] = {"threads": args.threads,
                                            "pytorch_fp32": latency(lambda: model(enc["input_ids"], enc["attention_mask"]))}
    for name in ("model.fp32.onnx", "model.int8.onnx"):
        pred = Predictor(out, name, threads=args.threads)
        report["latency_single_request"][name] = latency(lambda: pred.predict_words([words]))
    print(json.dumps(report["latency_single_request"], indent=1), report["size_mb"], flush=True)
    (out / "export_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
