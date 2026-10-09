"""Print the README results tables from the saved JSON files (no hand-copied numbers).

  python -m nlu.report > docs/results.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCS = ("ur-PK", "en-US")


def load(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def main():
    paper = load("docs/paper_reference.json")
    runs = {n: load(f"runs/{n}/metrics.json") for n in ("en_ur", "ur_only", "en_only")}
    tfidf, llm, exp = load("runs/baseline_tfidf.json"), load("runs/baseline_llm.json"), load("artifacts/export_report.json")
    label = {"en_ur": "XLM-R, fine-tuned on English + Urdu", "ur_only": "XLM-R, fine-tuned on Urdu only",
             "en_only": "XLM-R, fine-tuned on English only (Urdu is zero-shot)"}
    print("### Test set (2,974 utterances per language): intent accuracy / slot F1 / exact match, %\n")
    print("| Model | Urdu (ur-PK) | English (en-US) |\n|---|---|---|")
    for n, m in runs.items():
        if m:
            cells = [" / ".join(pct(m["test"][loc][k]) for k in ("intent_acc", "slot_f1", "exact_match")) for loc in LOCS]
            print(f"| {label[n]} | {cells[0]} | {cells[1]} |")
    if paper:
        f = lambda d: " / ".join(f"{d[k]:.1f}" for k in ("intent_acc", "slot_f1", "exact_match"))  # noqa: E731
        print(f"| Paper: XLM-R base, trained on all 51 languages | {f(paper['ur-PK']['full'])} | {f(paper['en-US']['full'])} |")
        print(f"| Paper: XLM-R base, English only (zero-shot) | {f(paper['ur-PK']['zero'])} | n/a |")
    if tfidf:
        for n, name in (("en_ur", "English + Urdu"), ("en_only", "English only")):
            print(f"| TF-IDF + logistic regression, {name} (intent only) | {pct(tfidf[n]['ur-PK']['intent_acc'])} / - / - "
                  f"| {pct(tfidf[n]['en-US']['intent_acc'])} / - / - |")
    if llm and exp:
        n = llm["n_per_locale"]
        print(f"\n### Fine-tuned model vs a zero-shot LLM, intent accuracy on the same {n} random test utterances, %\n")
        print(f"| Model | Urdu | English |\n|---|---|---|")
        ft = [pct(exp["test"][loc]["model.int8.onnx"].get("intent_acc_on_llm_subset")) for loc in LOCS]
        print(f"| XLM-R English + Urdu, int8 ONNX on CPU | {ft[0]} | {ft[1]} |")
        print(f"| {llm['model']}, zero-shot prompt with the 60 intent names | "
              f"{pct(llm['locales']['ur-PK']['intent_acc'])} | {pct(llm['locales']['en-US']['intent_acc'])} |")
    if exp:
        print("\n### Export: accuracy kept after int8 quantization, and CPU latency\n")
        print("| Model file | Size MB | Urdu exact match % | English exact match % | Urdu intent agreement with PyTorch % |")
        print("|---|---|---|---|---|")
        t = exp["test"]
        print(f"| PyTorch fp32 | - | {pct(t['ur-PK']['pytorch_fp32']['exact_match'])} | "
              f"{pct(t['en-US']['pytorch_fp32']['exact_match'])} | 100.0 |")
        for name in ("model.fp32.onnx", "model.int8.onnx"):
            print(f"| {name} | {exp['size_mb'][name]} | {pct(t['ur-PK'][name]['exact_match'])} | "
                  f"{pct(t['en-US'][name]['exact_match'])} | {pct(t['ur-PK'][name]['intent_agreement_with_pytorch'])} |")
        lat = exp["latency_single_request"]
        print(f"\nSingle Urdu request, {lat['threads']} CPU threads, median / p95 ms: "
              + "; ".join(f"{k} {v['p50_ms']} / {v['p95_ms']}" for k, v in lat.items() if k != "threads"))


if __name__ == "__main__":
    main()
