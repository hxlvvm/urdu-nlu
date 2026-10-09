# urdu-nlu

Intent detection and slot filling (entity extraction) for **Urdu and English** voice-assistant requests.
A multilingual transformer (XLM-R base) is fine-tuned in PyTorch on Amazon's MASSIVE dataset, checked against the
published paper, compared with a classical baseline and a zero-shot LLM, exported to int8 ONNX, and served on CPU
through FastAPI in Docker.

```
"مجھے کل صبح سات بجے جگا دیں"   ("wake me up at seven tomorrow morning")
  -> intent: alarm_set
  -> slots:  date = کل,  time = صبح سات بجے
```

## Results

<!-- RESULTS:START -->
### Test set (2,974 utterances per language): intent accuracy / slot F1 / exact match, %

| Model | Urdu (ur-PK) | English (en-US) |
|---|---|---|
| XLM-R, fine-tuned on English + Urdu | 86.2 / 71.2 / 63.0 | 89.1 / 80.7 / 71.3 |
| XLM-R, fine-tuned on Urdu only | 85.8 / 70.1 / 61.9 | 80.6 / 61.6 / 51.2 |
| XLM-R, fine-tuned on English only (Urdu is zero-shot) | 60.5 / 40.1 / 29.8 | 89.4 / 79.7 / 71.1 |
| Paper: XLM-R base, trained on all 51 languages | 83.2 / 68.0 / 59.0 | 88.3 / 78.7 / 69.7 |
| Paper: XLM-R base, English only (zero-shot) | 65.6 / 41.4 / 30.5 | n/a |
| TF-IDF + logistic regression, English + Urdu (intent only) | 82.6 / - / - | 84.7 / - / - |
| TF-IDF + logistic regression, English only (intent only) | 6.5 / - / - | 84.8 / - / - |

### Fine-tuned model vs a zero-shot LLM, intent accuracy on the same 300 random test utterances, %

| Model | Urdu | English |
|---|---|---|
| XLM-R English + Urdu, int8 ONNX on CPU | 86.3 | 87.7 |
| gemini-3.5-flash-lite, zero-shot prompt with the 60 intent names | 76.3 | 80.3 |

### Export: accuracy kept after int8 quantization, and CPU latency

| Model file | Size MB | Urdu exact match % | English exact match % | Urdu intent agreement with PyTorch % |
|---|---|---|---|---|
| PyTorch fp32 | - | 62.9 | 71.3 | 100.0 |
| model.fp32.onnx | 1059.1 | 62.9 | 71.3 | 100.0 |
| model.int8.onnx | 265.5 | 61.6 | 70.1 | 94.8 |

Single Urdu request, 4 CPU threads, median / p95 ms: pytorch_fp32 38.28 / 38.75; model.fp32.onnx 17.17 / 17.52; model.int8.onnx 6.66 / 6.79
<!-- RESULTS:END -->

All numbers come from `runs/*/metrics.json`, `runs/baseline_*.json` and `artifacts/export_report.json`, printed by
`python -m nlu.report`. Checkpoints are chosen on the dev split; the test split is scored once per run. One seed per
configuration, so differences of about a point are within run-to-run noise.

**What the comparisons show**

- **Against the paper.** Fine-tuning on English + Urdu matches or beats the published XLM-R base numbers for both
  languages. The paper trained on all 51 languages at once, so this is a sanity check, not a like-for-like contest.
- **Against TF-IDF.** For intent alone, word and character n-grams with logistic regression come within a few points.
  The transformer earns its place on slots, which TF-IDF cannot do, and on cross-lingual transfer.
- **Cross-lingual transfer.** Trained on English only and chosen on English dev data only, the model still gets most
  Urdu intents right without seeing a single Urdu label. TF-IDF trained on English gets 6.5 % of Urdu intents.
- **Against a zero-shot LLM.** On the same 300 random test utterances per language, the fine-tuned 278M-parameter
  model beats a prompted Gemini model by about 10 points in Urdu and 7 in English, answers in under 7 ms on 4 CPU
  threads, and costs nothing per request.
- **Quantization trade-off.** int8 makes the model 4x smaller (266 MB) and 2.6x faster than fp32 ONNX (5.7x faster
  than PyTorch on CPU) for about 1.3 points of exact match. fp32 ONNX matches PyTorch exactly.

**Where it fails.** The weakest intent in both languages is `general_quirky` (chit-chat), confused mostly with
`qa_factoid`; it is wrong as often in English as in Urdu, so the problem is overlapping labels, not the language.
Per-intent accuracy and the most common confusion are in `runs/en_ur/metrics.json` under `per_intent`.

## How it works

| Step | File | What it does |
|---|---|---|
| Data | `nlu/data.py` | Parses MASSIVE's inline slot markup (`[time : five am]`) into words + BIO tags; labels the first subword of each word and masks the rest |
| Model | `nlu/model.py` | XLM-R encoder with two heads: intent from the `<s>` token, a BIO tag per token |
| Training | `nlu/train.py` | AdamW with separate encoder/head learning rates, warmup and linear decay, bf16 on GPU; dev-based checkpoint selection; MLflow tracking |
| Metrics | `nlu/metrics.py` | Intent accuracy, span-level slot micro-F1 (seqeval) and exact match, the three MASSIVE metrics; per-intent error analysis |
| Baselines | `nlu/baselines.py`, `nlu/llm_baseline.py` | TF-IDF + logistic regression; zero-shot Gemini on a fixed random subset |
| Export | `nlu/export_onnx.py` | ONNX export, dynamic int8 quantization, accuracy and agreement with PyTorch on the full test sets, CPU latency |
| Serving | `nlu/predictor.py`, `serve/app.py` | ONNX Runtime + `tokenizers` only (no PyTorch in the image); FastAPI with input validation |

Before the full runs, the pipeline had to overfit 64 Urdu utterances to 100 % on all three metrics (it does, by
epoch 60), which rules out label-alignment and decoding bugs.

## Use it

```bash
# API with the released int8 model (no GPU, no PyTorch)
gh release download model-v1 -R hxlvvm/urdu-nlu -D artifacts --clobber -p model.int8.onnx -p tokenizer.json -p labels.json
docker build -t urdu-nlu . && docker run -p 8000:8000 urdu-nlu
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text": "مجھے کل صبح سات بجے جگا دیں"}'
```

```json
{"intent": "alarm_set", "confidence": 0.99995, "slots": {"date": "کل", "time": "صبح سات بجے"}, "tokens": [...], "tags": [...], "latency_ms": ...}
```

## Reproduce

```bash
pip install -r requirements-train.txt
curl -O https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
mkdir -p data/raw && tar -xzf amazon-massive-dataset-1.1.tar.gz -C data/raw
python -m pytest -q
python -m nlu.train --name en_ur   --train-locales en-US,ur-PK     # ~6 min on one A100
python -m nlu.train --name en_only --train-locales en-US
python -m nlu.train --name ur_only --train-locales ur-PK
python -m nlu.baselines
GEMINI_API_KEY=... python -m nlu.llm_baseline --n 300            # optional
python -m nlu.export_onnx --run runs/en_ur/best --out artifacts
python -m nlu.report
```

`jobs/*.sbatch` are SLURM templates of the scripts used for the runs (one GPU, 2 CPU cores, 12 GB); fill in your
cluster's partition names.

## Limitations

- MASSIVE's Urdu is written in Urdu script. Much real Pakistani chat is Roman Urdu ("mujhe kal subah utha dena"),
  which this model has not been trained or tested on.
- MASSIVE is translated from English voice-assistant requests (SLURP), so its Urdu is cleaner and more formal than
  real user messages.
- Slots are scored on whitespace words; the API splits raw text on whitespace too, so punctuation attached to a
  word stays part of it.
- One training seed per configuration.

## Data and licences

- Code: MIT (see `LICENSE`).
- Data: [MASSIVE](https://github.com/alexa/massive) by Amazon, CC BY 4.0 (FitzGerald et al., 2022,
  arXiv:2204.08582), localized from [SLURP](https://github.com/pswietojanski/slurp), CC BY 4.0. The released model
  weights are trained on MASSIVE and carry the same attribution.
- Encoder: [XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base), MIT.
