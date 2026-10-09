"""CPU inference with ONNX Runtime and the `tokenizers` library only (no torch, no transformers).

This is the code path the API and the Docker image run; the export step checks it against PyTorch.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from nlu.data import decode_word_tags, spans

MAX_LEN = 64


class Predictor:
    def __init__(self, artifacts, model_file="model.int8.onnx", threads=None):
        art = Path(artifacts)
        meta = json.loads((art / "labels.json").read_text(encoding="utf-8"))
        self.intents, self.id2tag = meta["intents"], dict(enumerate(meta["tags"]))
        self.tok = Tokenizer.from_file(str(art / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=MAX_LEN)
        self.pad_id = self.tok.token_to_id("<pad>")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(art / model_file), opts, providers=["CPUExecutionProvider"])

    def predict_words(self, batch_words):
        """List of word lists -> list of {intent, confidence, tags} (one tag per word)."""
        encs = self.tok.encode_batch(batch_words, is_pretokenized=True)
        n = max(len(e.ids) for e in encs)
        ids = np.full((len(encs), n), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(encs), n), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
        intent_logits, slot_logits = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})
        probs = np.exp(intent_logits - intent_logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        out = []
        for i, (e, words) in enumerate(zip(encs, batch_words)):
            tags = decode_word_tags(e.word_ids, slot_logits[i].argmax(-1), self.id2tag, len(words))
            k = int(probs[i].argmax())
            out.append({"intent": self.intents[k], "confidence": float(probs[i, k]), "tags": tags})
        return out

    def predict_text(self, text):
        words = text.split()
        r = self.predict_words([words])[0]
        return {**r, "tokens": words, "slots": spans(words, r["tags"])}
