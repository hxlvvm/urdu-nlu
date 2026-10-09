"""Joint intent + slot model: one multilingual encoder, two linear heads."""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class JointNLU(nn.Module):
    def __init__(self, encoder_name, n_intents, n_tags, dropout=0.1, pretrained=True):
        super().__init__()
        self.encoder = (AutoModel.from_pretrained(encoder_name) if pretrained
                        else AutoModel.from_config(AutoConfig.from_pretrained(encoder_name)))
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, n_intents)
        self.slot_head = nn.Linear(hidden, n_tags)

    def forward(self, input_ids, attention_mask):
        h = self.drop(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        return self.intent_head(h[:, 0]), self.slot_head(h)


def save(model, out_dir, encoder_name, intents, tags):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "labels.json").write_text(json.dumps({"encoder": encoder_name, "intents": intents, "tags": tags},
                                                ensure_ascii=False, indent=1))


def load(out_dir, device="cpu"):
    meta = json.loads((Path(out_dir) / "labels.json").read_text())
    model = JointNLU(meta["encoder"], len(meta["intents"]), len(meta["tags"]), pretrained=False)
    model.load_state_dict(torch.load(Path(out_dir) / "model.pt", map_location=device))
    return model.to(device).eval(), meta
