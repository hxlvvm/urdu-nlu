"""Regression checks on the real int8 model; skipped when artifacts/ has no model (e.g. in the unit-test CI job)."""
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parent.parent / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model.int8.onnx").exists(), reason="model artifacts not downloaded")


@pytest.fixture(scope="module")
def predictor():
    from nlu.predictor import Predictor
    return Predictor(ART, threads=2)


@pytest.mark.parametrize("text,intent,slots", [
    ("مجھے کل صبح سات بجے جگا دیں", "alarm_set", {"date": "کل", "time": "صبح سات بجے"}),
    ("کل کراچی میں موسم کیسا ہوگا", "weather_query", {"date": "کل", "place_name": "کراچی"}),
    ("play some music by atif aslam", "play_music", {"artist_name": "atif aslam"}),
])
def test_known_requests(predictor, text, intent, slots):
    r = predictor.predict_text(text)
    assert r["intent"] == intent and r["slots"] == slots and len(r["tags"]) == len(r["tokens"])


def test_batch_matches_single(predictor):
    texts = ["wake me up at seven tomorrow morning", "کل کراچی میں موسم کیسا ہوگا"]
    batch = predictor.predict_words([t.split() for t in texts])
    for t, b in zip(texts, batch):
        s = predictor.predict_text(t)
        assert (s["intent"], s["tags"]) == (b["intent"], b["tags"])
