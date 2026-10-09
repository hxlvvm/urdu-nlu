"""Classical intent baseline: TF-IDF (word + character n-grams) with logistic regression.

Gives the fine-tuned transformer something honest to beat. Intent only; slots need a sequence model.
  python -m nlu.baselines
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from nlu.data import load_locale

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/1.1/data"


def fit(rows):
    feats = FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                          ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True))])
    clf = make_pipeline(feats, LogisticRegression(C=10, max_iter=2000))
    return clf.fit([r["text"] for r in rows], [r["intent"] for r in rows])


def acc(clf, rows):
    pred = clf.predict([r["text"] for r in rows])
    return float((pred == [r["intent"] for r in rows]).mean())


def main():
    data = {loc: load_locale(RAW, loc) for loc in ("en-US", "ur-PK")}
    part = lambda loc, p: [r for r in data[loc] if r["partition"] == p]  # noqa: E731
    out = {}
    for name, train_locs in (("en_ur", ["en-US", "ur-PK"]), ("en_only", ["en-US"]), ("ur_only", ["ur-PK"])):
        clf = fit([r for loc in train_locs for r in part(loc, "train")])
        out[name] = {loc: {"intent_acc": acc(clf, part(loc, "test"))} for loc in ("en-US", "ur-PK")}
        print(name, out[name], flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/baseline_tfidf.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
