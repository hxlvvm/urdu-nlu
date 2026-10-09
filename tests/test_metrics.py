from nlu.metrics import per_intent, score


def rows():
    return [{"intent": "alarm_set", "tags": ["O", "B-time"]}, {"intent": "play_music", "tags": ["B-artist", "O"]}]


def test_perfect():
    m = score(rows(), ["alarm_set", "play_music"], [["O", "B-time"], ["B-artist", "O"]])
    assert m == {"n": 2, "intent_acc": 1.0, "slot_f1": 1.0, "exact_match": 1.0}


def test_exact_match_needs_intent_and_all_slots():
    m = score(rows(), ["alarm_set", "play_music"], [["O", "B-time"], ["O", "O"]])
    assert m["intent_acc"] == 1.0 and m["exact_match"] == 0.5 and m["slot_f1"] < 1.0


def test_per_intent_reports_top_confusion():
    pi = per_intent(rows(), ["alarm_set", "alarm_set"])
    assert pi["play_music"] == {"n": 1, "acc": 0.0, "top_confusion": "alarm_set"}
    assert pi["alarm_set"]["acc"] == 1.0
