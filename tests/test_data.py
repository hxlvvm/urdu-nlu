from nlu.data import IGNORE, align_labels, decode_word_tags, label_maps, parse_annotation, spans


def test_parse_english():
    tokens, tags = parse_annotation("wake me up at [time : five am] [date : this week]")
    assert tokens == ["wake", "me", "up", "at", "five", "am", "this", "week"]
    assert tags == ["O", "O", "O", "O", "B-time", "I-time", "B-date", "I-date"]


def test_parse_urdu():
    tokens, tags = parse_annotation("مجھے [date : اس ہفتے] [time : صبح کے پانچ بجے] جگا دیں")
    assert tokens == ["مجھے", "اس", "ہفتے", "صبح", "کے", "پانچ", "بجے", "جگا", "دیں"]
    assert tags == ["O", "B-date", "I-date", "B-time", "I-time", "I-time", "I-time", "O", "O"]


def test_parse_slot_at_edges_and_no_slots():
    assert parse_annotation("[device_type : lights] off") == (["lights", "off"], ["B-device_type", "O"])
    assert parse_annotation("hello there") == (["hello", "there"], ["O", "O"])


def test_label_maps_are_sorted_and_bio():
    rows = [{"intent": "b", "tags": ["B-time", "I-time"]}, {"intent": "a", "tags": ["O", "B-date"]}]
    intents, tags = label_maps(rows)
    assert intents == ["a", "b"]
    assert tags == ["O", "B-date", "I-date", "B-time", "I-time"]


def test_align_first_subword_only():
    tag2id = {"O": 0, "B-time": 1, "I-time": 2}
    word_ids = [None, 0, 1, 1, 1, 2, None]          # <s> w0 w1(3 pieces) w2 </s>
    out = align_labels(word_ids, ["O", "B-time", "I-time"], tag2id)
    assert out == [IGNORE, 0, 1, IGNORE, IGNORE, 2, IGNORE]


def test_decode_inverts_align():
    tag2id = {"O": 0, "B-time": 1, "I-time": 2}
    id2tag = {v: k for k, v in tag2id.items()}
    word_ids = [None, 0, 1, 1, 2, None]
    gold = ["O", "B-time", "I-time"]
    aligned = align_labels(word_ids, gold, tag2id)
    pred = [0 if a == IGNORE else a for a in aligned]
    assert decode_word_tags(word_ids, pred, id2tag, 3) == gold


def test_spans():
    assert spans(["set", "five", "am", "today"], ["O", "B-time", "I-time", "B-date"]) == {"time": "five am",
                                                                                         "date": "today"}
    assert spans(["x", "y"], ["I-time", "I-time"]) == {"time": "x y"}
