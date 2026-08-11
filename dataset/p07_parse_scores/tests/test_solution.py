from solution import parse_scores

def test_parse_scores_two():
    assert parse_scores("alice:90,bob:85") == {"alice": 90, "bob": 85}

def test_parse_scores_one():
    assert parse_scores("carol:70") == {"carol": 70}
