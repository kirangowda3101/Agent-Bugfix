from solution import apply_discount

def test_apply_discount_100():
    assert apply_discount(100) == 85

def test_apply_discount_200():
    assert apply_discount(200) == 170
