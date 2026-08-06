from solution import running_total


def test_basic():
    assert running_total([1, 2, 3]) == [1, 3, 6]


def test_empty():
    assert running_total([]) == []
