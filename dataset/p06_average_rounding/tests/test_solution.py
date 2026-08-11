from solution import average

def test_average_rounds():
    assert average([1, 2, 4]) == 2.33

def test_average_whole():
    assert average([10, 20]) == 15.0
