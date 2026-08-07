from solution import find_max

def test_find_max():
    assert find_max([3,9,2]) == 9

def test_find_max_number_at_beginning():
    assert find_max([10,4,7]) == 10

def test_find_max_number_at_end():
    assert find_max([1, 2, 5]) == 5

def test_find_max_number_single_element():
    assert find_max([7]) == 7

def test_find_max_number_negative():
    assert find_max([-1,-2,-3]) == -1

def test_find_max_negative_number():
    assert find_max([-5, -1, -9]) == -1
