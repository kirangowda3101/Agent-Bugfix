from solution import count_vowels

def test_count_vowels():
    assert count_vowels("hello") == 2

def test_count_vowels_with_no_vowels():
    assert count_vowels("bcd") == 0