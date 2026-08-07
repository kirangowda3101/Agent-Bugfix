from helpers import is_vowel

def count_vowels(word):
    count = 0
    for letter in word:
        if is_vowel(letter):
            count += 1
    return count
