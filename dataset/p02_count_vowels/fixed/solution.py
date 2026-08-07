def count_vowels(word):
    count = 0
    vowels = "aeiou"
    for letter in word:
        if letter in vowels:
            count += 1
    return count