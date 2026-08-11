def parse_scores(text):
    result = {}
    for pair in text.split(","):
        name, score = pair.split(":")
        result[name] = int(score)
    return result
