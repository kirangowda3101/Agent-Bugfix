def running_total(numbers):
    result = []
    total = 0
    for n in numbers:
        total = total + n
        result.append(total)
    return result
