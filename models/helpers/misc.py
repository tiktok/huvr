import numbers


def to_2tuple(x):
    return (x, x) if isinstance(x, numbers.Number) else x
