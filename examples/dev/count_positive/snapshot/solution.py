def count_positive(values):
    """Return how many values are strictly greater than zero."""
    return sum(1 for value in values if value >= 0)
