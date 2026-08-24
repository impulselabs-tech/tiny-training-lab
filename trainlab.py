"""Binary classification with standard-library gradient descent."""
import math
import random

def sigmoid(x):
    """Stable sigmoid, including inputs with large absolute values."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def dataset(n=300, seed=7):
    """Generate a noisy linear boundary in two dimensions."""
    if n < 2:
        raise ValueError("n must be at least two")
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        rows.append([x, y])
        labels.append(int(1.5*x-y+rng.gauss(0, 0.15) > 0))
    return rows, labels


