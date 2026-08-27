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


def split(rows, labels, fraction=0.25, seed=11):
    """Return paired training and held-out data, without overlap."""
    if len(rows) != len(labels) or len(rows) < 2 or not 0 < fraction < 1:
        raise ValueError("invalid paired data or test fraction")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n = max(1, min(len(rows)-1, round(len(rows)*fraction)))
    test, train = order[:n], order[n:]
    return ([rows[i] for i in train], [labels[i] for i in train],
            [rows[i] for i in test], [labels[i] for i in test])


def validate(rows, labels=None):
    """Reject empty, ragged, or nonfinite observations."""
    if not rows or not rows[0]:
        raise ValueError("nonempty observations required")
    width = len(rows[0])
    if any(len(r) != width or not all(math.isfinite(v) for v in r) for r in rows):
        raise ValueError("finite rectangular data required")
    if labels is not None and (len(labels) != len(rows) or any(y not in (0, 1) for y in labels)):
        raise ValueError("one binary label per row required")
    return width


