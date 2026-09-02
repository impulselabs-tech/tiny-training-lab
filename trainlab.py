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


def predict(rows, weights, bias):
    """Return class-one probabilities."""
    if validate(rows) != len(weights) or not all(math.isfinite(v) for v in [*weights, bias]):
        raise ValueError("invalid model parameters")
    return [sigmoid(sum(w*x for w,x in zip(weights,row))+bias) for row in rows]


def loss(labels, probabilities):
    """Mean clipped binary cross entropy."""
    if not labels or len(labels) != len(probabilities) or any(y not in (0,1) for y in labels):
        raise ValueError("paired binary targets required")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must be finite and in [0,1]")
    total = 0
    for y,p in zip(labels, probabilities):
        p = max(1e-15, min(1-1e-15,p))
        total -= y*math.log(p)+(1-y)*math.log1p(-p)
    return total/len(labels)


def fit(rows, labels, epochs=300, rate=0.2):
    """Full-batch gradient descent for logistic regression."""
    width = validate(rows, labels)
    if not isinstance(epochs,int) or epochs < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("positive epochs and finite positive rate required")
    weights, bias = [0.0]*width, 0.0
    for _ in range(epochs):
        errors = [p-y for p,y in zip(predict(rows,weights,bias),labels)]
        gradients = [sum(e*r[j] for e,r in zip(errors,rows))/len(rows) for j in range(width)]
        weights = [w-rate*g for w,g in zip(weights,gradients)]
        bias -= rate*sum(errors)/len(rows)
    return weights,bias


