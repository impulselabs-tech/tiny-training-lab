"""Binary classification with standard-library gradient descent."""
import math
import random

def sigmoid(x):
    """Stable sigmoid, including inputs with large absolute values."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


