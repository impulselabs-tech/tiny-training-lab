"""Reproduce a fixed grid of training experiments. Run from the repository root."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainlab import dataset, split, fit, predict, accuracy, loss

RATES = (0.025, 0.05, 0.1, 0.2, 0.4)
EPOCHS = (25, 50, 100, 200, 400)
CASE_COUNT = 500

def run(case):
    if not isinstance(case, int) or not 0 <= case < CASE_COUNT:
        raise ValueError("case must be an integer from 0 to 499")
    seed = case // 25
    rate = RATES[case % 5]
    epochs = EPOCHS[(case // 5) % 5]
    rows, labels = dataset(n=300, seed=seed)
    train_x, train_y, test_x, test_y = split(rows, labels, seed=seed + 101)
    weights, bias = fit(train_x, train_y, epochs=epochs, rate=rate)
    training = predict(train_x, weights, bias)
    held_out = predict(test_x, weights, bias)
    return {
        "case": case,
        "parameters": {"dataset_seed": seed, "split_seed": seed + 101,
                       "samples": 300, "epochs": epochs, "learning_rate": rate},
        "metrics": {"train_loss": loss(train_y, training), "test_loss": loss(test_y, held_out),
                    "train_accuracy": accuracy(train_y, training),
                    "test_accuracy": accuracy(test_y, held_out)},
        "model": {"weights": weights, "bias": bias},
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.case), indent=2, allow_nan=False))
