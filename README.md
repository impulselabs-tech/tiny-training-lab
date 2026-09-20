# tiny-training-lab

Standard-library logistic regression: stable probabilities, gradient descent, seeded data and held-out evaluation.

## Run

Python 3.10+; standard library only. No installation, dataset download, or hardware needed.

~~~sh
python demo.py --seed 7 --epochs 300
python -m unittest discover -s tests -v
~~~

## Reading guide

- [Evaluation](docs/evaluation.md)
- [Gradients](docs/gradients.md)
- [Limitations](docs/limitations.md)

Default output: [examples/default-result.json](examples/default-result.json).
Tests cover analytical cases, validation, and end-to-end behavior.
See [HISTORY.md](HISTORY.md) for how the retrospective timeline was assembled.
