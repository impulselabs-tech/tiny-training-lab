# Gradients

For p = sigmoid(w dot x + b), the cross-entropy gradient for each weight is (p-y) times its feature. The bias gradient is p-y. Average over the training batch and subtract rate times the gradient. Zero initialization is appropriate for logistic regression, but not for symmetric hidden units.
