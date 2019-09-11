# Neural Network Multiclass Classification-Octave
This project is part of the **Machine Learning** course offered by **Andrew Ng**.
# nnCostFunction.m -- *Script File*
**nnCostFunction.m** file has `nnCostFunction()` method, that is used to find cost **J** and **gradient** for neural network. The return values of this method are **J**  and **gradient** . **gradient** has two gradient parameters `Theta1_grad (25 x 401)` and  `Theta2_grad (10 x 26)` which are `unrolled` into a long vector. The method looks like in octave `function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels,X, y, lambda)`, where **X** is `m x (n+1)` matrix, **y** is `m x 1` matrix, **lambda** is regularization parameter, **nn_params** is the `unrolled` theta parameter which contains `Theta1` and `Theta2`, **hidden_layer_size** is the number of units in the hidden layer, and **num_labels** is the number of classes.
# sigmoidGradient.m -- *Script File*
It computes the gradient of sigmoid function. Input can be vector or matrix. `sigmoidGradient()` method is used in backpropogation algorithm in `nnCostFunction()` method.
# predict.m -- *Script File*
*predict.m** file has `predict()` method, that is used to find **p**, where **p** `(m x1)` is a vector of predicted classes for training data. The method looks like in octave `function p = predict(Theta1, Theta2, X) `, where **X** is `m x n` matrix, **Theta1** is `25 x (n+1)`  and **Theta1** is `(k x 26)`.
# randInitializeWeights.m -- *Script File*
When training neural networks, it is important to randomly initialize the parameters for symmetry breaking. One effective strategy for random initialization is to randomly select values for Θ(l) uniformly in the range [− Einit, + Einit]. Einit = 0.12. This range of values ensures that the parameters are kept small and makes the learning more efficient.
# sigmoid.m -- *Script File*
**sigmoid.m** file has `sigmoid()` method, that is used to compute sigmoid of input vector/matrix.
# displayData.m -- *Script File*
To display what hidden layer is doing.
# checkNNGradients.m -- *Script File*
To ensure that packpropogation algorithm is correct.
# computeNumericalGradient.m -- *Script File*
Compute numarical gradient.


