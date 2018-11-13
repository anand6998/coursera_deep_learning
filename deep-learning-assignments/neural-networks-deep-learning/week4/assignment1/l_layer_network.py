import numpy as np
from dnn_utils_v3 import relu, sigmoid, relu_backward, sigmoid_backward

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    # START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    # END CODE HERE ###

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        # START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros(shape=(layer_dims[l], 1))
        # END CODE HERE ###

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    # START CODE HERE ### (≈ 1 lines of code)
    cost = (-1. / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))
    # END CODE HERE ###

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    # START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + b
    # cache = (A, W, b)
    cache = dict()
    cache['A'] = A
    cache['W'] = W
    cache['b'] = b
    # END CODE HERE ###

    assert (Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        # START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        # END CODE HERE ###

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        # START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        # END CODE HERE ###

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = dict()
    cache.update(linear_cache)
    cache.update(activation_cache)
    # cache = {"linear": linear_cache, "activation": activation_cache}
    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = dict()
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        # START CODE HERE ### (≈ 2 lines of code)
        W_l = parameters["W" + str(l)]
        b_l = parameters["b" + str(l)]

        A, cache = linear_activation_forward(A_prev, W_l, b_l, activation="relu")
        caches[str(l)] = cache
        # END CODE HERE ###

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    # START CODE HERE ### (≈ 2 lines of code)
    # this is the final layer
    l += 1
    W_l = parameters["W" + str(l)]
    b_l = parameters["b" + str(l)]
    AL, cache = linear_activation_forward(A, W_l, b_l, activation="sigmoid")
    caches[str(l)] = cache
    # END CODE HERE ###

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']
    b = cache['b']

    m = A_prev.shape[1]
    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    if activation == 'relu':
        dZ = relu_backward(dA, cache)
        dA_prev, dW, db = linear_backward(dZ, cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, cache)
        dA_prev, dW, db = linear_backward(dZ, cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)

    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL)) + np.divide(1 - Y, 1 - AL)
    layer_cache = caches[str(L)]

    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = \
        linear_activation_backward(dAL, layer_cache, activation="sigmoid")

    for l in reversed(range(L - 1)):
        layer = str(l + 1)
        print('Computing layer ' + layer)

        layer_cache = caches[layer]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads['dA' + layer], layer_cache, activation='relu')

        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + layer] = dW_temp
        grads["db" + layer] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        W = parameters["W" + str(l + 1)]
        dW = grads["dW" + str(l + 1)]

        b = parameters["b" + str(l + 1)]
        db = grads["db" + str(l + 1)]
        parameters["W" + str(l + 1)] = W - learning_rate * dW
        parameters["b" + str(l + 1)] = b - learning_rate * db
        # END CODE HERE ###
    return parameters


if __name__ == '__main__':
    pass
