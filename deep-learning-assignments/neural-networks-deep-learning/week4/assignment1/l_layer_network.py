import numpy as np

def init_params():
    parameters = {'W1': np.array([[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
           [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
           [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
           [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]]), 'b1': np.array([[ 1.38503523],
           [-0.51962709],
           [-0.78015214],
           [ 0.95560959]]), 'W2': np.array([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
           [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
           [-0.37550472,  0.39636757, -0.47144628,  2.33660781]]), 'b2': np.array([[ 1.50278553],
           [-0.59545972],
           [ 0.52834106]]), 'W3': np.array([[ 0.9398248 ,  0.42628539, -0.75815703]]), 'b3': np.array([[-0.16236698]])}

    return parameters


def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return X, parameters


def sigmoid(Z):

    A =  1. / (1 + np.exp(-Z))
    cache = dict()
    cache["Z"] = Z

    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = dict()
    cache["Z"] = Z
    return A, cache

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

    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + b
    #cache = (A, W, b)
    cache = dict()
    cache["A"] = A
    cache["W"] = W
    cache["b"] = b
    ### END CODE HERE ###

    assert (Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache

# GRADED FUNCTION: linear_activation_forward

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
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        ### END CODE HERE ###

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        ### END CODE HERE ###

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

if __name__ == '__main__':
    X, params = L_model_forward_test_case_2hidden()
    print(X)
    print (params)

    print (len(params) // 2)

    caches = dict()
    A = X
    L = len(params) // 2

    for l in range(1, L):
        print ("calculation layer " + str(l))
        A_prev = A


        W_l = params["W" + str(l)]
        b_l = params["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W_l, b_l, activation="relu")

        print (A)

        caches[str(l)] = (cache)

    #Output layer

    l+= 1
    W_l = params["W" + str(l)]
    b_l = params["b" + str(l)]


    print(W_l)
    print(b_l)
    print(A_prev)
    AL, cache = linear_activation_forward(A, W_l, b_l, activation="sigmoid")

    print(AL)

    caches[str(l)]  = cache
    print(caches)


'''
Layer sizes
input layer     - 5
layer 1         - 4
layer 2         - 3
output layer    - 1

W1 (4, 5)
b1 (4, 1)

W2 (3, 4)
b2 (3, 1)

W3 (1, 3)
b3 (1, 1)
'''
