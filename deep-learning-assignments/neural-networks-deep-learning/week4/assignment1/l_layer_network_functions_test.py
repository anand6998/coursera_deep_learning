import unittest

import numpy as np

from l_layer_network import linear_forward, linear_activation_forward, L_model_forward, compute_cost, linear_backward, \
    linear_activation_backward, L_model_backward, update_parameters
from testCases_v5 import linear_forward_test_case, linear_activation_forward_test_case, \
    L_model_forward_test_case_2hidden, compute_cost_test_case, linear_backward_test_case, \
    linear_activation_backward_test_case, L_model_backward_test_case, update_parameters_test_case
import pprint


class L_LayerNetworkFunctionsTest(unittest.TestCase):

    def test_linear_forward(self):
        print('=== Linear Forward Test ===')
        A, W, b = linear_forward_test_case()

        Z, linear_cache = linear_forward(A, W, b)
        # print("Z = " + str(Z))

        print('Z = ' + str(Z))
        Z_test = np.array([[3.26295337, -1.23429987]])
        np.testing.assert_array_almost_equal(Z_test, Z, decimal=8, verbose=True)
        print('\n')

    def test_linear_activation_forward(self):
        print('=== Linear Activation Forward Test ===')
        A_prev, W, b = linear_activation_forward_test_case()

        A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
        print("With sigmoid: A = " + str(A))

        A_expected = np.array([[0.96890023, 0.11013289]])
        np.testing.assert_array_almost_equal(A_expected, A, decimal=8, verbose=True)

        A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
        print("With ReLU: A = " + str(A))

        A_expected = np.array([[3.43896131, 0.]])
        np.testing.assert_array_almost_equal(A_expected, A, decimal=8, verbose=True)
        print('\n')

    def test_L_model_forward(self):
        print('=== L Model Forward Test ===')
        X, parameters = L_model_forward_test_case_2hidden()
        L = (len(parameters) // 2)

        AL, caches = L_model_forward(X, parameters)
        print("AL = " + str(AL))
        print("Length of caches list = " + str(len(caches)))

        AL_expected = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
        np.testing.assert_array_almost_equal(AL_expected, AL, decimal=8, verbose=True)

        caches_expected = 3
        np.testing.assert_equal(caches_expected, len(caches))

        print('\n')

    def test_compute_cost(self):
        print('=== Compute Cost Test ===')
        Y, AL = compute_cost_test_case()
        cost = compute_cost(AL, Y)

        print('cost = ' + str(cost))
        cost_expected = 0.41493159961539694
        np.testing.assert_equal(cost_expected, cost)
        print('\n')

    def test_linear_backward(self):
        print('=== Linear Backward Test ===')

        dZ, cache = linear_backward_test_case()
        dA_prev, dW, db = linear_backward(dZ, cache)
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

        dA_prev_expected = np.array([
            [0.51822968, -0.19517421],
            [-0.40506361, 0.15255393],
            [2.37496825, - 0.89445391]
        ])

        dW_expected = np.array([[-0.10076895, 1.40685096, 1.64992505]])
        db_expected = np.array([[0.50629448]])

        np.testing.assert_array_almost_equal(dA_prev_expected, dA_prev, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(dW_expected, dW, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(db_expected, db, decimal=8, verbose=True)
        print('\n')

    def test_linear_activation_backward(self):
        print('=== Linear Activation Backward Test ===')
        dAL, linear_activation_cache = linear_activation_backward_test_case()

        dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="sigmoid")
        print("sigmoid:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db) + "\n")

        dA_prev_expected = np.array([[0.11017994, 0.01105339],
                                     [0.09466817, 0.00949723], [-0.05743092, -0.00576154]])

        dW_expected = np.array([[0.10266786, 0.09778551, -0.01968084]])
        db_expected = np.array([[-0.05729622]])

        np.testing.assert_array_almost_equal(dA_prev_expected, dA_prev, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(dW_expected, dW, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(db_expected, db, decimal=8, verbose=True)

        dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="relu")
        print("relu:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

        dA_prev_expected = np.array([[0.44090989, 0.], [0.37883606, 0.], [-0.2298228, 0.]])
        dW_expected = np.array([[0.44513824, 0.37371418, -0.10478989]])
        db_expected = np.array([[-0.20837892]])

        np.testing.assert_array_almost_equal(dA_prev_expected, dA_prev, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(dW_expected, dW, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(db_expected, db, decimal=8, verbose=True)
        print('\n')

    def test_L_model_backward(self):
        print('=== L Model Backward Test ===')
        AL, Y_assess, caches = L_model_backward_test_case()
        grads = L_model_backward(AL, Y_assess, caches)

        from pprint import pprint
        for (k, v) in grads.items():
            print(k + '->')
            pprint(v)
            print('\n')

        dW1_expected = np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167], [0., 0., 0., 0.],
                                 [0.05283652, 0.01005865, 0.01777766, 0.0135308]])
        db1_expected = np.array([[-0.22007063], [0.], [-0.02835349]])
        dA1_expected = np.array([[0.12913162, - 0.44014127], [-0.14175655, 0.48317296], [0.01663708, -0.05670698]])

        np.testing.assert_array_almost_equal(dW1_expected, grads['dW1'], decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(db1_expected, grads['db1'], decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(dA1_expected, grads['dA1'], decimal=8, verbose=True)

    def test_update_parameters(self):
        parameters, grads = update_parameters_test_case()
        parameters = update_parameters(parameters, grads, 0.1)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        print("W1 = " + pprint.pformat(W1, indent=4))
        print("b1 = " + pprint.pformat(b1, indent=4))
        print("W2 = " + pprint.pformat(W2, indent=4))
        print("b2 = " + pprint.pformat(b2, indent=4))

        W1_expected = np.array([[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
                                [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
                                [-1.0535704, -0.86128581, 0.68284052, 2.20374577]])
        b1_expected = np.array([[-0.04659241], [-1.28888275], [0.53405496]])
        W2_expected = np.array([[-0.55569196, 0.0354055, 1.32964895]])
        b2_expected = np.array([[-0.84610769]])

        np.testing.assert_array_almost_equal(W1_expected, W1, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(b1_expected, b1, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(W2_expected, W2, decimal=8, verbose=True)
        np.testing.assert_array_almost_equal(b2_expected, b2, decimal=8, verbose=True)


if __name__ == '__main__':
    unittest.main()
