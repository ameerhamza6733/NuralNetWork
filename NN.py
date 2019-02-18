import numpy as np


def sig(x):
    return 1 / (1 + np.exp(-x))


def sig_dritaive(x):
    return x * (1 - x)


input = np.array([[0, 0, 1],
                  [1, 1, 1],
                  [1, 0, 1],
                  [0, 1, 1]])
output_layer = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
waight = 2 * np.random.random((3, 1)) - 1

for iter in range(10000):
    input_layer = input

    output = sig(np.dot(input_layer, waight))

    error = output_layer - output

    waight_updating = error * sig_dritaive(output)

    waight += np.dot(output_layer.T, waight_updating)


test_input = np.array([[0, 0 , 0]])
print(sig(np.dot(test_input, waight)))
