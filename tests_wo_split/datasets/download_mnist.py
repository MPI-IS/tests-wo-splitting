from sklearn.datasets import fetch_openml
import pickle
import numpy as np


# load MNIST
def load_and_store():
    X, y = fetch_openml("mnist_784", return_X_y=True)
    # standardize X values
    X = X / 255
    digits = {}
    for i in range(10):
        digits[str(i)] = []
    for i in range(len(y)):
        digits[y[i]].append(X[i])

    path = "mnist.data"
    f = open(path, 'wb')
    pickle.dump(digits, f)
    f.close()


def downsample():
    with open('mnist.data', 'rb') as handle:
        X = pickle.load(handle)
    # X is a dictionary
    digits = {}
    for i in range(10):
        digits[str(i)] = []
    for i in range(10):
        current = np.array(X[str(i)])
        n = len(current)
        # make the dataset 2D again
        current = np.reshape(current, (n, 28, 28))

        current = np.reshape(current, (n, 7, 4, 7, 4))
        current = current.mean(axis=(2, 4))
        digits[str(i)] = np.reshape(current, (n, 49))

    path = "mnist_7x7.data"
    f = open(path, 'wb')
    pickle.dump(digits, f)
    f.close()


if __name__ == '__main__':
    load_and_store()
    downsample()
    pass

