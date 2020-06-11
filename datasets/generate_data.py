import numpy as np
import pickle
import os

dirname = os.path.dirname(__file__)


def blobs(theta, samplesize, spots=(3, 3), sigma=[0.1, 0.3]):
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    gaussians = np.array([np.random.normal(0, sigma[0], samplesize), np.random.normal(0, sigma[1], samplesize)])
    data = rotMatrix @ gaussians
    shifts = [np.random.randint(0, spots[0], samplesize), np.random.randint(0, spots[1], samplesize)]
    data = np.add(data, shifts)
    return np.transpose(data)


def generate_samples(dataset, hypothesis, samplesize):
    assert dataset in ['diff_var', 'mnist', 'blobs'], 'unknown dataset'
    if dataset == 'diff_var':
        if hypothesis == 'alternative':
            x = np.random.normal(loc=0., scale=1., size=samplesize)
            y = np.random.normal(loc=0., scale=1.5, size=samplesize)
        if hypothesis == 'null':
            x = np.random.normal(loc=0., scale=1., size=samplesize)
            y = np.random.normal(loc=0., scale=1., size=samplesize)

    if dataset == 'blobs':
        if hypothesis == 'alternative':
            x = blobs(theta=0, samplesize=samplesize)
            y = blobs(theta=1.57, samplesize=samplesize)
        if hypothesis == 'null':
            x = blobs(theta=0, samplesize=samplesize)
            y = blobs(theta=0, samplesize=samplesize)

    if dataset == 'mnist':
        # Note: this is not how we used it for our experiments. It reloads the whole dataset every time. If you want to
        # run a lot of experiments with the mnist datset, consider loading the data once and then simply
        # drawing new random indices every iteration.

        # load data
        # you must run the file 'download_mnist.py' before. This creates the with the 7x7 images
        file = os.path.join(dirname, 'mnist_7x7.data')
        with open(file, 'rb') as handle:

            X = pickle.load(handle)
        # define the distributions
        if hypothesis == 'null':
            P = np.vstack((X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['8'], X['9']))
            Q = np.copy(P)
        if hypothesis == 'alternative':
            P = np.vstack((X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['8'], X['9']))
            Q = np.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))

        # sample randomly from the datasets
        idx_X = np.random.randint(len(P), size=samplesize)
        x = P[idx_X, :]
        idx_Y = np.random.randint(len(Q), size=samplesize)
        y = Q[idx_Y, :]

    return x, y