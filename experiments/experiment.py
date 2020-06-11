'''File to run the experiments in Figure 1'''

import torch
from numpy.random import normal
import numpy as np
from typing import Tuple, Dict, List, Callable
import argparse
from pathlib import Path
import pickle
import yaml
import os

from methods import ost, split, kernel, wald, median
from datasets.generate_data import generate_samples


def estimate_power(samplesize_list: list, hypothesis: str, level, runs,
                   exp_number: int, bandwidths_factors: List, methods: List, constraints,
                   max_condition=1e-6, add_linear=False, dataset='diff_var') -> float:
    """
    Method that runs experiments. Iterate over the paramenter exp_number to consider different methods and samplesizes.
    All the parameters can be controlled via the file 'config.yml'.
    :param samplesize_list: Samplesizes considered
    :param hypothesis: 'null' or 'alternative'
    :param level: usually set to 0.05
    :param runs: number of independent trials that are averaged over. We used 5000, but it takes a while
    :param exp_number: iterator for different experiments
    :param bandwidths_factors: factors for the gaussian kernels that are considered.
    :param methods: list containing the considered methods
    :param constraints: 'Sigma' => leads to the suggested OST. 'positive' uses the canonical constraints without remark 1
    :param max_condition: just to numerically stabilize in case of almost singular covariance (see Appendix of the paper)
    :param add_linear: whether or not to consider a linear kernel (True for d=2 and d=6 in our experiments)
    :param dataset: which dataset to consider ('diff_var', 'mnist', 'blobs'
    :return:
    """
    np.random.seed(1 + exp_number % len(samplesize_list))
    # first check if folder for output exists and is empty
    folder = 'results'
    folder = str(Path(folder).expanduser())
    assert os.path.exists(folder), 'folder to store data does not exist'
    path = folder + '/results_' + str(args.exp_number) + '.data'
    path = str(Path(path).expanduser())

    samplesize = samplesize_list[exp_number % len(samplesize_list)]
    method = methods[int(exp_number / len(samplesize_list))]
    # Assume that samplesize is dvisible by four
    if samplesize % 4 != 0:
        samplesize = samplesize - samplesize % 4

    # create a list to store outcomes of individual runs
    outcome = [0]*runs

    for i in range(runs):
        # print(i)
        x, y = generate_samples(dataset, hypothesis, samplesize)

        # define kernels based on median heuristic
        med = median.median(x, y)
        bandwidths = [med * (2**factor) for factor in bandwidths_factors]
        # we square the bandwidth, since our implementation takes the squares
        kernels = [kernel.PTKGauss(bandwidths[u]**2) for u in range(len(bandwidths))]
        # add linear kernel if wanted
        if add_linear:
            kernels.append(kernel.Kpoly(d=1))
        mmd = kernel.LinearMMD(kernels)

        if method[:5] == 'split':
            method_category = 'split'
            splitratio = float(method[5:])
        else:
            method_category = method
            splitratio = None
        if method_category == 'ost':
            tau, Sigma = mmd.estimate(x, y)
            outcome[i] = ost.ost_test(tau=tau, Sigma=Sigma, alpha=level, selection="continuous",
                                      max_condition=max_condition, constraints=constraints)
        if method_category == 'wald':
            tau, Sigma = mmd.estimate(x, y)
            outcome[i] = wald.wald_test(tau=tau, Sigma=Sigma, alpha=level, max_condition=max_condition)
        if method_category == 'split':
            # compute the index at which to split
            cutoff = int(len(x) * splitratio)
            # ensure that it is divisible by 4. Needed for the estimation part
            if cutoff % 4 != 0:
                cutoff = cutoff - cutoff % 4
            # split the data
            x_train, x_test = x[:cutoff], x[cutoff:]
            y_train, y_test = y[:cutoff], y[cutoff:]
            # evaluate Sigma once with the whole dataset
            trash, Sigma = mmd.estimate(x, y)

            tau_tr, trash = mmd.estimate(x_train, y_train)
            tau_te, trash = mmd.estimate(x_test, y_test)
            outcome[i] = split.split_test(tau_tr=tau_tr, tau_te=tau_te, Sigma=Sigma, alpha=level, selection='continuous',
                                          max_condition=max_condition, constraints=constraints)
        if method_category == 'naive':
            # do the split test with the full sample for training and testing without correction
            tau, Sigma = mmd.estimate(x, y)
            outcome[i] = split.split_test(tau_tr=tau, tau_te=tau, Sigma=Sigma, alpha=level, selection='continuous',
                                          max_condition=max_condition, constraints=constraints)

    # store the mean over the runs
    power = np.mean(outcome)
    print(power)
    results = {'samplesize': samplesize, 'power': power, 'method': method}

    # write results to file
    f = open(path, 'wb')
    pickle.dump(results, f)
    f.close()


if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # the process number determines the samplesize and is iterated over
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_number', default=None, type=int, required=False,
                        help='The parameter that indexes the experiment')
    args = parser.parse_args()
    estimate_power(**config, exp_number=args.exp_number)
