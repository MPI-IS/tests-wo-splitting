'''File to run the experiments in Figure 1'''

import argparse
from pathlib import Path
import pickle
from typing import List

import numpy as np
import yaml

from tests_wo_split.methods import ost, split, kernel, wald, median
from tests_wo_split.datasets.generate_data import generate_samples


def estimate_power(samplesize_list: list, hypothesis: str, level, runs,
                   exp_number: int, data_dir: Path,
                   bandwidths_factors: List, methods: List, constraints,
                   max_condition=1e-6, add_linear=False, dataset='diff_var') -> float:
    """
    Method that runs experiments. Iterate over the paramenter exp_number to consider different methods and samplesizes.
    All the parameters can be controlled via the file 'config.yml'.
    :param samplesize_list: Samplesizes considered
    :param hypothesis: 'null' or 'alternative'
    :param level: usually set to 0.05
    :param runs: number of independent trials that are averaged over. We used 5000, but it takes a while
    :param exp_number: iterator for different experiments
    :paran data_dir: directory containing the results
    :param bandwidths_factors: factors for the gaussian kernels that are considered.
    :param methods: list containing the considered methods
    :param constraints: 'Sigma' => leads to the suggested OST. 'positive' uses the canonical constraints without remark 1
    :param max_condition: just to numerically stabilize in case of almost singular covariance (see Appendix of the paper)
    :param add_linear: whether or not to consider a linear kernel (True for d=2 and d=6 in our experiments)
    :param dataset: which dataset to consider ('diff_var', 'mnist', 'blobs'
    :return:
    """
    np.random.seed(1 + exp_number % len(samplesize_list))

    filename = "results_{}.data".format(args.exp_number)
    path = data_dir.joinpath(filename)

    samplesize = samplesize_list[exp_number % len(samplesize_list)]
    method = methods[int(exp_number / len(samplesize_list))]
    # Assume that samplesize is dvisible by four
    if samplesize % 4 != 0:
        samplesize = samplesize - samplesize % 4

    # create a list to store outcomes of individual runs
    outcome = [0] * runs

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
            outcome[i] = wald.wald_test(tau=tau, Sigma=Sigma, alpha=level,
                                        max_condition=max_condition)
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
    with open(path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':

    #: Default directory containing the results
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        help="Path to the config file",
        type=str,
        default="config.yml")
    parser.add_argument(
        '-n', '--exp_number',
        help='The parameter that indexes the experiment',
        type=int,
        required=True)
    parser.add_argument(
        '-d', '--dir',
        help="Directory containing the results",
        type=str,
        default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Read config file
    with open(args.config, 'r') as fd:
        config = yaml.safe_load(fd)

    # Create data directory
    data_dir = Path(args.dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Run the experiments
    estimate_power(**config, exp_number=args.exp_number, data_dir=data_dir)
