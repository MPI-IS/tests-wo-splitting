from methods import ost, split, kernel, wald, median
import warnings
import numpy as np
import matplotlib.pyplot as plt

def pvalue(x: list, y: list, bandwidths_factors=[-2,-1,0,1,2], method='ost', constraints='Sigma', max_condition=1e-6) \
        -> float:
    """
    Method that runs experiments. Iterate over the paramenter exp_number to consider different methods and samplesizes.
    All the parameters can be controlled via the file 'config.yml'.
    :param x: Sample from P
    :param y: Sample from Q
    :param bandwidths_factors: factors for the gaussian kernels that are considered.
    :param methods: which method ('wald', 'ost', 'split0.1'...)
    :param constraints: 'Sigma' => leads to the suggested OST. 'positive' uses the canonical constraints without remark 1
    :param max_condition: just to numerically stabilize in case of almost singular covariance (see Appendix of the paper)
    :return: pvalue
    """
    if len(x) != len(y):
        warnings.warn('The sample should be of equal size. I will truncate the longer')
        n = min(len(x), len(y))
        x = x[:n-1]
        y = y[:n-1]

    samplesize = len(x)
    if samplesize % 4 != 0:
        samplesize = samplesize - samplesize % 4

    # use mediah heuristic to define the first kernel. We only implement Gaussian kernels here.
    med = median.median(x, y)
    bandwidths = [med * (2**factor) for factor in bandwidths_factors]
    # we square the bandwidth, since our implementation takes the squares
    kernels = [kernel.PTKGauss(bandwidths[u]**2) for u in range(len(bandwidths))]
    # compute linear time MMD
    mmd = kernel.LinearMMD(kernels)

    if method[:5] == 'split':
        method_category = 'split'
        splitratio = float(method[5:])
    else:
        method_category = method
        splitratio = None
    if method_category == 'ost':
        tau, Sigma = mmd.estimate(x, y)
        p = ost.ost_test(tau=tau, Sigma=Sigma, selection="continuous",
                         max_condition=max_condition, constraints=constraints, pval=True)
    if method_category == 'wald':
        tau, Sigma = mmd.estimate(x, y)
        p = wald.wald_test(tau=tau, Sigma=Sigma,  max_condition=max_condition, pval=True)
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
        p = split.split_test(tau_tr=tau_tr, tau_te=tau_te, Sigma=Sigma, selection='continuous',
                             max_condition=max_condition, constraints=constraints, pval=True)
    if method_category == 'naive':
        # do the split test with the full sample for training and testing without correction
        tau, Sigma = mmd.estimate(x, y)
        p = split.split_test(tau_tr=tau, tau_te=tau, Sigma=Sigma, selection='continuous',
                             max_condition=max_condition, constraints=constraints, pval=True)

    return p


if __name__ == '__main__':
    # to test the p values, we compute the histogram of p-values under the null. Should be uniformly distributed
    runs = 1000
    size = 1000
    p = []
    for i in range(runs):
        x = np.random.normal(0,1, size=size)
        y = np.random.normal(0,1, size=size)
        p.append(pvalue(x=x, y=y))
    plt.hist(p)
    plt.show()