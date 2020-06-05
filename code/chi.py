"""
Module that implements the Wald test. To do this we use that
\sqrt(n) \Sigma^(-1/2) * MMD^2  ~ N(mu, 1)
"""

import numpy as np
from scipy.stats import chi


def preprocessing(tau, Sigma, max_condition=1e-6):
    """
    Check if Sigma is singular and remove redundant features
    :param tau:
    :param Sigma:
    :param max_condition: determines at which threshold eigenvalues are considered as 0
    :return:
    """

    # First redefine features such that all have unit variance
    stds = np.sqrt(np.diag(Sigma))
    tau = np.divide(tau, stds)
    normalization = np.outer(stds, stds)
    Sigma = np.divide(Sigma, normalization)

    # compute eigendecomposition
    (w, v) = np.linalg.eigh(Sigma)
    w_max = w[-1]
    threshold = max_condition * w_max
    min_index = np.min([i for i in range(len(w)) if w[i] > threshold])
    if min_index > 0:
        Sigma_new = np.transpose(v[:, min_index:]) @ Sigma @ v[:, min_index:]
        tau_new = np.transpose(v[:, min_index:]) @ tau
        return tau_new, Sigma_new
    else:
        return tau, Sigma



def chi_test(tau, Sigma, alpha, max_condition=1e-6):
    """
    Test based on the chi_d distribution.
    :param tau: observed test statistics (scaled with sqrt(n)
    :param Sigma: observed covariance matrix
    :param alpha: level of the test
    :param max_condition: determines at which threshold eigenvalues are considered as 0
    :return: level of the test
    """
    # instead of regularizing we preprocess Sigma and tau to get rid of 0 eigenvalues
    tau, Sigma = preprocessing(tau, Sigma, max_condition=max_condition)
    d = len(tau)
    # compute matrix inverse
    Sigma_inv = np.linalg.inv(Sigma)

    # below quantity is asymptotically standard normal
    t_obs = np.sqrt(tau @ Sigma_inv @ tau)

    # compute the 1-alpha quantile of the chi distribution with d degrees of freedom
    threshold = chi.ppf(q=1-alpha, df=d)
    if t_obs > threshold:
        return 1
    else:
        return 0
