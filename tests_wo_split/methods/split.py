"""
Module that implements the two-sample with data splitting
"""
from scipy.stats import norm
import numpy as np
from . import ost


def split_test(tau_tr, tau_te, Sigma,  alpha=0.05, selection='continuous',  max_condition=1e-6, constraints='Sigma',
               pval=False):
    """
    Function that computes the two samples test with data splitting. It already assumes that the data was split.
    :param tau_tr: observation used for training
    :param tau_te: observation used for testing
    :param Sigma: estimator of covariance matrix
    :param alpha: level for the test
    :param selection: If discrete: selects optimal kernel. If continuous: learns optimal convex combination
    :param max_condition: at which condition number the covariance matrix is truncated.
    :param constraints: if 'Sigma'  we work with the constraints (Sigma beta) >=0. If 'positive' we work with beta >= 0
    :param pval: if true, returns the conditional p value instead of the test result
    :return: rejection 0 or 1
    """
    assert constraints == 'Sigma' or constraints == 'positive', 'Constraints are not implemented'
    # if the selection is discrete we dont want anyu transformations
    if selection == 'discrete':
        constraints = 'positive'

    if constraints == 'Sigma':
        r_cond = max_condition                    # parameter which precision to use
        Sigma_inv = np.linalg.pinv(Sigma, rcond=r_cond, hermitian=True)
        tau_tr = Sigma_inv @ tau_tr
        Sigma = Sigma_inv

    beta_star = ost.optimization(tau=tau_tr, Sigma=Sigma, selection=selection)

    # used test statistics
    if constraints == 'Sigma':
        tau_te = Sigma_inv @ tau_te
        # we already changed Sigma = Sigma_inv above

    t_obs = beta_star @  tau_te
    threshold = np.sqrt(beta_star @ Sigma @ beta_star) * norm.ppf(q=1-alpha)

    if not pval:
        if t_obs > threshold:
            return 1
        else:
            return 0
    else:
        # compute the p value
        return 1 - norm.cdf(beta_star@tau_te / np.sqrt(beta_star@Sigma@beta_star))
