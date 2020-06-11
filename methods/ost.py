"""
Module that contains all the necessary function to calculate thresholds for hypothesis test, where all the data is
used for leaning and testing. The code works for tau_base and tau_ost.
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import chi as chi_stats
from cvxopt import matrix, solvers


def truncation(beta_star, tau, Sigma, accuracy=1e-6):
    """
    Compute
    :param beta_star: optimal projection of tau
    :param tau: vector of test statistics
    :param Sigma: Covariance matrix of test statistics
    :param accuracy: threshold to determine whether an entry is zero
    :return: Lower threshold of conditional distribution of beta_star^T tau
    """
    # dimension of data
    d = len(tau)
    # determine non-zero entries of betastar
    non_zero = [1 if beta_i > accuracy else 0 for beta_i in beta_star]
    # define the arguments of the maximization of V^-
    arguments = [(tau[i] * (beta_star @ Sigma @ beta_star) - (np.eye(1, d, i) @ Sigma @ beta_star) * (beta_star @ tau))
                 / (np.sqrt(Sigma[i][i]) * np.sqrt(beta_star @ Sigma @ beta_star) - np.eye(1, d, i) @ Sigma @ beta_star)
                 for i in range(len(tau)) if non_zero[i] == 0]
    # catch cases for which we have 0/0 and hence nan. We dont consider these
    arguments = np.array([argument if argument > -10e6 else -10e6 for argument in arguments])
    if len(arguments) == 0:
        return -10e6
    v_minus = np.max(arguments)
    return v_minus


def truncated_gaussian(var, v_minus, level):
    """
    Computes the (1-level) threshold of  a truncated normal (the original normal is assumed to be centered)
    :param var: variance of the original normal
    :param v_minus: lower truncation
    :param level: desired level
    :return:
    """
    # normalize everything
    lower = v_minus / np.sqrt(var)
    # compute normalization of the truncated section
    renormalize = 1 - norm.cdf(lower)
    if renormalize == 0:
        # force a reject
        return np.sqrt(var) * 10000
    assert renormalize > 0, "renormalize is not positive"

    threshold = np.sqrt(var) * norm.ppf(renormalize * (1 - level) + norm.cdf(lower))
    return threshold


def optimization(tau, Sigma, selection='continuous'):
    """
    optimizes the signal to noise ratio. If tau has at least one positive entry, we fix the nominator to some constant
    by setting beta^T tau = 1 and then optimize the denominator.
    If tau has only negative entries, the signal to noise ratio is given by the optimum of the discrete optimization
    :param tau: Signal
    :param Sigma: noise
    :param selection: discrete (select from base tests) / continuous (OST in canoncical form)
    :return: optimal vector beta_star
    """

    if np.max(tau) < 1e-6:
        # If all entries are negative, then for the continuous case we also select the best of the base tests
        selection = 'discrete'

    # determine dimensionality
    d = len(tau)
    if selection == 'continuous':
        tau = np.ndarray.tolist(tau)
        Sigma = np.ndarray.tolist(Sigma)

        # define quadratic program in cvxopt
        P = matrix(Sigma)
        q = matrix(np.zeros(d))
        G = matrix(np.diag([-1.] * d))
        h = matrix(np.zeros(d))
        A = matrix(np.array([tau]))
        b = matrix([1.])

        initialization = matrix([1.]*d)
        solvers.options['reltol'] = 1e-40
        solvers.options['abstol'] = 1e-10
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 10000
        sol = solvers.qp(P, q, G, h, A, b, initvals=initialization)

        beta_star = np.array(sol['x']).flatten()
        # normalize betastar
        beta_star = beta_star / np.linalg.norm(beta_star, ord=1)
        return beta_star
    else:
        arguments = tau / np.sqrt(np.diag(Sigma))
        # in case of division by zero, we do not consider it since it implies also that the nominator is zero
        arguments = np.array([argument if argument > -10e6 else -10e6 for argument in arguments])
        j = int(np.argmax(arguments))
        beta_star = [0] * d
        beta_star[j] = 1
        return np.array(beta_star)


def ost_test(tau, Sigma, alpha, selection='discrete', max_condition=1e-6, accuracy=1e-6, constraints='Sigma'):
    """
    Runs the full test suggested in our paper.
    :param tau: observed statistic
    :param Sigma: covariance matrix
    :param alpha: level of test
    :param selection: continuous/discrete (discrete is not extensively tested)
    :param max_condition: at which condition number the covariance matrix is truncated.
    :param accuracy: threshold to determine whether an entry is zero
    :param constraints: if 'Sigma'  we work with the constraints (Sigma beta) >=0. If 'positive' we work with beta >= 0
    :return: 1 (reject), 0 (no reject)
    """
    assert constraints == 'Sigma' or constraints == 'positive', 'Constraints are not implemented'
    # if the selection is discrete we dont want any transformations
    if selection == 'discrete':
        constraints = 'positive'

    # check if there are entries with 0 variance
    zeros = [i for i in range(len(tau)) if Sigma[i][i] < 1e-10]
    tau = np.delete(tau, zeros)
    Sigma = np.delete(Sigma, zeros, 0)
    Sigma = np.delete(Sigma, zeros, 1)

    if constraints == 'Sigma':
        # compute pseudoinverse to also handle singular covariances (see Appendix)
        r_cond = max_condition                    # parameter which precision to use
        Sigma_inv = np.linalg.pinv(Sigma, rcond=r_cond, hermitian=True)

        # use Remark 1 to convert the problem
        tau = Sigma_inv @ tau
        Sigma = Sigma_inv

    # Apply Theorem 1 in the canonical form with beta>=0 constraints
    beta_star = optimization(tau=tau, Sigma=Sigma, selection=selection)

    # determine active set
    non_zero = [1 if beta_i > accuracy else 0 for beta_i in beta_star]

    projector = np.diag(non_zero)
    effective_sigma = projector @ Sigma @ projector

    # Use the rank of effective Sigma to determine how many degrees of freedom the covariance has after conditioning
    # for non-singular original covariance, this is the same number as the number of active dimensions |mathcal{U}|,
    # however, for singular cases using the rank is the right way to go.
    tol = max_condition*np.max(np.linalg.eigvalsh(Sigma))
    r = np.linalg.matrix_rank(effective_sigma, tol=tol, hermitian=True)
    # go back to notation used in the paper
    l = r
    if l > 1:
        test_statistic = beta_star  @ tau / np.sqrt(beta_star @  Sigma @ beta_star)
        threshold = chi_stats.ppf(q=1-alpha, df=l)
    else:
        vminus = truncation(beta_star=beta_star, tau=tau, Sigma=Sigma, accuracy=accuracy)
        threshold = truncated_gaussian(var=beta_star@ Sigma @ beta_star, v_minus=vminus, level=alpha)
        test_statistic = beta_star @ tau

    if test_statistic > threshold:
        # reject
        return 1
    else:
        # cannot reject
        return 0
