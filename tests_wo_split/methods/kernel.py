"""
Module that implements the estimation of the linear MMDs and the covariance of the MMD estimates for different kernels
"""
import torch
import numpy as np


class Kpoly:
    """
    Implementation of polynomial kernel kernel
    """
    def __init__(self, d, normalized=False):
        self.d = d          # degree of the polynomial kernel

    def eval_lin(self, X, Y):
        """
        Evaluate only the relevant entries for the linear time mmd
        ----------
        X : n1 x d Torch Tensor
        Y : n2 x d Torch Tensor
        Return
        ------
        K : a n/2 list of entries.
        """
        if X.dim() == 1:
            dimension = 1
        else:
            dimension = len(X[0])

        if dimension > 1:
            return (torch.sum(X * Y, dim=1).view(1, -1))**self.d
        else:
            return (X * Y)**self.d


class PTKGauss:
    """
    Pytorch implementation of the isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        """
        sigma2: a number representing squared width
        """
        sigma2 = torch.tensor(sigma2)
        assert (sigma2 > 0).any(), 'sigma2 must be > 0. Was %s' % str(sigma2)
        self.sigma2 = sigma2

    def eval_lin(self, X, Y):
        """
        Evaluate only the relevant entries for the linear time mmd
        ----------
        X : n1 x d Torch Tensor
        Y : n2 x d Torch Tensor
        Return
        ------
        K : a n/2 list of entries.
        """
        sigma2 = torch.sqrt(self.sigma2 ** 2)
        if X.dim() == 1:
            dimension = 1
        else:
            dimension = len(X[0])

        if dimension > 1:
            sumx2 = torch.sum(X ** 2, dim=1).view(1, -1)
            sumy2 = torch.sum(Y ** 2, dim=1).view(1, -1)
            D2 = sumx2 - 2 * torch.sum(X * Y, dim=1).view(1, -1) + sumy2
        else:
            sumx2 = (X ** 2)
            sumy2 = (Y ** 2)
            D2 = sumx2 - 2 * torch.mul(X, Y) + sumy2

        K = torch.exp(-D2.div(2.0 * sigma2))
        return K


class LinearMMD:
    """
    To compute linear time MMD estimates and the covariance matrix of the asymptotic distribution of the linear time
    MMD for d different kernels.
    """

    def __init__(self, kernels):
        """
        :param kernels: list of kernels, which will be considered

        :returns
        mmd: linear time mmd estimates for all the kernels. Scaled with sqrt(n)
        Sigma: covariance matrix of the asymptotic normal distribution of linear mmd estimates
        """
        self.kernels = kernels

        # number of kernels considered
        self.d = len(kernels)

    def estimate(self, x_sample, y_sample):
        """
        Computes the linear time estimates of the MMD, for all kernels that should be considered. Further
        it computes the asymptotic covariance matrix of the linear time MMD for the kernels.
        The samplesize is taken into account on the side of the MMD, i.e., we estimate sqrt(n) MMD^2
        :param x_sample: data from P
        :param y_sample: data from Q
        :return:
        """
        if not isinstance(x_sample, torch.Tensor):
            # convert data to torch tensors
            x_sample = torch.tensor(x_sample)
            y_sample = torch.tensor(y_sample)
        assert list(x_sample.size())[0] == list(y_sample.size())[0], 'datasets must have same samplesize'

        # determine length of the sample
        size = list(x_sample.size())[0]
        # for linear time mmd assume that the number of samples is 2n. Truncate last data point if uneven
        size = size - size % 2
        n = int(size / 2)
        # define the
        x1, x2 = x_sample[:n], x_sample[n:size]
        y1, y2 = y_sample[:n], y_sample[n:size]

        # tensor of all functions h defined for the kernels
        h = torch.zeros(self.d, n)

        # compute values of h on the data
        for u in range(self.d):
            gram_xx = self.kernels[u].eval_lin(X=x1, Y=x2)
            gram_xy = self.kernels[u].eval_lin(X=x1, Y=y2)
            gram_yx = self.kernels[u].eval_lin(X=y1, Y=x2)
            gram_yy = self.kernels[u].eval_lin(X=y1, Y=y2)

            h[u] = gram_xx - gram_xy - gram_yx + gram_yy

        mmd = torch.sum(h, dim=1) / n
        Sigma = 1 / n * h.matmul(h.transpose(0,1)) - mmd.view(-1,1).matmul(mmd.view(1,-1))

        # We consider sqrt(n) * mmd. Therefore we will keep Sigma on a scale independent of n
        mmd = np.sqrt(n) * mmd

        return np.array(mmd), np.array(Sigma)
