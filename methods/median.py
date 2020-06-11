import numpy as np
from scipy.spatial import distance_matrix


def median(x, y):
    # Compute the median distance in the aggregated sample
    nmax = 500
    #  we only take part of the data to evaluate the median heuristic. This saves time and does not change our analysis
    if len(x) > nmax:
        x = x[:nmax]
    if len(y) > nmax:
        y = y[:nmax]
    if len(x.shape) == 1:
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))

    data = np.concatenate((x, y))
    distances = (distance_matrix(data, data))

    return np.median(distances)
