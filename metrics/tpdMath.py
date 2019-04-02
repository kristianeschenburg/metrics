import scipy
import numpy as np


def tpdVector(matr, diagonal=None):
    """
    Convert tpd matrix data to vector.

    Parameters:
    - - - - -
        matr : topological matrix, each index counts strength of adjacency
                between two labels
    """

    if diagonal:
        np.fill_diagonal(matr, 0)

    row_sums = matr.sum(axis=1)

    matr = matr / (1.*row_sums[:, np.newaxis])

    nans = np.isnan(matr)
    matr[nans] = 0

    vec = np.reshape(matr, np.product(matr.shape))

    return vec


def tpd(v1, v2):
    """
    Compute the topological distance between the left and right hemisphere.

    Parameters:
    - - - - -
        v1 : topological vector
        v2 : topological vector
    """

    metric = scipy.spatial.distance.cosine(v1, v2)

    return metric
