import kernel
import numpy as np


def labelCounts(volume, values, size=None):
    """
    For each v in values, count number of vertices of other labels that are
    adjacent to v.

    Parameters:
    - - - - -
        volume : data volume
        values : set of labels to count
    """

    values = list(set(values).difference({0}))

    krn = kernel.kernel3D(size=size)
    reshaped = np.reshape(volume, np.product(volume.shape))

    aggregateCounts = {k: {} for k in values}

    # this could be parallelized
    for v in values:

        currentCounts = np.zeros((len(values),))

        finds = np.where(volume == v)
        finds = np.asarray(finds).T

        instances = finds.shape[0]

        for i in np.arange(instances):
            coordinate = list(finds[i, :])
            inds = kernel.neighborhood(coordinate, krn, size)
            subs = np.ravel_multi_index(inds, volume.shape)

            ns_counts = neighborhoodCounts(subs, reshaped, values)
            currentCounts += ns_counts.values()

        currentCounts = np.int32(currentCounts)
        aggregateCounts[v] = dict(zip(values, currentCounts))

    return aggregateCounts


def neighborhoodCounts(subscripts, reshaped, values):
    """
    Compute the number of neighbors of each label directory adjacency to a
    vertex.

    Parameters:
    - - - - -
        subscripts : indices of directly-adjacency vertices
        reshaped : reshaped data volume
        values : accepted label values
    """

    labels = list(reshaped[subscripts])
    counts = {}.fromkeys(values)

    for v in values:
        counts[v] = labels.count(v)

    return counts


def tpdMatrix(aggCounts, values):
    """
    Compute the topological distance vector for a given image.

    Parameters:
    - - - - -
        aggCounts : aggregate label counts for a given image.
        values : values of interest
    """

    counts = np.zeros((len(values), len(values)))

    for i, k in enumerate(aggCounts):
        counts[i, :] = aggCounts[k].values()

    return counts
