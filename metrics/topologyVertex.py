import numpy as np


def labelCounts(label, values, adjacencyList):

    """
    For each v in values, count number of vertices of other labels that are
    adjacent to v.

    Parameters:
    - - - - -
        label : label array
        values : values to compute neighborhood structure of
        adjacencyList : adjacency structure for surface corresponding to
                        cortical map

    """

    values = list(set(values).difference({0}))

    aggregateCounts = {k: {h: 0 for h in values} for k in values}
    aggregateMatrix = np.zeros((len(values), len(values)))

    for j, v in enumerate(values):

        idxv = np.where(label == v)[0]

        for ind in idxv:
            n = adjacencyList[ind]
            nCounts = neighborhoodCounts(n, label, values)

            for n in nCounts:
                aggregateCounts[v][n] += nCounts[n]

        aggregateMatrix[j, :] = aggregateCounts[v].values()
        row_sums = aggregateMatrix.sum(axis=1)
        rowNorm = aggregateMatrix / (1.*row_sums[:, np.newaxis])

    return [aggregateCounts, rowNorm]


def neighborhoodCounts(subscripts, reshaped, values):
    """
    Compute the number of neighbors of each label directory adjacency to a
    vertex.

    Parameters:
    - - - - -
        subscripts : indices of directly-adjacency vertices / voxels
        reshaped : label vector
        values : accepted label values
    """

    labels = list(reshaped[subscripts])
    counts = {}.fromkeys(values)

    for v in values:
        counts[v] = labels.count(v)

    return counts
