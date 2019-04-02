import numpy as np


def labelCounts(label, adjacencyList):

    """
    For each vertex, count the number of vertices with other labels that are
    adjacent to it.

    Parameters:
    - - - - -
    label : int, array
        label vector
    adjacencyList : SurfaceAdjacency
        adjacency list for surface mesh
    """

    values = list(set(np.unique(label)).difference({-1,0 }))

    aggregateCounts = {k: {h: 0 for h in values} for k in values}
    aggregateMatrix = np.zeros((len(values), len(values)))

    # loop over each unique label value
    for j, v in enumerate(values):

        # get indices of label
        idxv = np.where(label == v)[0]

        # loop over vertices with this label and count number of neighboring vertices
        # with different label values
        for ind in idxv:
            n = adjacencyList[ind]
            nCounts = neighborhoodCounts(n, label, values)

            for n in nCounts:
                aggregateCounts[v][n] += nCounts[n]

        counts = aggregateCounts[v].values()
        aggregateMatrix[j, :] = counts
    
    rowSums = aggregateMatrix.sum(axis=1)
    rowNormed = aggregateMatrix / rowSums[:, None]

    return [aggregateMatrix, rowNormed]


def neighborhoodCounts(subscripts, label, values):

    """
    Compute the number of neighbors of each label directly adjacent to a
    vertex.

    Parameters:
    - - - - -
    subscripts : list
        indices of directly-adjacent vertices / voxels
    reshaped : int, array
        label vector
    values : accepted label values
    """

    neighbors = list(label[subscripts])
    counts = {}.fromkeys(values)

    for v in values:
        counts[v] = neighbors.count(v)

    return counts
