import networkx as nx
import numpy as np


def scalarDistance(scalarMap, surfaceGraph, samples, maxDistance):
    """
    Sample vertices from the surface graph. For each sampled vertex,
    compute the similarity of its features to vertices less than or equal
    to maxDistance away from it using the absolute difference as a measure of
    similarity.
    """

    nodes = surfaceGraph.nodes()
    sampleNodes = np.random.choice(nodes, size=samples, replace=False)

    distances = np.arange(maxDistance+1)
    distanceMap = {k: [] for k in distances}

    print('Computing distances for {:} sampled nodes'.format(samples))
    print('Max Distances: {:}'.format(maxDistance))

    for j, s in enumerate(sampleNodes):
        sds = nx.single_source_shortest_path_length(
            G=surfaceGraph, source=s, cutoff=maxDistance)

        for node, dist in sds.items():
            distanceMap[dist].append(np.abs(scalarMap[s]-scalarMap[node]))

    return distanceMap


def vectorDistance(featureMatrix, surfaceGraph, samples, maxDistance):

    """
    Sample vertices from the surface graph. For each sampled vertex,
    compute the similarity of its features to vertices less than or equal
    to maxDistance away from it using the correlation as a measure of
    similarity.
    """

    nodes = surfaceGraph.nodes()
    sampleNodes = np.random.choice(nodes, size=samples, replace=False)

    distances = np.arange(maxDistance+1)
    distanceMap = {k: [] for k in distances}

    print('Computing distances for {:} sampled nodes'.format(samples))
    print('Max Distances: {:}'.format(maxDistance))

    for j, s in enumerate(sampleNodes):
        sds = nx.single_source_shortest_path_length(
            G=surfaceGraph, source=s, cutoff=maxDistance)

        for node, dist in sds.items():
            cc = np.corrcoef(featureMatrix[s, :], featureMatrix[node, :])
            distanceMap[dist].append(cc[0, 1])

    return distanceMap
