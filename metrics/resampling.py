import numpy as np
import networkx as nx


def indexresample(adjacency, distance=[], samples=[], percentage=[]):
    """
    Bootstrap method to resample neighborhood indices.  For a given
    adjacency list, sample indices that are at most a distance=k units away
    from a vertex.

    Parameters:
    - - - - -
        adjacency: surface adjacency list
        distance: distance radius from vertex to sample from
        samples: number of resamples to take
        percentage: percentage of indices to resample from
    Returns:
    - - - -
        indices: target indices
        resampled: resampled indices
    """

    assert distance >= 1
    assert percentage <= 1 and percentage >= 0

    indices = adjacency.keys()

    if samples == []:
        samples = 1

    if percentage != []:
        vsamps = np.int32(percentage*len(indices))
        indices = np.random.choice(indices, vsamps, replace=False)

    distance = np.ceil(distance).astype(np.int32)
    G = nx.from_dict_of_lists(adjacency)

    resampled = np.zeros((len(indices), samples))

    for j, idx in enumerate(indices):
        lengths = nx.single_source_shortest_path_length(
            G, idx, cutoff=distance)
        lengths = {k: v for k, v in lengths.items() if v <= distance}
        randex = np.random.choice(lengths.keys(), samples, replace=True)
        resampled[j, :] = randex

    resampled = resampled.astype(np.int32)

    return [indices, resampled]


def metricresample(metric, samples):

    """
    Resample a metric vector using random resamples computed using
    indexresample.

    Parameters:
    - - - - -
        metric: array of scalar values describing mesh
        samples: indices to sample from
    """

    resamples = metric[samples]

    return resamples
