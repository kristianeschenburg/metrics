import numpy as np
from sklearn.metrics import jaccard_similarity_score as jss
from sklearn.metrics import adjusted_rand_score as ars
from scipy.stats import spearmanr


def dice(l1, l2):

    """
    Compute the Dice Coefficient between two vectors.
    Parameters:
    - - - - -
    l1, l2 : array
        cluster assignments
    """

    st1 = set(l1)
    st2 = set(l2)

    inter = len(st1.intersection(st2))
    union = len(st1)+len(st2)

    dice = inter / union

    return dice


def rand(l1, l2):

    """
    Compute the Adjusted Rand Index between two clusterings.
    Parameters:
    - - - - -
    l1, l2 : array
        cluster assignments
    """

    return ars(l1, l2)


def jaccard(l1, l2):

    """
    Compute the Jaccard Index between two clusterings.
    Parameters:
    - - - - -
    l1, l2 : array
        cluster assignments
    """

    return jss(l1, l2)


def regionalSimilarity(featureData, label, c='Pearson'):

    """
    Computes the regional homogeneity of a set of parcels.  For each region in
    a cortical parcellation, compute the mean correlation (Spearman or Pearson)
    of the feature vectors in that region.

    Parameters:
    - - - - -
        featureData : feature matrix
        label : cortical parcellation vector
    """

    assert c in ['Spearman', 'Pearson']

    uniqueLabels = list(set(label).difference({0, -1}))

    correlation = {}.fromkeys(uniqueLabels)

    for ul in uniqueLabels:
        idx = np.where(label == ul)[0]
        if len(idx) > 1:
            data = featureData[idx, :]

            indx = np.triu_indices(data.shape[0], k=1)

            if c == "Spearman":
                [r, p] = spearmanr(data.T)
            else:
                r = np.corrcoef(data)
            r = np.squeeze(r)

            correlation[ul] = np.nanmean(r[indx])
        else:
            correlation[ul] = 0

    return correlation


def maximalOverlap(source, target,):

    """
    For each region in a cortical map (source), compute region in other
    map (target) with which it overlaps the most.

    Parameters:
    - - - - -
    source : array
        cortical map whose overlap to quantify
    target : array
        cortical map to use as template
    Returns:
    - - - -

    """

    source_hot = label2hot(source)
    target_hot = label2hot(target)

    counts = source_hot.sum(0)[:, None]

    intersection = source_hot.T.dot(target_hot)
    intersection = intersection / counts

    max_overlap = np.argmax(intersection, axis=1)+1

    return max_overlap


def label2hot(label):
    """
    Convert labeling to one-hot matrix, where each sample is assigned to
    a single label.
    Parameters:
    - - - - -
    label : array
        clustering assignments
    Returns:
    - - - -
    one_hot : array
        binary array of size n_samples by n_labels
    """

    unique_labs = np.unique(label)[1:]
    one_hot = np.zeros((len(label), len(unique_labs)))

    for j, l in enumerate(unique_labs):
        one_hot[:, j] = (label == l)

    return one_hot
