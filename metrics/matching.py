import numpy as np

from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.metrics.pairwise import euclidean_distances

from metrics import homogeneity as hmg

def costMatrix(row_feats, col_feats, row_labels, col_labels, metric="Pearson"):

    """
    Compute the matching cost matrix between two label sets, given
    their features, labels, and a metric.  Costs are computed using either
    the Pearson correlation coefficient, the Dice coefficient.

    Parameters:
    - - - - -
        row_feats,col_feats : arrays of feature data corresponding to a
                                cortical map
        row_labels,col_labels : arrays of cortical map labels
        metric : metric to use to build a similarity matrix.
                The matrix index values will be mnipulated accordingly to
                generate positive, integer-valued costs.
    """

    # Get unique label values in non-moving and moving brain
    row_labs = np.asarray(list(set(row_labels).difference({0})))
    col_labs = np.asarray(list(set(col_labels).difference({0})))

    # Initialize cost matrix
    costMatrix = np.zeros((len(row_labs), len(col_labs)))

    # Compute pairwise costs between all label sets
    for i, r in enumerate(row_labs):
        indr = np.where(row_labels == r)[0]
        lr = len(indr)

        if metric in ["Spearman","Euclidean","Pearson"]:
            featr = row_feats[indr, :]

        for j, c in enumerate(col_labs):
            indc = np.where(col_labels == c)[0]
            
            if metric in ["Spearman","Euclidean","Pearson"]:
                featc = col_feats[indc, :]

            if metric == "Spearman":
                [rVal, pVal] = spearmanr(featr, featc, axis=1)
                rVal = 1-rVal[lr:, 0:lr]
            elif metric == "Pearson":
                rVal = np.mean(cdist(featr, featc))
            elif metric == "Euclidean":
                rVal = euclidean_distances(featr, featc)
            elif metric == "Dice":
                rVal = 1-hmg.dice(indr, indc)

            costMatrix[i, j] = rVal

    return [row_labs, col_labs, costMatrix]


def linear_assignment(row_list, col_list, costMatrix):
    """
    Compute the linear assignment between two label sets.

    Parameters:
    - - - - -
        row_list : list of label values in non-moving brain
        col_list : list of label values in moving brain
        costMatrix : matrix of costs between labels in each brain
    """

    # Compute linear assignment
    ar, ac = lsa(costMatrix)

    rows = row_list[ar]
    cols = col_list[ac]

    non_mapped = set(col_list).difference(set(cols))

    # Remap assignment indices to true label values
    remapped = dict(zip(rows, cols))
    unmapped = list(non_mapped)

    return remapped, unmapped


def linearAssignmentParcellation(col_labels, label_mapping, slabels):
    """
    Generate the new cortical map, based on label to label assignments.

    Parameters:
    - - - - -
        col_labels : original cortical map vector of moving brain
        label_mapping : vector matching labels in the moving brain to labels
                        in the stable brain
        unmapped: labels in source brain that were not mapped to any label
    """

    z = np.zeros((len(col_labels),))

    for k, v in label_mapping.items():
        indv = np.where(col_labels == v)[0]
        z[indv] = k

    maxt = np.max(z)
    inds = np.where(col_labels>0)[0]
    zeros = inds[(z[inds]==0)]

    leftovers = np.unique(col_labels[zeros])

    for j,left in enumerate(leftovers):
        indlft = np.where(col_labels == left)
        z[indlft] = maxt + j + 1

    return z
