import numpy as np


def adjustArraySize(downsamples, scalarmap, N):

    """
    Given a scalar map that has been spatially downsampled, adjust the size
    of the map to include the removed spatial points again.

    Parameters:
    - - - - -
        downsamples : indices in true map that were removed
        scalarmap : downsampled spatial map
        N : true number of indices
    """

    coords = list(set(range(0, N)).difference(set(downsamples)))

    if scalarmap.ndim == 1:
        cdata = np.zeros((N, 1))
    else:
        cdata = np.zeros((N, scalarmap.shape[1]))

    cdata[coords, :] = scalarmap

    return cdata.squeeze()


def adjustMatchingSize(mappings, s, sourceDS, t, targetDS):
    """

    Method to adjust the coordinates and size of a match.

    For example, with the HCP data, the original data is 32492 vertices.  We
    excluded the 2796 midline vertices in the surface matching step (so we
    included only 29696 total vertices).  The indices in the match correspond
    to positions between 1 and 29696, not 1 and 32492.

    This method corrects for the coordiante differences, and returns a
    matching of length sN.

    Expects that the matching has already been adjusted for the
    Matlab-to-Python indexing conversion (i.e. that 1 has been subtracted).

    The current method works for matching between surfaces with different
    numbers of surface vertices and different midline coordinates.

    Paramters:
    - - - - -
        mappings : output of DiffeoSpectralMatching (corr12, corr21)

        s : number of vertices in full source surface
        sourceDS : vector containing indices of midline for source surface

        t : number of vertices in full target surface
        targetDS : vector containing indices of midline for target surface

    Returns:
    - - - -
        adjusted : list where matching indices are converted to range of
                    surface vertices
    """

    # get coordinates of vertices in target surface not in the midline
    full_coords = list(set(range(0, t)).difference(set(targetDS)))

    # get list of coordinates of length matching
    match_coords = np.arange(0, len(mappings))

    # create dictionary mapping matching indices to non-midline indices
    # m is an index in the matching vector
    # f is an index on a surface mesh
    convert = dict((m, f) for m, f in zip(match_coords, full_coords))

    # convert the matching coordinates to non-midline coordinates
    adjusted = list(convert[x] for x in list(mappings))

    if len(adjusted) < s:

        cdata = np.zeros((s,))
        coords = list(set(range(0, s)).difference(set(sourceDS)))
        cdata[coords] = adjusted
        cdata[list(sourceDS)] = -1

        adjusted = cdata

    return adjusted.squeeze()
