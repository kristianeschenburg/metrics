import numpy as np


def kernel3D(size=3):
    """
    Generates a 3D cube voxel kernel of size^3.
    """

    kernels = []

    if np.mod(size, 2) != 1:
        raise ValueError('If specified, size must be odd.')

    s = (size-1)/2
    rg = np.arange(-s, s+1)

    for x in rg:
        for y in rg:
            for z in rg:

                kernel = np.eye(4, 4)

                if not [x, y, z] == [0, 0, 0]:

                    kernel[0:3, 3] = [x, y, z]
                    kernels.append(kernel)

    return kernels


def neighborhood(coordinate, kernel=None, size=3):

    """
    Get 26-neighborhood coordinates of voxel.

    Parameter:
    - - - -
        coordinates : list of [x,y,z] location in volume
    """

    if not kernel:
        kernel = kernel3D(size=size)

    c = coordinate + [1]

    indices = []
    for k in kernel:
        indices.append(np.dot(k, c))
    indices = np.asarray(indices).astype(np.int32)

    return [indices[:, 0], indices[:, 1], indices[:, 2]]
