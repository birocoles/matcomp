import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import copy


def plot_matrices(
    matrices,
    size,
    tol,
    shape,
    colormap="turbo",
    bad_color="black",
    same_scale=True,
):

    num_matrices = len(matrices)
    assert num_matrices <= shape[0] * shape[1], "matrices does not match shape"

    cmap = copy(mpl.cm.get_cmap(colormap))
    cmap.set_bad(color=bad_color)

    if same_scale is True:
        vmax = []
        vmin = []
        for matrix in matrices:
            vmax.append(np.max(matrix))
            vmin.append(np.min(matrix))
        vmax = np.max(vmax)
        vmin = np.min(vmin)

    plt.figure(figsize=size)

    for i, matrix in enumerate(matrices):

        assert matrix.ndim == 2, "matrices must contain 2D arrays"
        plt.subplot(shape[0], shape[1], i + 1)
        matrix_masked = np.ma.masked_where(np.abs(matrix) < tol, matrix)
        if same_scale is True:
            plt.matshow(
                matrix_masked, fignum=0, cmap=cmap, vmin=vmin, vmax=vmax
            )
        else:
            plt.matshow(matrix_masked, fignum=0, cmap=cmap)
        if matrix.shape[0] == 1:
            plt.yticks([0])
        if matrix.shape[1] == 1:
            plt.xticks([0])
        plt.colorbar()

    plt.tight_layout()
    plt.show()
