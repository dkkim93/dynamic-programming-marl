import numpy as np
import matplotlib.pyplot as plt


def vis(table):
    fig, ax = plt.subplots()

    ax.matshow(table, cmap='seismic')

    for (i, j), z in np.ndenumerate(table):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plt.show()
