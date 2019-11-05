import numpy as np
import matplotlib.pyplot as plt


def plot_intensity_bars(Tresh, over_tresh, path):

    bar_heights = [over_tresh[0]]
    bar_labels = ['>'+str(Tresh[0])]

    for i in range(len(over_tresh) - 1):
        bar_heights.append(over_tresh[i+1] - over_tresh[i])
    bar_heights.append(3000 - over_tresh[-1])

    for i in range(len(Tresh) - 1):
        bar_labels.append(str(Tresh[i+1])+' - '+str(Tresh[i]))
    bar_labels.append('<'+str(Tresh[-1]))

    alignment = np.arange(len(over_tresh)+1)

    print(alignment)
    print(bar_heights)
    print(bar_labels)

    plt.bar(alignment,bar_heights)
    plt.xticks(alignment,bar_labels)
    plt.savefig(path)
    plt.show()