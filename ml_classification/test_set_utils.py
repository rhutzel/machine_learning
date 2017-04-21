from matplotlib import pyplot


def plot_precision_recall(precisions, recalls, thresholds):
    pyplot.plot(thresholds, precisions[:-1], "b--", label="Precision")
    pyplot.plot(thresholds, recalls[:-1], "g-", label="Recall")
    pyplot.xlabel("Threshold")
    pyplot.legend(loc="upper left")
    pyplot.ylim([0, 1])
    pyplot.show()

