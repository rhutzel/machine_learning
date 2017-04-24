from matplotlib import cm, pyplot
import numpy


def plot_precision_recall(precisions, recalls, thresholds):
    pyplot.plot(thresholds, precisions[:-1], "b--", label="Precision")
    pyplot.plot(thresholds, recalls[:-1], "g-", label="Recall")
    pyplot.xlabel("Threshold")
    pyplot.legend(loc="upper left")
    pyplot.ylim([0, 1])
    pyplot.show()


def plot_confusion_matrix_errors(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = matrix / row_sums
    numpy.fill_diagonal(matrix_normalized, 0)
    pyplot.matshow(matrix_normalized, cmap=cm.gray)
    pyplot.show()


def render_digit(digit_byte_array):
    digit_image = digit_byte_array.reshape(28, 28)
    pyplot.imshow(digit_image, cmap=cm.binary, interpolation='nearest')
    pyplot.axis('off')
    pyplot.show()

