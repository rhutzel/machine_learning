import numpy
import struct


# Non-working code from book, due to HTTP 500
#mnist = fetch_mldata('MNIST original')
#X, y = mnist["data"], mnist["target"]
#X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def import_mnist(train_images_file, train_labels_file, test_images_file, test_labels_file):
    with open(train_labels_file, 'rb') as labels_file:
        magic, num_items = struct.unpack(">II", labels_file.read(8))
        y_train = numpy.fromfile(labels_file, dtype=numpy.int8)
    with open(train_images_file, 'rb') as images_file:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", images_file.read(16))
        X_train_3d = numpy.fromfile(images_file, dtype=numpy.uint8).reshape(len(y_train), num_rows, num_cols)
    with open(test_labels_file, 'rb') as labels_file:
        magic, num_items = struct.unpack(">II", labels_file.read(8))
        y_test = numpy.fromfile(labels_file, dtype=numpy.int8)
    with open(test_images_file, 'rb') as images_file:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", images_file.read(16))
        X_test_3d = numpy.fromfile(images_file, dtype=numpy.uint8).reshape(len(y_test), num_rows, num_cols)

    # scikit-learn fit expects 2D numeric arrays. Reshape these 3D arrays.
    num_images, num_x, num_y = X_train_3d.shape
    X_train = X_train_3d.reshape((num_images, num_x * num_y))
    num_images, num_x, num_y = X_test_3d.shape
    X_test = X_test_3d.reshape((num_images, num_x * num_y))

    return X_train, y_train, X_test, y_test

