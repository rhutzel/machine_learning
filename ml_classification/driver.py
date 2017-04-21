import mnist_utils
import test_set_utils
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_predict


X_train, y_train, X_test, y_test = mnist_utils.import_mnist(
    'train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

stoch_grad_desc_cls = SGDClassifier(random_state=42)

y_train_predictions = cross_val_predict(stoch_grad_desc_cls, X_train, y_train_5, cv=3)

print("Confusion matrix: ", confusion_matrix(y_train_5, y_train_predictions))

y_scores = cross_val_predict(stoch_grad_desc_cls, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
test_set_utils.plot_precision_recall(precisions, recalls, thresholds)

