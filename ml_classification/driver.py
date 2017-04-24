import mnist_utils
import numpy
import test_set_utils
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

X_train, y_train, X_test, y_test = mnist_utils.import_mnist(
    'train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)
stoch_grad_desc_cls = SGDClassifier(random_state=42)
y_scores = cross_val_predict(stoch_grad_desc_cls, X_train, y_train_5, cv=3, method="decision_function")

#precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#test_set_utils.plot_precision_recall(precisions, recalls, thresholds)

# Decide that a threshold of around 70,000 will get really good precision.
print("\n70,000 Precision: ", precision_score(y_train_5, (y_scores > 70000)),
      "\n70,000 Recall: ", recall_score(y_train_5, (y_scores > 70000)))

# Simply scaling the inputs really helps accuracy!
X_train_scaled = StandardScaler().fit_transform(X_train.astype(numpy.float64))

#y_scores_scaled = cross_val_score(stoch_grad_desc_cls, X_train_scaled, y_train, cv=3, scoring="accuracy")
#print("\nCross-value scores: ", y_scores, " --> Scaled inputs: ", y_scores_scaled)

#y_train_predictions_5 = cross_val_predict(stoch_grad_desc_cls, X_train_scaled, y_train_5, cv=3)
#print("Confusion matrix (5 vs non-5): ", confusion_matrix(y_train_5, y_train_predictions_5))

# Look for the weak comparisons among the errors
y_train_predictions = cross_val_predict(stoch_grad_desc_cls, X_train_scaled, y_train, cv=3)
#test_set_utils.plot_confusion_matrix_errors(confusion_matrix(y_train, y_train_predictions))

# 3s and 5s are often confused
X_3s_predicted_as_5s = X_train[(y_train == 3) & (y_train_predictions == 5)]
test_set_utils.render_digit(X_3s_predicted_as_5s[1])
