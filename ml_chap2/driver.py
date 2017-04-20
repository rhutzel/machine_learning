from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
import numpy
import test_set_utils
import transform_utils

def main():
    full_data_frame = transform_utils.file_to_dataframe("housing.csv")

    print("\nOcean proximity value counts: \n", full_data_frame["ocean_proximity"].value_counts())
    print("\nData frame description: \n", full_data_frame.describe())
    #histograms(full_data_frame)
    #scatter_plot(full_data_frame, 0.1)

    # Split off a testing set.
    data_frame_with_id = full_data_frame.reset_index() # adds an index column
    train_set, test_set = test_set_utils.split_train_test_by_id(data_frame_with_id, 0.2, "index")
    print("Train: %s  /  Test: %s \n" % (train_set.shape, test_set.shape))

    # Display attribute correlations.
    print("Correlations: ", train_set.corr(), "\n")
    #visual_correlations(train_set)

    # A copy of the data without the the house value label.
    train_set_unlabeled = train_set.drop("median_house_value", axis=1)
    train_set_labels = train_set["median_house_value"].copy()

    # Prepare the non-numeric fields.
    numeric_attributes = list(train_set_unlabeled.drop("ocean_proximity", axis=1))
    categorical_attributes = ["ocean_proximity"]
    pipeline = transform_utils.transform_pipeline(categorical_attributes, numeric_attributes)
    train_set_prepared = pipeline.fit_transform(train_set_unlabeled)

    # How about Linear Regression?
    print("Linear regression RMSE: %d" %
          model_rmse(LinearRegression(), train_set_prepared, train_set_labels))
    # Expect around $70,000, which sucks.

    # How about a Decision Tree Regressor?
    print("Decision tree regressor RMSE: %d" %
          model_rmse(DecisionTreeRegressor(), train_set_prepared, train_set_labels))
    # Expect 0, which indicates hugely overfit modelling and the exact reason why test data is pulled aside.

    print("Random forest regressor RMSE: %d" %
          model_rmse(RandomForestRegressor(), train_set_prepared, train_set_labels))

    cross_val_rmses = model_cross_validation_scores(RandomForestRegressor(), train_set_prepared, train_set_labels)
    print("\nRandom forest cross-validated RMSEs:\n", cross_val_rmses,
          "\nMean: ", cross_val_rmses.mean(), "\nStdDev: ", cross_val_rmses.std())

    grid_search_params = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
    grid_search = GridSearchCV(RandomForestRegressor(), grid_search_params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_set_prepared, train_set_labels)
    print("\nRandom forest grid search best parameters:\n", grid_search.best_params_)


def model_rmse(model, training_set, labels) -> float:
    model.fit(training_set, labels)
    predictions = model.predict(training_set)
    return numpy.sqrt(mean_squared_error(labels, predictions))


def model_cross_validation_scores(model, training_set, labels):
    cross_val_scores = cross_val_score(model, training_set, labels, scoring="neg_mean_squared_error", cv=10)
    return numpy.sqrt(-cross_val_scores)  # Utility function vs cost function.


def histograms(data_frame):
    data_frame.hist(bins=50, figsize=(15,10))
    plt.show()


def scatter_plot(data_frame, alpha):
    data_frame.plot(kind="scatter", x="longitude", y="latitude", alpha=alpha)
    plt.show()


def visual_correlations(data_frame):
    scatter_matrix(data_frame[["median_house_value", "median_income"]], figsize=(12, 8))
    plt.show()

if __name__ == '__main__':
    main()

