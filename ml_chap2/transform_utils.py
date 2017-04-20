import numpy
import os
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler

rooms_index, household_index = 3, 6


def file_to_dataframe(filename):
    csv_path = os.path.join(os.getcwd(), filename)
    return pandas.read_csv(csv_path)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Scikit-Learn doesn't know how to handle Pandas DataFrames."""
    def __init__(self, attribute_names):
        self.atrribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.atrribute_names].values


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_index] / X[:, household_index]
        return numpy.c_[X, rooms_per_household]


def transform_pipeline(categorical_attribs, numeric_attribs):
    numeric_pipeline = Pipeline([
        ('selector', DataFrameSelector(numeric_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('selector', DataFrameSelector(categorical_attribs)),
        ('label_binarizer', LabelBinarizer())
    ])

    return FeatureUnion(transformer_list=[
        ('numeric_pipeline', numeric_pipeline),
        ('categorical_pipeline', categorical_pipeline)
    ])
