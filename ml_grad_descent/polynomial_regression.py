import numpy
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

X = 6 * numpy.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + numpy.random.randn(100, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y.ravel())

pyplot.plot(X, y.ravel(), "g.")
pyplot.plot(X, lin_reg.predict(X_poly), "r.")

pyplot.show()

print(lin_reg.predict(numpy.array([[2, 3], [3, 4], [1, 20]])))

cross_val_scores = cross_val_score(RandomForestRegressor(), X_poly, y.ravel(), scoring="neg_mean_squared_error", cv=5)
print("Cross-validation mean error: ", numpy.sqrt(-cross_val_scores))
