import numpy
from matplotlib import pyplot
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X = numpy.random.rand(100, 1)
y = 4 + 3 * X + numpy.random.randn(100, 1)
X_scaled = StandardScaler().fit_transform(X, y)
print("Shapes: ", X.shape, y.shape)

learning_rate = 0.1
epochs = 50
regressor = SGDRegressor(n_iter=epochs, penalty=None, eta0=learning_rate)

regressor.fit(X, y.ravel())
print("\nSGD unscaled intercept [", regressor.intercept_, "] coefficient [", regressor.coef_, "]")
print("Unscaled MSE: %.2f" % numpy.mean((regressor.predict(X) - y) ** 2))
pyplot.plot(X, y, "b.", label='Unscaled')
pyplot.plot(X, regressor.predict(X), "b-", label="Unscaled Prediction")

regressor.fit(X_scaled, y.ravel())
print("\nSGD scaled intercept [", regressor.intercept_, "] coefficient [", regressor.coef_, "]")
print("Scaled MSE: %.2f" % numpy.mean((regressor.predict(X_scaled) - y) ** 2))
pyplot.plot(X_scaled, y, "g.", label='Scaled')
pyplot.plot(X_scaled, regressor.predict(X_scaled), "g-", label="Scaled Prediction")

pyplot.show()
