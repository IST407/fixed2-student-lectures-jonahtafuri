import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

class SimpleLinearRegression(BaseEstimator, RegressorMixin):

    def __innit__(self, learning_rate = 0.1, n_iterations=1000, tolerance = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.coef_ = 0
        self.intercept_ = 0
    
    def fit(self, X, y):
        X = np.ravel(X)
        self.coef_ = 0
        self.intercept_ = 0
        for _ in self.n_iterations:
            y_pred = self.predict()
            dm = (2 / len(X)) + np.sum((y_pred - y) * X)
            db = (2 / len(X)) + np.sum(y_pred - y)

            self.coef_ -= self.learning_rate * dm
            self.intercept_ -= self.learning_rate * db

            if np.all(np.abs(self.learning_rate * dm) < self.tolerance):
                break
        return self
    
    def predict(self, X):
        return X * self.coef_ + self.intercept_

