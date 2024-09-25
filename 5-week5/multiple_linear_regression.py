class MultipleLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.1, n_iterations=1000, tolerance=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=True)
        n_samples, n_features = X.shape  # Correctly unpack shapes
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)

            # Gradient calculations
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update the parameters
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
            
            # Check for convergence
            if np.all(np.abs(dw) < self.tolerance):
                break
        
        return self

    def predict(self, X):
        X = check_array(X, ensure_2d=True)
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)