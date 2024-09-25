class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, base = np.e): # e is euler's constant (for logarithmic transformation)
        self.base = base

    def fit(self, X, y=None): # no need to do anything here since we are doing log transfrom NOT NEEDED FOR ANYTHING
        return self
    
    def transform(self, X):
        X = np.array(X)
        if np.any(X <= 0):
            raise ValueError("Non positive values in sample")
        return np.log(X)
    
    def inverse_transfrom(self):
        return np.power(self.base, X)