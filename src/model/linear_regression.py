import numpy as np

class LinearRegression():
    def __init__(self, lr=0.01, n_iter=1000):
        self.learning_rate = lr
        self.num_iterations = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 

        y = y.reshape(-1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.num_iterations):
            y_pred = np.matmul(X, self.weights) + self.bias
            error = y_pred - y

            dw = (2 / n_samples) * np.matmul(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        y_pred = np.matmul(X, self.weights) + self.bias
        return y_pred