import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.learning_rate = lr
        self.num_iterations = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.weight_history = [] 
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 

        y = y.reshape(-1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.num_iterations):
            y_pred = np.matmul(X, self.weights) + self.bias
            error = y_pred - y

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            self.weight_history.append(np.append(self.weights, self.bias))  # ← ekle (bias'ı da θ₀ olarak dahil edebilirsin)

            # Convergence Rule - Loss Difference
            if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.learning_rate * 1e-5:
                print(f"\nConverged at iteration {i}\n")
                break

            dw = (2 / n_samples) * np.matmul(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
        
        return self

    def predict(self, X):
        y_pred = np.matmul(X, self.weights) + self.bias
        return y_pred