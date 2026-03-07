import numpy as np

class Metrics:
    def mean_squared_error(self, y_true, y_pred):
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def mean_absolute_error(self, y_true, y_pred):
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        mae = np.mean(np.abs(y_true - y_pred))
        return mae

    def root_mean_squared_error(self, y_true, y_pred):
        mse = self.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse

    def r_squared(self, y_true, y_pred):
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        y_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def summary(self, y_true, y_pred):
        results = {
            "MSE": self.mean_squared_error(y_true, y_pred),
            "MAE": self.mean_absolute_error(y_true, y_pred),
            "RMSE": self.root_mean_squared_error(y_true, y_pred),
            "R2 Score": self.r_squared(y_true, y_pred)
        }
        return results
