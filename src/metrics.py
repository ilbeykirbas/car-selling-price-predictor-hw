import numpy as np

def mserror(self, y, y_pred):
    y = np.array(y).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    mse = np.mean((y - y_pred) ** 2)
    return mse

def maerror(self, y, y_pred):
    y = np.array(y).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    mae = np.mean(np.abs(y - y_pred))
    return mae

def rmserror(self, y, y_pred):
    mse = self.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def r_squared(self, y, y_pred):
    y = np.array(y).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
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
