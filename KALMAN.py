## 3. KALMAN

import numpy as np

class KalmanFilterReg:
    def __init__(self):
        self.x = np.array([0.0, 1.0])   # intercept, slope
        self.P = np.eye(2) * 1e3
        self.A = np.eye(2)
        self.Q = np.eye(2) * 1e-4
        self.R = np.array([[1.0]])

    def update(self, x, y):
        C = np.array([[1.0, x]])
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        self.x = self.x + (K @ (y - C @ self.x)).flatten()
        self.P = (np.eye(2) - K @ C) @ self.P

    def predict(self):
        self.P = self.A @ self.P @ self.A.T + self.Q

    def zscore(self, y, x):
        spread = y - (self.x[0] + self.x[1]*x)
        return (spread - np.mean(spread)) / np.std(spread)
