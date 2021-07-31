import numpy as np
from scipy.stats import norm


class UtilScore:

    def __init__(self):
        self.design_points = -1+np.arange(1, 401)/100
        self.weight = norm.cdf((self.design_points-1.5)/0.4)

    def twCRPS(self, prediction, true_observations):
        observations = self.indicator(true_observations)
        twCRPS_res = 4*np.mean(np.multiply(np.square(prediction-observations),
                               self.weight[np.newaxis, :]))
        return twCRPS_res

    def benchmark_twCRPS(self, ecdf, true_observations):
        prediction = np.tile(ecdf.reshape((1, -1)),
                             (true_observations.shape[0], 1))
        return self.twCRPS(prediction, true_observations)

    def indicator(self, x):
        x_mat = np.tile(x.reshape((-1, 1)), (1, 400))
        indication = (x_mat <= self.design_points[np.newaxis, :]).astype(float)
        return indication

    def ecdf(self, x):
        x = x.ravel()
        x = x[~np.isnan(x)]
        if x.shape[0] == 0:
            print('Warning: no valid value to compute eCDF')
            return np.zeros((400,))
        bins = (np.arange(0, 401)/100-1)
        bins[0] = -10.0
        bins[-1] = 10.0
        hist, _ = np.histogram(x, bins=bins)
        c_hist = np.cumsum(hist)
        return (c_hist/c_hist[-1])

    def epdf(self, x):
        x = x.ravel()
        if x.shape[0] == 0:
            return np.zeros((400,))
        x = x[np.logical_not(np.isnan(x))]
        bins = (np.arange(0, 401)/100-1)
        bins[0] = -10.0
        bins[-1] = 10.0
        hist, _ = np.histogram(x, bins=bins)
        return (hist/hist.sum())
