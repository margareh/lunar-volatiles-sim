# date:     12-5-2025
# author:   margaret hansen
# purpose:  generate nss observations for ice depth and fraction

import pandas as pd
import numpy as np
from torch import Tensor

from scipy.interpolate import RectBivariateSpline, SmoothBivariateSpline

# Normalization class for normalizing data
class norm():

    def __init__(self, x, eps=1e-6):
        self.x = x
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True) + eps
        self.x_norm = (self.x - self.mean) / self.std

    def norm(self, x):
        if isinstance(x, Tensor):
            m = Tensor(self.mean)
            s = Tensor(self.std)
        else:
            m = self.mean
            s = self.std
        return (x - m) / s

    def invert(self, x_norm, var=False):
        m = 0 if var else self.mean
        return (x_norm * self.std) + m


# NSS sensor model class
# contains ability to do forward (get wt % and depth given NSS obs) and 
#   inverse (simulate NSS obs given wt % and depth) modeling
class NSS():

    def __init__(self, file, nss_norm=False, depth_norm=False):
        # load file with sensor data
        data_pd = pd.read_csv(file)
        self.data = data_pd.to_numpy() # wt pct, depth, det1, det2
        self.nss_norm=nss_norm
        self.depth_norm=depth_norm

        # create interpolation grid
        self.setup()

    # forward model (depth and wt % from observations)
    def forward(self, obs1, obs2):
        pred_d = self.depth(obs1, obs2, grid=False)
        pred_wt = self.wt_pct(obs1, obs2, grid=False)
        return pred_d, pred_wt

    # inverse model (simulated obs from depth and wt %)
    def inverse(self, wt_pct, depth, noise=False):
        pred1 = self.det1(depth, wt_pct, grid=False)
        pred2 = self.det2(depth, wt_pct, grid=False)
        # simulated poisson noise if desired
        # taken from NeoGeographyToolkit/vipersci
        if noise:
            pred1, pred2 = np.random.default_rng().poisson(lam=(pred1, pred2))
        return pred1, pred2

    # create interpolation grid based on direction of model
    def setup(self):
        data_rshp = self.data.reshape((15,21,4))
        if self.nss_norm is not None:
            # need to normalize the nss data
            self.data[...,2:4] = self.nss_norm.norm(self.data[...,2:4])
        if self.depth_norm is not None:
            # need to normalize the depth and wt frac data
            self.data[...,0:2] = self.depth_norm.norm(self.data[...,0:2])
        # order of dataset: depth, wt %, detector 1 counts, detector 2 counts
        #self.det1 = SmoothBivariateSpline(self.data[...,1], self.data[...,0], self.data[...,2])
        #self.det2 = SmoothBivariateSpline(self.data[...,1], self.data[...,0], self.data[...,3])
        self.det1 = RectBivariateSpline(np.unique(self.data[...,1]), np.unique(self.data[...,0]), data_rshp[...,2].T)
        self.det2 = RectBivariateSpline(np.unique(self.data[...,1]), np.unique(self.data[...,0]), data_rshp[...,3].T)
        self.depth = SmoothBivariateSpline(self.data[...,2], self.data[...,3], self.data[...,1])
        self.wt_pct = SmoothBivariateSpline(self.data[...,2], self.data[...,3], self.data[...,0])
