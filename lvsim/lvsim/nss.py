# date:     12-5-2025
# author:   margaret hansen
# purpose:  generate nss observations for ice depth and fraction

import pandas as pd
import numpy as np

from scipy.interpolate import SmoothBivariateSpline

# NSS sensor model class
# contains ability to do forward (get wt % and depth given NSS obs) and 
#   inverse (simulate NSS obs given wt % and depth) modeling
class NSS():

    def __init__(self, file):
        # load file with sensor data
        data_pd = pd.read_csv(file)
        self.data = data_pd.to_numpy() # wt pct, depth, det1, det2

        # create interpolation grid
        self.setup()

    # forward model (depth and wt % from observations)
    def forward(self, obs1, obs2):
        pred_d = self.depth(obs1, obs2)
        pred_wt = self.wt_pct(obs1, obs2)
        return pred_d, pred_wt

    # inverse model (simulated obs from depth and wt %)
    def inverse(self, wt_pct, depth, noise=False):
        pred1 = self.det1(depth, wt_pct)
        pred2 = self.det2(depth, wt_pct)
        # simulated poisson noise if desired
        # taken from NeoGeographyToolkit/vipersci
        if noise:
            pred1, pred2 = np.random.default_rng().poisson(lam=(pred1, pred2))
        return pred1, pred2

    # create interpolation grid based on direction of model
    def setup(self):
        # order of dataset: depth, wt %, detector 1 counts, detector 2 counts
        self.det1 = SmoothBivariateSpline(self.data[...,1], self.data[...,0], self.data[...,2])
        self.det2 = SmoothBivariateSpline(self.data[...,1], self.data[...,0], self.data[...,3])
        self.depth = SmoothBivariateSpline(self.data[...,2], self.data[...,3], self.data[...,1])
        self.wt_pct = SmoothBivariateSpline(self.data[...,2], self.data[...,3], self.data[...,0])
