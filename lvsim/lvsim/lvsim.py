# date:     7-28-2025
# author:   margaret hansen
# purpose:  generate synthetic data of lunar volatiles distributions

import os
import math
import raytrace
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from synthterrain import crater
from synthterrain.crater import functions
from synthterrain.crater.diffusion import make_crater_field
# from moonpies import moonpies

from utils import LvSimCfg

logger = logging.getLogger(__name__)

# class for lunar volatiles sim
class LvSim():

    def __init__(self, cfg):

        self.cfg = cfg

        # set seed
        np.random.seed(cfg.args.seed)
        

        # set up initial list of craters
        poly = box(cfg.args.bbox[0], cfg.args.bbox[3], cfg.args.bbox[2], cfg.args.bbox[1])

        crater_dist = getattr(functions, cfg.args.csfd)(a=cfg.args.d_lim[0], b=cfg.args.d_lim[1])

        self.crater_df = crater.synthesize(
            crater_dist,
            polygon=poly,
            # by_bin=False,
            min_d=cfg.args.d_lim[0],
            max_d=cfg.args.d_lim[1],
            return_surfaces=True,
        )

        # create initial surface based on crater list
        transform = from_origin(
            poly.bounds[0], poly.bounds[3], cfg.args.res, cfg.args.res
        )
        window = from_bounds(*poly.bounds, transform=transform)

        self.surface = make_crater_field(
            self.crater_df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform
        )

        # save the list of craters without the individual surfaces
        self.crater_df.drop("surface", axis=1, inplace=True)
        crater.to_file(self.crater_df, os.path.join(cfg.args.outpath, 'crater_list.csv'), False)

        # initialize list of times
        self.time_steps = np.flip(np.arange(0, cfg.args.max_age, cfg.args.time_delta))


    # Apply synthterrain diffusion model to current time step to get height map
    def gen_terrain(self, time):
        pass


    # Use illumination maps and moonpies to generate ice distribution
    def gen_ice_dist(self, time):
        pass    

    # Produce sensor observations for a specific location
    def gen_sensor_obs(self, time):
        pass


    # Run through all steps
    def run_all(self):
        pass



if __name__ == "__main__":

    # parse args
    cfg = LvSimCfg()

    # init sim
    lvsim = LvSim(cfg)

    # # run sim
    # lvsim.run_all()

