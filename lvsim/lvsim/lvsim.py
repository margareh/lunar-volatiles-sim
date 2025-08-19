# date:     7-28-2025
# author:   margaret hansen
# purpose:  generate synthetic data of lunar volatiles distributions

import os
import math
import copy
import raytrace
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from synthterrain import crater
from synthterrain.crater import functions
from synthterrain.crater.diffusion import make_crater_field, diffuse_d_over_D_by_bin
# from moonpies import moonpies

from utils import LvSimCfg

logger = logging.getLogger(__name__)

# class for lunar volatiles sim
class LvSim():

    def __init__(self, cfg):

        self.cfg = cfg

        # set seed
        np.random.seed(cfg.args.seed)
        random.seed(cfg.args.seed)

        # set up initial list of craters
        self.poly = box(cfg.args.bbox[0], cfg.args.bbox[3], cfg.args.bbox[2], cfg.args.bbox[1])
        self.transform = from_origin(
            self.poly.bounds[0], self.poly.bounds[3], cfg.args.res, cfg.args.res
        )
        self.window = from_bounds(*self.poly.bounds, transform=self.transform)
        self.crater_dist = getattr(functions, cfg.args.csfd)(a=cfg.args.d_lim[0], b=cfg.args.d_lim[1])

        self.crater_df = crater.synthesize(
            self.crater_dist,
            polygon=self.poly,
            min_d=cfg.args.d_lim[0],
            max_d=cfg.args.d_lim[1],
            return_surfaces=True
        )

        # create initial surface based on crater list
        self.surface = self.create_surface()

        # save the list of craters without the individual surfaces
        self.crater_df.drop("surface", axis=1, inplace=True)
        crater.to_file(self.crater_df, os.path.join(cfg.args.outpath, 'crater_list.csv'), False)

        # save production function information
        self.prod_fn = crater.determine_production_function(self.crater_dist.a, self.crater_dist.b)


    # Use illumination maps and moonpies to generate ice distribution
    def gen_ice_dist(self, time):
        pass    

    # Produce sensor observations for a specific location
    def gen_sensor_obs(self, time):
        pass


    # Run through all steps
    def run_all(self):
        
        i = 0
        t = copy.copy(self.cfg.args.max_age)
        while t >= 0:

            # Print update every 10th time step
            if i % 10 == 0:
                logger.information("Now on time step " + str(t))

            # terrain changes (diffusion, production of new craters, removal of old craters)
            self.evolve_terrain()

            # ice delivery

            # ice movement

            # ice removal

            # save the results every 10th time step
            if i % 10 == 0:
                pass

            # increment counters and decrement time
            i += 1
            t -= self.cfg.args.time_delta

    # code to generate surface based on data frame
    # based on code to return surfaces in diffuse_d_over_D_by_bin
    def create_surface(self):
        self.surface = make_crater_field(
            self.crater_df, np.zeros((math.ceil(self.window.height), math.ceil(self.window.width))), self.transform
        )

        
    # Evolve the terrain from one time step to the next
    def evolve_terrain(self):

        # Update crater ages for stored craters
        self.crater_df["age"] += self.cfg.args.time_delta

        # Add new craters with production function
        # code for diameters taken from generate_diameters function from crater module of synthterrain
        # but modified to use production function instead of equilibrium distribution
        # prod_fn provides count per Ga, want this by time_delta so multiply by that
        df = pd.DataFrame()
        min_count = int(self.prod_fn.csfd(self.cfg.args.d_lim[1]) * self.poly.area) * self.cfg.args.time_delta
        max_count = int(self.prod_fn.csfd(self.cfg.args.d_lim[0]) * self.poly.area) * self.cfg.args.time_delta
        size = max_count - min_count
        diameters = []

        while len(diameters) != size:
            d = self.prod_fn.rvs(size=(size - len(diameters)))
            diameters += d[np.logical_and(min <= d, d <= max)].tolist()
            
        df["diameter"] = diameters
        df["age"] = np.random.default_rng().uniform(0, self.cfg.args.time_delta, size=len(diameters))
        xlist, ylist = crater.random_points(self.poly, len(df))
        df["x"] = xlist
        df["y"] = ylist

        # Apply diffusion model to craters
        df = diffuse_d_over_D_by_bin(
            df, start_dd_mean="Stopar step", return_surfaces=True
        )

        # Remove stored craters with d/D < threshold
        self.crater_df.drop("d/D" < self.cfg.args.d_to_D_threshold, axis=0, inplace=True)

        # Update stored surface


        pass



if __name__ == "__main__":

    # parse args
    cfg = LvSimCfg()

    # init sim
    logger.info("Initializing simulation")
    lvsim = LvSim(cfg)

    # run sim
    logger.info("Starting simulation run")
    lvsim.run_all()

