# date:     7-28-2025
# author:   margaret hansen
# purpose:  generate synthetic data of lunar volatiles distributions

import os
import math
import copy
import raytrace
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from synthterrain import crater
from synthterrain.crater import functions
from synthterrain.crater.profile import stopar_fresh_dd
from synthterrain.crater.diffusion import make_crater_field, diffuse_d_over_D_by_bin
# from moonpies import moonpies

from utils import LvSimCfg

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
            return_surfaces=True,
            start_dd_std=cfg.args.start_dd_std,
        )
        # print(self.crater_df)

        # create initial surface based on crater list
        self.create_surface()

        # save production function information
        self.prod_fn = crater.determine_production_function(self.crater_dist.a, self.crater_dist.b)

        # save initial data
        # note that this function also deletes the surface column from the dataframe
        self.t = cfg.args.max_age
        self.save()


    # Use illumination maps and moonpies to generate ice distribution
    def gen_ice_dist(self, time):
        pass    

    # Produce sensor observations for a specific location
    def gen_sensor_obs(self, time):
        pass


    # Run through all steps
    def run_all(self):
        
        i = 0
        while self.t >= 0:

            # Take a step
            self.t -= self.cfg.args.time_delta
            i += 1

            # Print update every 10th time step
            if i % 10 == 0:
                print("Now on " + str(self.t) + " Ga")

            # terrain changes (diffusion, production of new craters, removal of old craters)
            self.evolve_terrain()

            # ice delivery

            # ice movement

            # ice removal

            # save the results every 10th time step
            # if i % 10 == 0:
            self.save()


    # code to generate surface based on data frame
    # based on code to return surfaces in diffuse_d_over_D_by_bin
    def create_surface(self):
        self.surface = make_crater_field(
            self.crater_df, np.zeros((math.ceil(self.window.height), math.ceil(self.window.width))), self.transform
        )

        
    # Evolve the terrain from one time step to the next
    def evolve_terrain(self):

        # Add new craters with production function
        # code for diameters taken from generate_diameters function from crater module of synthterrain
        # but modified to use production function instead of equilibrium distribution
        # prod_fn provides count per Ga, want this by time_delta so multiply by that
        new_df = pd.DataFrame()
        min_count = int(self.prod_fn.csfd(self.cfg.args.d_lim[1]) * self.poly.area * self.cfg.args.time_delta)
        max_count = int(self.prod_fn.csfd(self.cfg.args.d_lim[0]) * self.poly.area * self.cfg.args.time_delta)
        size = max_count - min_count
        diameters = []
        print(size)

        while len(diameters) != size:
            d = self.prod_fn.rvs(size=(size - len(diameters)))
            diameters += d[np.logical_and(self.cfg.args.d_lim[0] <= d, d <= self.cfg.args.d_lim[1])].tolist()
            
        new_df["diameter"] = diameters
        new_df["age"] = np.random.default_rng().uniform(0, self.cfg.args.time_delta, size=len(diameters))
        xlist, ylist = crater.random_points(self.poly, len(new_df))
        new_df["x"] = xlist
        new_df["y"] = ylist
        new_df["new"] = True
        new_df["start_dd_mean"] = stopar_fresh_dd(np.array(diameters))
        new_df["start_dd_std"] = self.cfg.args.start_dd_std

        # Set up dataframe for old craters
        old_df = copy.copy(self.crater_df)
        old_df["age"] = self.cfg.args.time_delta # only want to diffuse since last diffusion model (AKA over length of time step)
        old_df["new"] = False
        old_df["start_dd_mean"] = old_df["d/D"]
        old_df["start_dd_std"] = self.cfg.args.start_dd_std

        # Append
        all_df = pd.concat([old_df, new_df], ignore_index=True)

        # Apply diffusion model to craters
        self.crater_df = diffuse_d_over_D_by_bin(
            all_df, start_dd_mean="rows", return_surfaces=True
        )

        # Update crater ages for stored craters
        self.crater_df["new" == False]["age"] += self.cfg.args.time_delta

        # Remove stored craters with d/D < threshold
        self.crater_df.drop("d/D" < self.cfg.args.d_to_D_threshold, axis=0, inplace=True)

        # Update stored surface
        self.create_surface()


    # save crater dataframe and plots
    def save(self):

        time = str(int(self.t*1e3))

        # save crater dataframe without surfaces
        self.crater_df.drop("surface", axis=1, inplace=True)
        crater.to_file(self.crater_df, os.path.join(self.cfg.args.outpath, 'crater_list_'+time+'.csv'), False)

        # save plot of surface
        fig, ax = plt.subplots()
        ax.imshow(self.surface, cmap='terrain')
        if self.cfg.args.plot:
            plt.show()
        else:
            plt.savefig(os.path.join(self.cfg.args.outpath, 'plots', 'surface_'+time+'.png'), dpi=100, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":

    # parse args
    cfg = LvSimCfg()

    # init sim
    print("Initializing simulation")
    lvsim = LvSim(cfg)

    # run sim
    print("Starting simulation run")
    lvsim.run_all()

