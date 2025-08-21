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

from numpy.polynomial import Polynomial
from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from diffusion.diffusion import diffusion_cuda

from synthterrain.crater import functions, determine_production_function, random_points, to_file
from synthterrain.crater.diffusion import make_crater_field

from utils import LvSimCfg


# crater profile copied from FTmod_Crater class in synthterrain
# used here to create initial profiles when a crater is first added
def profile(dd, diameter, D=200):
    """Returns a numpy array of elevation values based in the input numpy
    array of radius fraction values, such that a radius fraction value
    of 1 is at the rim, less than that interior to the crater, etc.

    A ValueError will be thrown if any values in r are < 0.
    """

    x = np.linspace(-2, 2, D)  # spans a space 2x the diameter.
    xx, yy = np.meshgrid(x, x, sparse=True)  # square domain
    r = np.sqrt(xx**2 + yy**2)

    if not isinstance(r, np.ndarray):
        r = np.ndarray(r)

    out_arr = np.zeros_like(r)

    if np.any(r < 0):
        raise ValueError("The radius fraction value can't be less than zero.")

    inner_idx = np.logical_and(0 <= r, r <= 0.98)
    rim_idx = np.logical_and(0.98 < r, r <= 1.02)
    outer_idx = np.logical_and(1.02 < r, r <= 1.5)

    inner_poly = Polynomial([-0.228809953, 0.227533882, 0.083116795, -0.039499407])
    outer_poly = Polynomial([0.188253307, -0.187050452, 0.01844746, 0.01505647])

    rim_hoverd = 0.036822095

    out_arr[inner_idx] = inner_poly(r[inner_idx])
    out_arr[rim_idx] = rim_hoverd
    out_arr[outer_idx] = outer_poly(r[outer_idx])

    floor = rim_hoverd - (dd)
    out_arr[out_arr < floor] = floor

    return out_arr * diameter

# Stopar depth/diameter ratio for fresh craters
# Copied from synthterrain and modified to be usable with arrays of diameters
def stopar_fresh_dd(diameter):
    """
    Returns a depth/Diameter ratio based on the set of graduated d/D
    categories in Stopar et al. (2017), defined down to 40 m.  This
    function also adds two extrapolated categories.
    """
    # The last two elements are extrapolated
    d_lower_bounds = (0, 10, 40, 100, 200, 400)
    dds = (0.10, 0.11, 0.13, 0.15, 0.17, 0.21)

    dd_list = np.ones_like(diameter) * np.nan
    for d, dd in zip(d_lower_bounds, dds):
        dd_list[diameter >= d] = dd
    return dd_list


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

        # self.crater_df = crater.synthesize(
        #     self.crater_dist,
        #     polygon=self.poly,
        #     min_d=cfg.args.d_lim[0],
        #     max_d=cfg.args.d_lim[1],
        #     return_surfaces=True,
        #     start_dd_std=cfg.args.start_dd_std,
        # )
        # print(self.crater_df["surface"][0].shape) # D x D

        # crater datafame and surface are initially empty and flat
        self.crater_df = pd.DataFrame(columns=['x','y','diameter','age','d/D','surface'])
        self.create_surface()

        # save production function information
        self.prod_fn = determine_production_function(self.crater_dist.a, self.crater_dist.b)

        # save starting age of model
        self.t = cfg.args.max_age


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
        if len(self.crater_df) > 0:
            self.surface = make_crater_field(
                self.crater_df, np.zeros((math.ceil(self.window.height), math.ceil(self.window.width))), self.transform
            )
        else:
            self.surface = np.zeros((math.ceil(self.window.height), math.ceil(self.window.width)))

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
        xlist, ylist = random_points(self.poly, len(new_df))
        new_df["x"] = xlist
        new_df["y"] = ylist
        new_df["new"] = True
        new_df["d/D"] = stopar_fresh_dd(np.array(diameters))
        new_df["surface"] = new_df.apply(lambda row: profile(row["d/D"], row["diameter"], D=self.cfg.args.domain_size), axis=1)

        # Set up dataframe for old craters
        old_df = copy.copy(self.crater_df)
        if len(old_df) > 0:
            old_df["age"] = self.cfg.args.time_delta # only want to diffuse since last diffusion model (AKA over length of time step)
            old_df["new"] = False
        else:
            old_df["age"] = None
            old_df["new"] = None

        # Append
        if len(old_df) > 0:
            all_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            all_df = copy.copy(new_df)

        # Apply diffusion model to craters
        surfs_np = np.array(all_df["surface"].tolist())
        new_ratios, new_surfs = diffusion_cuda(all_df["diameter"], all_df["d/D"], all_df["age"], surfs_np, D=self.cfg.args.domain_size)
        
        # Update d/D ratios, ages, remove rows with small enough d/D
        all_df["d/D"] = new_ratios
        all_df["surface"] = [s for s in new_surfs[:,...]]

        if len(old_df) > 0:
            all_df["new" == False]["age"] = self.crater_df["age"] + self.cfg.args.time_delta

        drop_inds = all_df[all_df["d/D"] < self.cfg.args.d_to_D_threshold].index
        all_df.drop(drop_inds, axis=0, inplace=True)
        self.crater_df = copy.copy(all_df)

        # Update stored surface
        self.create_surface()


    # save crater dataframe and plots
    def save(self):

        time = str(int(self.t*1e3))

        # save crater dataframe without surfaces
        self.crater_df.drop("surface", axis=1, inplace=True)
        to_file(self.crater_df, os.path.join(self.cfg.args.outpath, 'crater_list_'+time+'.csv'), False)

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

