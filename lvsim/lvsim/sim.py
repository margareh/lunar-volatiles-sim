# date:     7-28-2025
# author:   margaret hansen
# purpose:  generate synthetic data of lunar volatiles distributions

import os
import math
import copy
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # to avoid nonsense tkinter errors that cause crashes
import matplotlib.pyplot as plt

from multiprocessing import Pool
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Proj, CRS
from shapely.geometry import box
from rasterio.transform import from_origin, rowcol
from rasterio.windows import from_bounds, get_data_window, Window
from skimage.transform import rescale

from diffusion.diffusion import diffusion_cuda
from raytrace.raytrace import raytrace_horizon
from illumination.illumination import illuminate_cuda

from synthterrain.crater import functions, determine_production_function, random_points
from synthterrain.crater import generate_diameters
from synthterrain.crater.age import equilibrium_age

from lvsim.utils import LvSimCfg
from lvsim.crater import profile, stopar_fresh_dd, in_crater

# some global defines
# conversion from KM to AU (necessary for ephemeris data)
KM_AU = 149597870.700

# WKT string for converting between grid definition (lunar polar stereographic) to lat long
# copied from Haworth DEM file
WKT_STR = """PROJCS["PolarStereographic Moon",GEOGCS["D_Moon",DATUM["D_Moon",SPHEROID["Moon_polarRadius",1737400,0]],PRIMEM["Reference_Meridian",180],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]"""

# Load the ephemeris data from JPL Horizons
def load_ephemeris_data(file):
    cols = ['date', 'bl1' ,'bl2', 'obs_sublon', 'obs_sublat', 'sun_sublon', 'sun_sublat', 'sun_range', 'sun_rdot']
    dtypes = {'date' : str,
              'bl1' : str,
              'bl2' : str,
              'obs_sublon' : np.float64, 
              'obs_sublat' : np.float64,
              'sun_sublon' : np.float64,
              'sun_sublat' : np.float64,
              'sun_range' : np.float64,
              'sun_rdot' : np.float64}
    eph_df = pd.read_csv(file, index_col=False, names=cols, dtype=dtypes, parse_dates=['date'], sep=',')
    eph_df = eph_df[['date', 'sun_sublon', 'sun_sublat', 'sun_range']]
    return eph_df


def rescale_surface(args):

    # parse args
    surface = args[0]
    scale = args[1]
    c = args[2]
    r = args[3]
    w_init = args[4]

    # rescale the surface
    surf_rescaled = rescale(surface, scale, preserve_range=True, anti_aliasing=True)

    # compute bounds
    h, w = surf_rescaled.shape
    col_off = c - int(w / 2)
    row_off = r - int(h / 2)

    # compare to bounds of terrain model and adjust if don't fully intersect
    w_tm = Window(0, 0, w_init.width, w_init.height)
    w_surf = Window(col_off - w_init.col_off, row_off - w_init.row_off, w, h)
    w_tm2 = Window(w_init.col_off - col_off, w_init.row_off - row_off, w_init.width, w_init.height)
    w_surf2 = Window(0, 0, w, h)

    tm_intersect = w_tm.intersection(w_surf)
    surf_intersect = w_surf2.intersection(w_tm2)
    tm_slices = tm_intersect.toslices()
    surf_slices = surf_intersect.toslices()

    return (surf_rescaled[surf_slices], tm_slices)


def remove_old_craters(args):
    i = args[0] # index for current crater
    data_np = args[1] # crater dataset for comparison (age check applied)
    append = args[2] # dataset we need to append to also search this (no age check applied)
    return_age = args[3] # flag for whether to return index of newer crater(s)
    curr_crater = data_np[i,:]
    newer_craters = data_np[data_np[:,1] < curr_crater[1],:]
    if append is not None:
        newer_craters = np.vstack((newer_craters, append))
    rad_diff = (newer_craters[:,2] - curr_crater[2]) / 2
    xpos_diff_sq = pow((curr_crater[3] - newer_craters[:,3]), 2)
    ypos_diff_sq = pow((curr_crater[4] - newer_craters[:,4]), 2)
    inside_rad = (xpos_diff_sq + ypos_diff_sq <= np.sign(rad_diff)*pow(rad_diff, 2))
    
    if np.any(inside_rad):
        out_val = curr_crater[0]
        if return_age:
            max_age_new = np.max(newer_craters[inside_rad, 1]) # "oldest" crater
            age = curr_crater[1] + max_age_new
            return (out_val, age)
        else:
            return out_val
    else:
        out_val = -1.0
        if return_age:
            age = -1.0
            return (out_val, age)
        else:
            return out_val
    

# Make a crater heightmap based on initial surface and dataframe of craters
def make_heightmap(df, init_surface, tf):

    # transform things to numpy
    diams = df["diameter"].to_numpy()
    # print(diams) # 7293
    surf_list = df["surface"].values
    surfaces = np.stack(surf_list, axis=-1)
    w_init = get_data_window(init_surface)
    # print(surfaces.shape) # N x N x D

    # compute resolution of each row
    surf_gsd = 2 * diams / surfaces.shape[0] # 2 * diameter / surface shape

    # transform row and column values of center point of crater
    r, c = rowcol(tf, df['x'].values, df['y'].values)

    # process all surfaces in the model
    new_surf = copy.copy(init_surface)

    with Pool() as p:
        args = [(surfaces[...,i], surf_gsd[i] / tf.a, c[i], r[i], w_init) for i in range(len(diams))]
        out = p.map(rescale_surface, args) # this produces a list of results
    
    for i in range(len(diams)):
        new_surf[out[i][1]] += out[i][0]

    return new_surf


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

        # crater datafame and surface are initially empty and flat
        self.crater_df = pd.DataFrame(columns=['x','y','diameter','age','d/D','surface','new'])
        self.init_surface = np.zeros((math.ceil(self.window.height), math.ceil(self.window.width)))
        self.create_surface()
        self.surface_age = np.zeros_like(self.surface)
        self.size = self.surface.shape[0]

        # save production function information
        self.prod_fn = determine_production_function(self.crater_dist.a, self.crater_dist.b)

        # starting age of model and crater index
        self.t = cfg.args.max_age
        self.i = 0

        # save lat and long information based on assumed grid
        # build a grid
        x = np.arange(0, self.size * cfg.args.res, cfg.args.res)
        x -= float(self.size) * cfg.args.res / 2 # center the grid values on 0
        XX, YY = np.meshgrid(x, -x)
        grid = np.dstack((XX, YY)).reshape((self.size*self.size, 2))

        # convert to lat long so we can compare to the ephemeris data
        # which projection is best? polar stereographic (ups) used for other data of south pole, gnomonic (gnom) would preserve geodesics as straight lines (do we care about this?)
        crs = CRS.from_wkt(WKT_STR)
        proj = Proj(crs)
        lons, lats = proj(-grid[:,1], grid[:,0], inverse=True)
        lons += 180
        self.grid_ll = np.dstack((lats, lons)).reshape((self.size*self.size, 2))

        # since the surface starts out as flat, we can illuminate it easily by setting elevation to 0 deg for all azimuths
        self.azims = np.arange(0, 360, cfg.args.azim_res)
        self.elev_db = np.tile(np.zeros_like(self.surface), (len(self.azims), 1, 1))
        # print(self.elev_db.shape)
        eph_df = load_ephemeris_data(cfg.args.eph_file)
        self.eph = eph_df[['sun_sublat', 'sun_sublon', 'sun_range']].to_numpy() # lat long range
        print("\tIlluminating model...")
        self.illuminate()

        # save the list of craters, current surface, and illumination model
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
        start_all = time.time()
        while self.t-self.cfg.args.time_delta >= 0:

            # Take a step
            self.t -= self.cfg.args.time_delta
            i += 1

            # Print time update
            print("Now on " + str(self.t) + " Ga")

            # terrain changes (diffusion, production of new craters, removal of old craters)
            self.evolve_terrain()
            if self.cfg.args.d_to_D_threshold > 0:
                self.update_surface_age()

            # compute illumination
            self.calc_horizons()
            self.illuminate()

            # ice delivery

            # ice movement

            # ice removal

            # save the results
            self.save()

        end_all = time.time()
        print("Overall runtime (minutes): %4.4f" % ((end_all - start_all) / 60))


    # code to generate surface based on data frame
    # based on code to return surfaces in diffuse_d_over_D_by_bin
    def create_surface(self):
        if len(self.crater_df) > 0:
            # add the craters in the dataframe to the initial surface
            # initial surface = flat + whatever craters have stopped "evolving"
            # i.e. their d/D ratio is too low to degrade them anymore and they 
            # are no longer in our dataset
            self.surface = make_heightmap(self.crater_df, self.init_surface, self.transform)
        else:
            # initialize to flat surface
            self.surface = copy.copy(self.init_surface)

    # Evolve the terrain from one time step to the next
    def evolve_terrain(self):

        # Add new craters with production function
        # code for diameters taken from generate_diameters function from crater module of synthterrain
        # but modified to use production function instead of equilibrium distribution
        # prod_fn provides count per Ga, want this by time_delta so multiply by that
        if self.cfg.args.use_prod_fn:
            min_count = int(self.prod_fn.csfd(self.cfg.args.d_lim[1]) * self.poly.area * self.cfg.args.time_delta)
            max_count = int(self.prod_fn.csfd(self.cfg.args.d_lim[0]) * self.poly.area * self.cfg.args.time_delta)
            size = max_count - min_count
            diameters = []

            while len(diameters) != size:
                d = self.prod_fn.rvs(size=(size - len(diameters)))
                diameters += d[np.logical_and(self.cfg.args.d_lim[0] <= d, d <= self.cfg.args.d_lim[1])].tolist()

            ages = np.random.default_rng().uniform(0, self.cfg.args.time_delta * 1e9, size=len(diameters))

        else:
            diameters = generate_diameters(self.crater_dist, self.poly.area, self.cfg.args.d_lim[0], self.cfg.args.d_lim[1])
            # new_df = generate_ages(diameters, self.prod_fn.csfd, self.crater_dist.csfd)
            eq_ages = equilibrium_age(diameters, self.prod_fn.csfd, self.crater_dist.csfd)
            age_ranges = np.minimum(eq_ages, self.cfg.args.time_delta * 1e9)
            ages = np.random.default_rng().uniform(0, age_ranges)

        new_df = pd.DataFrame()
        new_df["diameter"] = diameters
        new_df["age"] = ages

        xlist, ylist = random_points(self.poly, len(new_df))
        new_df["x"] = xlist
        new_df["y"] = ylist

        # Remove any "new" craters that are smaller and within a newer crater
        print("Number of new craters, pre-filtering: %d" % (len(new_df)))
        new_craters_np = np.array([new_df.index, new_df.age.values, new_df.diameter.values, new_df.x.values, new_df.y.values]).T
        with Pool() as p:
            args = [(i, new_craters_np, None, False) for i in range(new_craters_np.shape[0])]
            drop_inds_all = p.map(remove_old_craters, args) # this produces a list of results

        drop_inds = [i for i in drop_inds_all if i >= 0]        
        new_df.drop(drop_inds, axis=0, inplace=True)
        print("Number of new craters, post-filtering: %d" % (len(new_df)))

        new_df["new"] = True
        new_df["d/D"] = stopar_fresh_dd(np.array(new_df["diameter"].values))
        new_df["surface"] = new_df.apply(lambda row: profile(row["d/D"], row["diameter"], D=self.cfg.args.domain_size), axis=1)

        new_df.index = np.arange(self.i, self.i+len(new_df))
        self.i += len(new_df)

        # Remove old craters within newer craters
        old_df = copy.copy(self.crater_df)
        if len(self.crater_df) > 0:
            print("Number of old craters, pre-filtering: %d" % (len(old_df)))
            old_craters_np = np.array([old_df.index, old_df.age.values, old_df.diameter.values, old_df.x.values, old_df.y.values]).T
            with Pool() as p:
                args = [(i, old_craters_np, new_craters_np, False) for i in range(old_craters_np.shape[0])]
                drop_inds_all = p.map(remove_old_craters, args) # this produces a list of results

            drop_inds = [i for i in drop_inds_all if i >= 0]
            old_df.drop(drop_inds, axis=0, inplace=True)
            print("Number of old craters, post-filtering: %d" % (len(old_df)))

            # Set up old dataframe for diffusion
            prev_ages = old_df["age"].values
            old_df["age"] = self.cfg.args.time_delta * 1e9 # only want to diffuse since last diffusion model (AKA over length of time step)
            old_df["new"] = False
            all_df = pd.concat([old_df, new_df])

        else:
            all_df = copy.copy(new_df)

        # Apply diffusion model to craters
        surfs_np = np.array(all_df["surface"].tolist())
        start = time.time()
        new_ratios, new_surfs = diffusion_cuda(all_df["diameter"].values, all_df["d/D"].values, all_df["age"].values, surfs_np, D=self.cfg.args.domain_size)
        end = time.time()
        print("Diffusion runtime = %4.4f s" % (end-start))

        # Update d/D ratios, ages, remove rows with small enough d/D
        all_df["d/D"] = new_ratios
        all_df["surface"] = [s for s in new_surfs[:,...]]

        if len(old_df) > 0:
            all_df.loc[all_df.new == False, "age"] = prev_ages + self.cfg.args.time_delta * 1e9

        print("Number of craters before depth-to-diam filtering: %d" % (len(all_df)))
        drop_df = all_df[all_df["d/D"] < self.cfg.args.d_to_D_threshold]

        if len(drop_df) > 0:
            # add craters that are being removed from the list to our initial surface
            # so we keep them in the heightmap in the future
            self.init_surface = make_heightmap(drop_df, self.init_surface, self.transform)

            # drop the craters from the dataframe
            all_df.drop(drop_df.index, axis=0, inplace=True)
        
        self.crater_df = copy.copy(all_df)
        print("Current number of craters: %d" % (len(self.crater_df)))

        # Update stored surface with craters in the dataset
        self.create_surface()

    # Compute age based on last crater
    def update_surface_age(self):
        
        # get flags for whether points are inside a crater for all rows of dataframe
        flag = self.crater_df.apply(lambda row: in_crater(row["x"], row["y"], row["diameter"], self.surface.shape[0], self.cfg.args.res, self.transform), axis=1)
        flags_np = np.transpose(np.array(flag.tolist()), (1, 2, 0))
        ages_np = flags_np * self.crater_df.age.values
        ages_np[~flags_np] = np.nan

        # compute ages based on last crater to hit that spot
        # or add in the time delta for this time step
        crater_cond = np.max(flags_np, axis=-1)
        self.surface_age[crater_cond] = np.nanmin(ages_np[crater_cond,:], axis=-1)
        self.surface_age[~crater_cond] += self.cfg.args.time_delta


    # Compute horizons for a given terrain map
    def calc_horizons(self):

        # Add on border based on how far we're searching for horizon
        # currently setting border values to closest real value
        b = int(self.cfg.args.max_range * 1000 * self.cfg.args.res)
        s = int(2*b + self.size)
        surf = np.zeros((s,s))

        # interior portion
        surf[b:-b, b:-b] = copy.copy(self.surface)

        # border sides
        surf[b:-b, 0:b] = np.tile(self.surface[:,0], (b,1)).T # left
        surf[b:-b, -b:] = np.tile(self.surface[:,-1], (b,1)).T # right
        surf[0:b, b:-b] = np.tile(self.surface[0,:], (b,1)) # top
        surf[-b:, b:-b] = np.tile(self.surface[-1,:], (b,1)) # bottom

        # border corners
        surf[0:b, 0:b] = np.ones((b,b)) * self.surface[0,0] # top left
        surf[0:b, -b:] = np.ones((b,b)) * self.surface[0,-1] # top right
        surf[-b:, 0:b] = np.ones((b,b)) * self.surface[-1,0] # bottom left
        surf[-b:, -b:] = np.ones((b,b)) * self.surface[-1,-1] # bottom right
        
        # Compute max slope to define starting elevation
        grad_surf = np.array(np.gradient(surf))
        grad_max = np.max(np.sqrt(np.sum(pow(grad_surf, 2), axis=0)))
        min_elev = np.maximum(-np.rad2deg(np.arctan(grad_max))-1, -89) # really shouldn't have any larger but just in case
        print("Min elevation for horizon calcs: %4.2f" % (min_elev)) # just out of curiosity

        # Compute horizons
        self.elev_db = raytrace_horizon(surf, self.azims, res=self.cfg.args.res, max_range=self.cfg.args.max_range, min_elev=min_elev, elev_delta=self.cfg.args.elev_delta)


    # compute the illumination fraction for a given terrain model
    def illuminate(self):
        
        start = time.time()
        new_illumin_frac, new_psrs = illuminate_cuda(self.elev_db, self.eph, self.grid_ll, psr_threshold=self.cfg.args.psr_threshold)
        end = time.time()
        print("Illumination runtime = %4.4f s" % (end-start))
        self.illumin_frac = new_illumin_frac
        self.psr = new_psrs

    # save crater dataframe and plots
    def save(self):

        ts = str(int(self.t*1e3))

        # save crater dataframe without surfaces
        crater_df_small = self.crater_df.drop("surface", axis=1)
        # to_file(crater_df_small, os.path.join(self.cfg.args.outpath, 'crater_list_'+ts+'.csv'), False)
        crater_df_small.to_csv(
            os.path.join(self.cfg.args.outpath, 'crater_list_'+ts+'.csv'),
            columns=["x", "y", "diameter", "age", "d/D"],
        )

        # save surface and illumination
        np.savez_compressed(os.path.join(self.cfg.args.outpath, 'maps_'+ts+'.npz'), surface=self.surface, illumin_frac=self.illumin_frac, psr=self.psr, age=self.surface_age)

        # plots
        fig, ax = plt.subplots(2,3)
        fig.set_size_inches(30,20)
        
        # first row (surface, illumination fraction, PSR mask)
        im0 = ax[0][0].imshow(self.surface, cmap='terrain', vmin=-100, vmax=10)
        im1 = ax[0][1].imshow(self.illumin_frac, cmap='inferno', vmin=0, vmax=1)
        im2 = ax[0][2].imshow(self.psr, cmap='gray_r', vmin=0, vmax=1)

        div0 = make_axes_locatable(ax[0][0])
        div1 = make_axes_locatable(ax[0][1])
        div2 = make_axes_locatable(ax[0][2])

        cax0 = div0.append_axes('right', size='5%', pad=0.05)
        cax1 = div1.append_axes('right', size='5%', pad=0.05)
        cax2 = div2.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(im0, cax=cax0, orientation='vertical')
        fig.colorbar(im1, cax=cax1, orientation='vertical')
        fig.colorbar(im2, cax=cax2, orientation='vertical')

        ax[0][0].set_title('Surface')
        ax[0][1].set_title('Illumination Fraction')
        ax[0][2].set_title('PSR Mask')

        ax[0][0].set_axis_off()
        ax[0][1].set_axis_off()
        ax[0][2].set_axis_off()

        # second row (min, avg, max elevation of the horizon)
        min_elev = np.min(self.elev_db, axis=0)
        mean_elev = np.mean(self.elev_db, axis=0)
        max_elev = np.max(self.elev_db, axis=0)
        im3 = ax[1][0].imshow(min_elev, cmap='Blues', vmin=-20, vmax=30)
        im4 = ax[1][1].imshow(mean_elev, cmap='Blues', vmin=-20, vmax=30)
        im5 = ax[1][2].imshow(max_elev, cmap='Blues', vmin=-20, vmax=30)

        div3 = make_axes_locatable(ax[1][0])
        div4 = make_axes_locatable(ax[1][1])
        div5 = make_axes_locatable(ax[1][2])

        cax3 = div3.append_axes('right', size='5%', pad=0.05)
        cax4 = div4.append_axes('right', size='5%', pad=0.05)
        cax5 = div5.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(im3, cax=cax3, orientation='vertical')
        fig.colorbar(im4, cax=cax4, orientation='vertical')
        fig.colorbar(im5, cax=cax5, orientation='vertical')

        ax[1][0].set_title('Min Horizon Elevation')
        ax[1][1].set_title('Avg Horizon Elevation')
        ax[1][2].set_title('Max Horizon Elevation')

        ax[1][0].set_axis_off()
        ax[1][1].set_axis_off()
        ax[1][2].set_axis_off()

        if self.cfg.args.plot:
            plt.show()
        else:
            plt.savefig(os.path.join(self.cfg.args.outpath, 'plots', 'surface_'+ts+'.png'), dpi=100, bbox_inches='tight')
            plt.close('all')


if __name__ == "__main__":

    # parse args
    cfg = LvSimCfg()

    # init sim
    print("Initializing simulation")
    lvsim = LvSim(cfg)

    # run sim
    print("Starting simulation run")
    lvsim.run_all()

