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
import matplotlib.pyplot as plt

from multiprocessing import Pool
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Proj, CRS
from shapely.geometry import box
from rasterio.transform import from_origin, rowcol
from rasterio.windows import from_bounds, get_data_window, Window
from skimage.transform import rescale

from diffusion.diffusion import diffusion_cuda
from raytrace.raytrace import raytrace_horizon
from illumination.illumination import illuminate_cuda

from synthterrain.crater import functions, determine_production_function, random_points, to_file
from synthterrain.crater import generate_diameters, generate_ages

from lvsim.utils import LvSimCfg
from lvsim.crater import profile, stopar_fresh_dd

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

    # surface = args[0]
    # scale = args[1]
    # c = args[2]
    # r = args[3]
    # w_init = args[4]

    with Pool(8) as p:
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
        self.crater_df = pd.DataFrame(columns=['x','y','diameter','age','d/D','surface'])
        self.create_surface()
        self.size = self.surface.shape[0]

        # save production function information
        self.prod_fn = determine_production_function(self.crater_dist.a, self.crater_dist.b)

        # starting age of model
        self.t = cfg.args.max_age

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
        while self.t >= 0:

            # Take a step
            self.t -= self.cfg.args.time_delta
            i += 1

            # Print time update
            print("Now on " + str(self.t) + " Ga")

            # terrain changes (diffusion, production of new craters, removal of old craters)
            self.evolve_terrain()

            # compute illumination
            self.calc_horizons()
            self.illuminate()

            # ice delivery

            # ice movement

            # ice removal

            # save the results
            self.save()


    # code to generate surface based on data frame
    # based on code to return surfaces in diffuse_d_over_D_by_bin
    def create_surface(self):
        if len(self.crater_df) > 0:
            self.surface = make_heightmap(self.crater_df, np.zeros((math.ceil(self.window.height), math.ceil(self.window.width))), self.transform)
        else:
            self.surface = np.zeros((math.ceil(self.window.height), math.ceil(self.window.width)))

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

            new_df = pd.DataFrame()
            new_df["diameter"] = diameters
            new_df["age"] = np.random.default_rng().uniform(0, self.cfg.args.time_delta, size=len(diameters))

        else:
            diameters = generate_diameters(self.crater_dist, self.poly.area, self.cfg.args.d_lim[0], self.cfg.args.d_lim[1])
            new_df = generate_ages(diameters, self.prod_fn.csfd, self.crater_dist.csfd)

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
        start = time.time()
        new_ratios, new_surfs = diffusion_cuda(all_df["diameter"], all_df["d/D"], all_df["age"], surfs_np, D=self.cfg.args.domain_size)
        end = time.time()
        print("Diffusion runtime = %4.4f s" % (end-start))

        # Update d/D ratios, ages, remove rows with small enough d/D
        all_df["d/D"] = new_ratios
        all_df["surface"] = [s for s in new_surfs[:,...]]

        if len(old_df) > 0:
            all_df.loc[all_df.new == False, "age"] = self.crater_df["age"] + self.cfg.args.time_delta

        drop_inds = all_df[all_df["d/D"] < self.cfg.args.d_to_D_threshold].index
        all_df.drop(drop_inds, axis=0, inplace=True)
        self.crater_df = copy.copy(all_df)

        # Update stored surface
        self.create_surface()

    # Compute horizons for a given terrain map
    def calc_horizons(self):

        # add on border based on how far we're searching for horizon
        buffer = int(self.cfg.args.max_range * 1000 * self.cfg.args.res)
        s = int(2*buffer + self.size)
        surf = np.zeros((s,s))
        surf[buffer:-buffer,buffer:-buffer] = copy.copy(self.surface)
        
        # Loop through azimuths and compute horizon for all points on surface with CUDA raytracing code
        # TODO: figure out how to get CUDA to work with a version that computes horizons for all surface points and azimuths at once
        for i in tqdm(range(len(self.azims)), desc="Horizon calculations: "):
            a = np.array([self.azims[i]])
            elevs = raytrace_horizon(surf, a, res=self.cfg.args.res, max_range=self.cfg.args.max_range, min_elev=self.cfg.args.min_elev, elev_delta=self.cfg.args.elev_delta)
            elevs[np.abs(elevs-self.cfg.args.min_elev) < 0.0001] = np.nan # if too close to minimum elevation, return NaN
            self.elev_db[i,...] = copy.copy(elevs[...,0]) # copy results to elevation database

    # compute the illumination fraction for a given terrain model
    def illuminate(self):
        
        start = time.time()
        new_illumin_frac, new_psrs = illuminate_cuda(self.elev_db, self.eph, self.grid_ll, psr_threshold=cfg.args.psr_threshold)
        end = time.time()
        print("Illumination runtime = %4.4f s" % (end-start))
        self.illumin_frac = new_illumin_frac
        self.psr = new_psrs

    # save crater dataframe and plots
    def save(self):

        ts = str(int(self.t*1e3))

        # save crater dataframe without surfaces
        crater_df_small = self.crater_df.drop("surface", axis=1)
        to_file(crater_df_small, os.path.join(self.cfg.args.outpath, 'crater_list_'+ts+'.csv'), False)

        # save surface and illumination
        np.savez_compressed(os.path.join(self.cfg.args.outpath, 'maps_'+ts+'.npz'), surface=self.surface, illumin_frac=self.illumin_frac, psr=self.psr)

        # save plot of surface
        fig, ax = plt.subplots(1,3)
        
        im0 = ax[0].imshow(self.surface, cmap='terrain')
        im1 = ax[1].imshow(self.illumin_frac, cmap='inferno', vmin=0, vmax=1)
        im2 = ax[2].imshow(self.psr, cmap='gray', vmin=0, vmax=1)

        div0 = make_axes_locatable(ax[0])
        div1 = make_axes_locatable(ax[1])
        div2 = make_axes_locatable(ax[2])

        cax0 = div0.append_axes('right', size='5%', pad=0.05)
        cax1 = div1.append_axes('right', size='5%', pad=0.05)
        cax2 = div2.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(im0, cax=cax0, orientation='vertical')
        fig.colorbar(im1, cax=cax1, orientation='vertical')
        fig.colorbar(im2, cax=cax2, orientation='vertical')

        ax[0].set_title('Surface')
        ax[1].set_title('Illumination Fraction')
        ax[2].set_title('PSR Mask')
        
        if self.cfg.args.plot:
            plt.show()
        else:
            plt.savefig(os.path.join(self.cfg.args.outpath, 'plots', 'surface_'+ts+'.png'), dpi=100, bbox_inches='tight')
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

