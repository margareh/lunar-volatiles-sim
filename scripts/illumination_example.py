# date:     4-24-2026
# author:   margaret hansen
# purpose:  simple example for horizon finding + illumination calcs

import os
import copy
import random
import numpy as np
from tqdm import tqdm
from pyproj import Proj, CRS

from raytrace.raytrace import raytrace_horizon
from illumination.illumination import illuminate_cuda

from lvsim.utils import LvSimCfg
from lvsim.sim import load_ephemeris_data

from display_horizons import display_data, make_images, horizon_picker

# for transforming metric polar stereographic to lunar lat long coords
WKT_STR = """PROJCS["PolarStereographic Moon",GEOGCS["D_Moon",DATUM["D_Moon",SPHEROID["Moon_polarRadius",1737400,0]],PRIMEM["Reference_Meridian",180],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]"""

# horizon display mode (options: display, save, interact)
# display: show each azimuth's elevation map in sequence
# save: save a gif of above
# interact: show horizon elevation profile across all azimuths based on picking points in the map
MODE = 'interact'

if __name__ == "__main__":

    # define some things
    cfg = LvSimCfg()

    # set seed
    np.random.seed(cfg.args.seed)
    random.seed(cfg.args.seed)
    
    # load in a surface heightmap in npy format
    # TODO: replace with surface you want to run horizon / illumination modeling on
    surface = np.load(os.path.join(cfg.args.outpath, 'surface.npy'))

    # grid with lat longs based on surface
    size = surface.shape[0]
    x = np.arange(0, size * cfg.args.res, cfg.args.res)
    x -= float(size) * cfg.args.res / 2 # center the grid values on 0
    XX, YY = np.meshgrid(x, -x)
    grid = np.dstack((XX, YY)).reshape((size*size, 2))

    crs = CRS.from_wkt(WKT_STR)
    proj = Proj(crs)
    lons, lats = proj(-grid[:,1], grid[:,0], inverse=True)
    lons += 180
    grid_ll = np.dstack((lats, lons)).reshape((size*size, 2))

    # buffer region for horizon generation
    buffer = int(cfg.args.max_range * 1000 * cfg.args.res)
    s = int(2*buffer + size)
    surf = np.zeros((s,s))
    surf[buffer:-buffer,buffer:-buffer] = copy.copy(surface)

    # run horizon generation
    print("Making horizon database")
    azims = np.arange(0, 360, cfg.args.azim_res)
    elev_db = np.zeros((len(azims), size, size))
    for i in tqdm(range(len(azims)), desc='Horizon raytracing: '):
        a = np.array([azims[i]])
        elevs = raytrace_horizon(surf, a, res=cfg.args.res, max_range=cfg.args.max_range, min_elev=cfg.args.min_elev, elev_delta=cfg.args.elev_delta)
        elevs[np.abs(elevs-cfg.args.min_elev) < 0.0001] = np.nan # if too close to minimum elevation, return NaN
        elev_db[i,...] = copy.copy(elevs[...,0]) # copy results to elevation database, A x N X N

    eph_df = load_ephemeris_data(cfg.args.eph_file)
    eph = eph_df[['sun_sublat', 'sun_sublon', 'sun_range']].to_numpy() # lat long range

    # run illumination
    illumin_frac, psr = illuminate_cuda(elev_db, eph, grid_ll, psr_threshold=cfg.args.psr_threshold)

    # save horizon db and illumination maps
    np.savez_compressed(os.path.join(cfg.args.outpath, 'horizon_illumin.npz'), elev_db=elev_db, illumin_frac=illumin_frac, psr=psr)

    # show horizons
    if MODE == 'display':
        display_data(elev_db, azim_res=cfg.args.azim_res)
    elif MODE == 'save':
        make_images(elev_db, cfg.args.outpath, azim_res=cfg.args.azim_res)
    elif MODE == 'interact':
        horizon_picker(elev_db, surface, cfg.args.res, cfg.args.max_range, azim_res=cfg.args.azim_res)
