# date:     4-24-2026
# author:   margaret hansen
# purpose:  simple example for horizon finding + illumination calcs

import os
import copy
import random
import rasterio as rs
import numpy as np
from tqdm import tqdm
from pyproj import Proj, CRS

from raytrace.raytrace import raytrace_horizon
from illumination.illumination import illuminate_cuda

from lvsim.utils import LvSimCfg
from lvsim.sim import load_ephemeris_data

from display_horizons import display_data, make_images, horizon_picker


# for transforming metric polar stereographic to lunar lat long coords
# WKT_STR = """PROJCS["PolarStereographic Moon",GEOGCS["D_Moon",DATUM["D_Moon",SPHEROID["Moon_polarRadius",1737400,0]],PRIMEM["Reference_Meridian",180],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]"""

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
    hw_dem_f = rs.open(os.path.join('../data/Haworth_DEM_1mpp/Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif'))
    hw_dem = hw_dem_f.read(1)
    crs = hw_dem_f.crs
    hw_dem_f.close()

    # downsample surface
    surface_lg = hw_dem[::10, ::10]
    surface = surface_lg[20:1186, :]

    cfg.args.bbox = [0, 1166, 1166, 0]
    cfg.args.res *= 10 # 10 m per pixel
    cfg.args.max_range = 0.2 # 200 m

    # grid with lat longs based on surface
    size = surface.shape[0]
    # print(size) # 1166
    x = np.arange(0, size * cfg.args.res, cfg.args.res)
    x -= float(size) * cfg.args.res / 2 # center the grid values on 0
    XX, YY = np.meshgrid(x, -x)
    grid = np.dstack((XX, YY)).reshape((size*size, 2))

    # crs = CRS.from_wkt(WKT_STR)
    proj = Proj(crs)
    lons, lats = proj(-grid[:,1], grid[:,0], inverse=True)
    lons += 180
    grid_ll = np.dstack((lats, lons)).reshape((size*size, 2))

    # buffer region for horizon generation
    b = int(cfg.args.max_range * 1000 / cfg.args.res) # border pixels
    s = int(2*b + size) # size of side
    # print(b) # 20
    # print(s) # 1206 - this should be smaller
    surf = np.zeros((s,s))
    surf[b:-b,b:-b] = copy.copy(surface)
    # print(surf.shape) # 1206 x 1206

    surf[b:-b, 0:b] = np.tile(surface[:,0], (b,1)).T # left
    surf[b:-b, -b:] = np.tile(surface[:,-1], (b,1)).T # right
    surf[0:b, b:-b] = np.tile(surface[0,:], (b,1)) # top
    surf[-b:, b:-b] = np.tile(surface[-1,:], (b,1)) # bottom

    # border corners
    surf[0:b, 0:b] = np.ones((b,b)) * surface[0,0] # top left
    surf[0:b, -b:] = np.ones((b,b)) * surface[0,-1] # top right
    surf[-b:, 0:b] = np.ones((b,b)) * surface[-1,0] # bottom left
    surf[-b:, -b:] = np.ones((b,b)) * surface[-1,-1] # bottom right

    # run horizon generation
    print("Making horizon database")
    azims = np.arange(0, 360, cfg.args.azim_res)
    elev_db = np.zeros((len(azims), size, size))
    grad_surf = np.array(np.gradient(surf))
    grad_max = np.max(np.sqrt(np.sum(pow(grad_surf, 2), axis=0)))
    min_elev = np.maximum(-np.rad2deg(np.arctan(grad_max))-1, -89) # really shouldn't have any larger but just in case
    print("Min elevation for horizon calcs: %4.2f" % (min_elev)) # just out of curiosity

    # Compute horizons
    elev_db = raytrace_horizon(surf, azims, res=cfg.args.res, max_range=cfg.args.max_range, min_elev=min_elev, elev_delta=cfg.args.elev_delta)

    # save horizon data
    # np.savez_compressed(os.path.join(cfg.args.outpath, 'horizon_illumin.npz'), elev_db=elev_db)
    # data = np.load(os.path.join(cfg.args.outpath, 'horizon_illumin.npz'))
    # elev_db = data['elev_db']

    # load ephemeris data
    eph_df = load_ephemeris_data(cfg.args.eph_file)
    eph = eph_df[['sun_sublat', 'sun_sublon', 'sun_range']].to_numpy() # lat long range

    # run illumination
    print("Illuminating map")
    illumin_frac, psr = illuminate_cuda(elev_db, eph, grid_ll, psr_threshold=cfg.args.psr_threshold)

    # save horizon db and illumination maps
    print("Saving map")
    np.savez_compressed(os.path.join(cfg.args.outpath, 'horizon_illumin.npz'), elev_db=elev_db, illumin_frac=illumin_frac, psr=psr)

    # show horizons
    print("Displaying horizons map")
    if MODE == 'display':
        display_data(elev_db, azim_res=cfg.args.azim_res)
    elif MODE == 'save':
        make_images(elev_db, cfg.args.outpath, azim_res=cfg.args.azim_res)
    elif MODE == 'interact':
        horizon_picker(elev_db, surface, cfg.args.res, cfg.args.max_range, azim_res=cfg.args.azim_res)
