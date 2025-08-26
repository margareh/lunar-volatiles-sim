# date:     8-22-2025
# author:   margaret hansen
# purpose:  compare illumination model using CUDA to version without it that we know works (ish)

import math
import random
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt

from pyproj import Proj, CRS
from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from raytrace.raytrace import raytrace_horizon
from illumination.illumination import illuminate_cuda

from lvsim.lvsim import load_ephemeris_data
from lvsim.utils import LvSimCfg
from lvsim.utils import LvSimCfg, latlon2enu, cartesian2spherical

from synthterrain import crater
from synthterrain.crater import functions
from synthterrain.crater.diffusion import make_crater_field

KM_AU = 149597870.700
WKT_STR = """PROJCS["PolarStereographic Moon",GEOGCS["D_Moon",DATUM["D_Moon",SPHEROID["Moon_polarRadius",1737400,0]],PRIMEM["Reference_Meridian",180],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]"""


def get_illumin(row, cfg, lats, lons, elev_db):

    # Solar parameters
    sun_rad_deg = cfg.args.solar_disc_angle / 2
    sun_rad_sq = sun_rad_deg ** 2
    sun_area_degsq = np.pi * sun_rad_sq

    # sunlat = row.sun_sublat.valuesq
    # sunlon =row.sun_sublon.values
    # sunrange = row.sun_range.values * KM_AU * 1000

    # Current position of sun relative to local frames
    # TODO: only compute this once at the beginning and store somewhere, then load as needed
    v_local = latlon2enu(row.sun_sublat, row.sun_sublon, row.sun_range * KM_AU * 1000, lats, lons, deg=True, esu=True)
    # v_local = latlon2enu(sunlat, sunlon, sunrange, lats, lons, deg=True, esu=True)
    _, elev, azim = cartesian2spherical(v_local[0,...], v_local[1,...], v_local[2,...], deg=True)
    elev = elev.reshape(elev_db[0,...].shape)
    azim = azim.reshape(elev_db[0,...].shape)

    # Horizon elevations for this azimuth
    azim_low = np.floor(azim).astype(int)
    azim_high = np.ceil(azim).astype(int)
    azim_low[azim_low > 359] -= 360
    azim_high[azim_high > 359] -= 360

    horizon_elev_low = np.squeeze(np.take_along_axis(elev_db, azim_low[None,...], axis=0))
    horizon_elev_high = np.squeeze(np.take_along_axis(elev_db, azim_high[None,...], axis=0))
    horizon_elev = (horizon_elev_low + horizon_elev_high) / 2
    # print(horizon_elev.shape)

    # Solar elevation for low and high points on solar disc
    sun_elev_low = elev - sun_rad_deg
    sun_elev_high = elev + sun_rad_deg

    # Area under chord across solar disc at average terrain elevation
    all_lit = (horizon_elev < sun_elev_low)
    all_dark = (horizon_elev > sun_elev_high)
    lower_disc = (sun_elev_low < horizon_elev) * (horizon_elev < elev)
    h = sun_elev_high - horizon_elev
    # print(lower_disc.shape)
    # print(h.shape)
    h[lower_disc] = horizon_elev[lower_disc] - sun_elev_low[lower_disc]
    h[all_dark] = 0
    h[all_lit] = 0

    # catching warnings here because they're annoying and I deal with them afterwards
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lit_area_degsq = sun_rad_sq * np.arccos(1 - (h / sun_rad_deg)) - (sun_rad_deg - h) * np.sqrt(sun_rad_sq - (sun_rad_deg - h)**2)

    lit_area_degsq[lower_disc] = sun_area_degsq - lit_area_degsq[lower_disc]
    lit_area_degsq[all_dark] = 0.0
    lit_area_degsq[all_lit] = sun_area_degsq
    lit_area_degsq /= sun_area_degsq # normalize by the total area of th solar disc

    # lit_area_degsq = elev
    return lit_area_degsq


# compute the illumination fraction for a given terrain model
def illuminate(eph_df, elev_db, grid_ll, cfg):
    
    # TODO: figure out how to get multiprocessing to work so we don't have to use pandas apply
    # Calculate illumination
    # sum_illumin = np.zeros_like(self.surface)
    # i = 722
    # illumin = get_illumin(eph_df.iloc[[i]], cfg, grid_ll[:,0], grid_ll[:,0], elev_db)
    illumin = eph_df.apply(lambda row: get_illumin(row, cfg, grid_ll[:,0], grid_ll[:,1], elev_db), axis=1)
    # with Pool(8) as p:
    #     results = [p.apply_async(get_illumin(row), [row]) for row in self.eph_df.itertuples(index=False, name=None)]
    #     for r in results:
    #         out = r.get()
    #         print(out.shape)
    
    # PSR mask
    # illumin_frac = copy.copy(illumin)
    illumin_frac = np.sum(illumin, axis=0) / len(eph_df)
    # self.illumin_frac = sum_illumin / len(self.eph_df)
    psr = (illumin_frac < cfg.args.psr_threshold)
    # print(self.illumin_frac.shape)
    # print(self.psr.shape)

    return illumin_frac, psr



if __name__ == "__main__":

    # create list of craters
    cfg = LvSimCfg()

    # set seed
    np.random.seed(cfg.args.seed)
    random.seed(cfg.args.seed)

    # set up initial list of craters
    poly = box(cfg.args.bbox[0], cfg.args.bbox[3], cfg.args.bbox[2], cfg.args.bbox[1])
    transform = from_origin(
        poly.bounds[0], poly.bounds[3], cfg.args.res, cfg.args.res
    )
    window = from_bounds(*poly.bounds, transform=transform)
    crater_dist = getattr(functions, cfg.args.csfd)(a=cfg.args.d_lim[0], b=cfg.args.d_lim[1])

    # diffusion with synthterrain to give us something interesting
    print("Initializing crater list")
    crater_df = crater.synthesize(
        crater_dist,
        polygon=poly,
        min_d=cfg.args.d_lim[0],
        max_d=cfg.args.d_lim[1],
        return_surfaces=True,
        by_bin=False
    )
    # print(self.crater_df)

    # surface from synthterrain crater list
    surface = make_crater_field(
        crater_df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform
    )

    # plt.imshow(surface)
    # plt.show()

    # grid with lat longs
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
    # print(grid_ll)

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(grid_ll[:,0].reshape((size, size)))
    # ax[1].imshow(grid_ll[:,1].reshape((size, size)))
    # plt.show()

    # since the surface starts out as flat, we can illuminate it easily by setting elevation to 0 deg for all azimuths
    azims = np.arange(0, 360, cfg.args.azim_res)

    # make horizon database
    # TODO: extrapolate based on nearest points instead of making flat surface
    buffer = int(cfg.args.max_range * 1000 * cfg.args.res)
    s = int(2*buffer + size)
    surf = np.zeros((s,s))
    surf[buffer:-buffer,buffer:-buffer] = copy.copy(surface)

    # plt.imshow(surface)
    # plt.show()
    
    # Loop through azimuths and compute horizon for all points on surface with CUDA raytracing code
    # TODO: figure out how to get CUDA to work with a version that computes horizons for all surface points and azimuths at once
    print("Making horizon database")
    elev_db = np.zeros((len(azims), size, size))
    for i in range(len(azims)):
        a = np.array([azims[i]])
        elevs = raytrace_horizon(surf, a, res=cfg.args.res, max_range=cfg.args.max_range, min_elev=cfg.args.min_elev, elev_delta=cfg.args.elev_delta)
        elevs[np.abs(elevs-cfg.args.min_elev) < 0.0001] = np.nan # if too close to minimum elevation, return NaN
        elev_db[i,...] = copy.copy(elevs[...,0]) # copy results to elevation database
    # print(elev_db.shape)

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(elev_db[0,...], cmap='Blues_r')
    # ax[1].imshow(elev_db[179,...], cmap='Blues_r')
    # plt.show()

    # load the ephemeris data
    eph_df = load_ephemeris_data(cfg.args.eph_file)
    eph = eph_df[['sun_sublat', 'sun_sublon', 'sun_range']].to_numpy() # lat long range

    # elevs = []
    # azims = []
    # for _, row in eph_df.iterrows():
        
    #     v_local = latlon2enu(row['sun_sublat'], row['sun_sublon'], row['sun_range'] * KM_AU * 1000, -89., 1., deg=True)
    #     _, elev, azim = cartesian2spherical(v_local[0,...], v_local[1,...], v_local[2,...], deg=True)
    #     elevs.append(elev)
    #     azims.append(azim)

    # elevs_np = np.array(elevs)
    # azims_np = np.array(azims)
    # t = np.arange(0, len(elevs))

    # print(np.min(elevs_np))
    # print(np.max(elevs_np))
    # print(np.min(azims_np))
    # print(np.max(azims_np))

    # fig, ax = plt.subplots(2,2)
    # ax[0,0].plot(t[:1000], elevs_np[:1000])
    # ax[0,0].set_title('Elevations, SP')
    # ax[1,0].plot(t[:1000], azims_np[:1000])
    # ax[1,0].set_title('Azimuths, SP')
    # ax[0,1].plot(t[:1000], eph_df['sun_sublat'].values[:1000])
    # ax[0,1].set_title('Subsolar Latitude')
    # ax[1,1].plot(t[:1000], eph_df['sun_sublon'].values[:1000])
    # ax[1,1].set_title('Subsolar Longitude')
    # plt.show()

    # illumination with old model
    print("Running old illumination code")
    illumin_frac_old, psr_old = illuminate(eph_df, elev_db, grid_ll, cfg)

    # illumination with cuda
    print("Running CUDA illumination code")
    illumin_frac, psr = illuminate_cuda(elev_db, eph, grid_ll, psr_threshold=cfg.args.psr_threshold)

    # compare the two
    fig, ax = plt.subplots(2,2)

    # original model
    ax[0,0].imshow(illumin_frac_old, cmap='inferno')
    ax[0,1].imshow(psr_old, vmin=0, vmax=1, cmap='gray_r')
    ax[0,0].set_title('Illumination, Original')
    ax[0,1].set_title('PSRs, Original')

    # new model
    ax[1,0].imshow(illumin_frac, cmap='inferno')
    ax[1,1].imshow(psr, vmin=0, vmax=1, cmap='gray_r')
    ax[1,0].set_title('Illumination, CUDA')
    ax[1,1].set_title('PSRs, CUDA')

    plt.show()
