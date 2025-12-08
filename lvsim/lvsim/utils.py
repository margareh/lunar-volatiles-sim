# date:     7-29-2025
# author:   margaret hansen
# purpose:  input arguments class for lunar volatiles sim

import os
import argparse
import numpy as np
import sys
import copy

class LvSimCfg():

    def __init__(self, addtl_args=None):

        # setup argument parser
        parser = argparse.ArgumentParser()

        # paths
        parser.add_argument('--outpath',
                            type=str,
                            default='/home/margareh/lunar-volatiles-sim/output',
                            help='Location for storing output')
        
        # args for lvsim
        parser.add_argument('--max_age',
                            type=float,
                            default=3.8,
                            help='Maximum age in gigayears (AKA start time of model)')
        parser.add_argument('--time_delta',
                            type=float,
                            default=0.01,
                            help='Time delta in gigayears')
        parser.add_argument('--d_to_D_threshold',
                            type=float,
                            default=0.04,
                            help='Depth to diameter ratio threshold for removing old craters')
        parser.add_argument('--use_prod_fn',
                            action='store_true',
                            help='Flag for using the production function directly to generate craters instead of the equilibrium SFD')
        parser.add_argument('--iters',
                            type=int,
                            default=100,
                            help='Iterations for Monte Carlo simulations')

        # args for synthterrain
        parser.add_argument('--bbox',
                            type=float,
                            nargs=4,
                            default=[0, 1000, 1000, 0],
                            help='Bounding box in m, ordered as min-x, may-y, max-x, min-y')
        parser.add_argument('--res',
                            type=float,
                            default=1.,
                            help='Resolution of terrain model to generate in meters')
        parser.add_argument('--d_lim',
                            type=float,
                            nargs=2,
                            default=[2, 1000],
                            help='Minimum and maximum allowed crater diameters in m')
        parser.add_argument('--csfd',
                            type=str,
                            default='VIPER_Env_Spec',
                            help='Name of crater size-frequency distribution to use (only supports Trask or VIPER_Env_Spec)')
        parser.add_argument('--start_dd_std',
                            type=float,
                            default=0.02,
                            help='Standard deviation for crater bin definition when using bins to apply diffusion model')
        parser.add_argument('--domain_size',
                            type=int,
                            default=100,
                            help='Domain size for generating crater models when performing diffusion')

        # args for horizon and illumination models
        parser.add_argument('--azim_res',
                            type=float,
                            default=1,
                            help='Resolution for azimuths for computing horizon, in degrees')
        parser.add_argument('--eph_file',
                            type=str,
                            default='/home/margareh/lunar-volatiles-sim/data/JPLHorizons/sun_position_2004_2024_values_only.txt',
                            help='File location for ephemeris data from JPL Horizons')
        parser.add_argument('--psr_threshold',
                            type=float,
                            default=0.0001,
                            help='Threshold percentage below which pixels are classified as PSR')
        parser.add_argument('--solar_disc_angle',
                            type=float,
                            default=0.53,
                            help='Diameter angle of solar disc from lunar surface')
        parser.add_argument('--min_elev',
                            type=float,
                            default=-89,
                            help='Minimum elevation angle for horizon computation in degrees')
        parser.add_argument('--max_range',
                            type=float,
                            default=0.2,
                            help='Maximum range for searching for horizon in km')
        parser.add_argument('--elev_delta',
                            type=float,
                            default=0.25,
                            help='Elevation step size in degrees for horizon search')

        # behavior flags
        parser.add_argument('--plot',
                            action='store_true',
                            help='Flag for plotting results')
        
        # other args
        parser.add_argument('--seed',
                            type=int,
                            default=12957973,
                            help='Random seed')
        
        # args for moonpies
        parser.add_argument('--mp_cfg',
                            type=str,
                            default='../../moonpies/moonpies/configs/lvsim_config.py',
                            help='Config file to use for moonpies')
        
        # args for nss
        parser.add_argument('--nss_file',
                            type=str,
                            default='../../data/NSS/leaf_LP.csv',
                            help='File path to file with NSS calibration data')
        parser.add_argument('--ice_density',
                            type=float,
                            default=934.,
                            help='Ice density to use (kg / m^3)')
        parser.add_argument('--reg_density',
                            type=float,
                            default=1500.,
                            help='Regolith density to use (kg / m^3)')
        
        # parse args at the end and save
        # this also adds any additional arguments that were added
        args_to_parse = copy.copy(sys.argv[1:])
        if addtl_args is not None:
            for k, v in addtl_args.items():
                args_to_parse.append('--'+k)
                args_to_parse.append(v)

        self.args = parser.parse_args(args_to_parse)

        # validate inputs
        self.validate_args()


    # check that args are correct
    def validate_args(self):

        # check the crater equilibrium SFD
        if self.args.csfd not in ['Trask', 'VIPER_Env_Spec']:
            print("Invalid crater SFD provided, defaulting to VIPER_Env_Spec")
            self.args.csfd = 'VIPER_Env_Spec'

        # make the outpath directories if necessary
        if os.path.exists(self.args.outpath) == False:
            os.mkdir(self.args.outpath)

        # add seed to output file path and make directory
        self.args.outpath = os.path.join(self.args.outpath, str(self.args.seed))

        if os.path.exists(self.args.outpath) == False:
            os.mkdir(self.args.outpath)
        
        if os.path.exists(os.path.join(self.args.outpath, 'plots')) == False:
            os.mkdir(os.path.join(self.args.outpath, 'plots'))



# conversion from spherical coordinates to cartesian
# r = range
# theta = elevation
# phi = azimuth
def spherical2cartesian(r, theta, phi, deg=False):
    if deg:
        # transform angles to radians
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


# conversion from spherical coordinates to cartesian
# x, y, z = cartesian coordinates
# theta is elevation angle
def cartesian2spherical(x, y, z, deg=False):

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arcsin(z / r)

    # define azimuth to be 0 if x and y are zero
    # extra pi added here to make azimuth match horizon database azimuth values
    azim_0 = (np.abs(x) < 1e-15) * (np.abs(y) < 1e-15)
    # print(azim_0)
    phi = np.zeros_like(x)
    phi[~azim_0] = np.sign(y[~azim_0]) * np.arccos(x[~azim_0] / np.sqrt(x[~azim_0]**2 + y[~azim_0]**2)) + np.pi

    if deg:
        # return angles in degrees
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)

    return r, theta, phi


# convert from latitude and longitude to local ENU frame
# lat, lon are the subsolar points
# range is the range of the sun
# lat0, lon0 are the local grid point(s)
def latlon2enu(lat, lon, range, lat0, lon0, deg=False, esu=False):

    # make sure all inputs are arrays
    if not isinstance(lat, np.ndarray):
        lat = np.array([lat])
    if not isinstance(lon, np.ndarray):
        lon = np.array([lon])
    if not isinstance(lat0, np.ndarray):
        lat0 = np.array([lat0])
    if not isinstance(lon0, np.ndarray):
        lon0 = np.array([lon0])

    # if deg = true, convert inputs to radians
    if deg:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        lat0 = np.deg2rad(lat0)
        lon0 = np.deg2rad(lon0)

    # position of sun in cartesian coordinates centered on moon
    # elevation = latitude
    # azimuth = longitude
    x_moon, y_moon, z_moon = spherical2cartesian(range, lat, lon)

    # local grid points in cartesian coordinates centered on moon
    x_l, y_l, z_l = spherical2cartesian(1737400, lat0, lon0)

    # convert to local coordinates for each of the points in the grid 
    cos_th = np.cos(lat0)
    sin_th = np.sin(lat0)
    cos_phi = np.cos(lon0)
    sin_phi = np.sin(lon0)

    # below definition is based on uvw2enu from https://github.com/geospace-code/pymap3d/blob/main/src/pymap3d/ecef.py#L365
    n = len(lat0)
    R = np.tile(np.eye(3).reshape((3,3,1)), (1,1,n))
    R[0, 0, :] = -sin_phi
    R[0, 1, :] = cos_phi
    R[1, 0, :] = -sin_th * cos_phi
    R[1, 1, :] = -sin_th * sin_phi
    R[1, 2, :] = cos_th
    R[2, 0, :] = cos_th * cos_phi
    R[2, 1, :] = cos_th * sin_phi
    R[2, 2, :] = sin_th

    # transform point for the sun in cartesian coordinates from global to local ENU surface frame for each point in grid
    m = len(lat)
    local_t = np.array([x_l, y_l, z_l]).reshape((3,n))
    p = np.transpose(np.dstack((x_moon, y_moon, z_moon)), (0,2,1))
    a = np.tile(local_t.reshape((3,n,1)), (1,1,m))
    b = np.tile(p.reshape((3,1,m)), (1,n,1))
    diff = b - a
    R_all = np.tile(R.reshape((3,3,n,1)), (1,1,1,m))
    v_local = np.einsum('ijkl,jkl->ikl', R_all, diff)

    # additional transform from ENU to ESU (left-handed frame with x east, y down) if desired
    if esu:
        v_local[1,...] *= -1

    return np.squeeze(v_local)
