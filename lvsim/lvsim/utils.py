# date:     7-29-2025
# author:   margaret hansen
# purpose:  input arguments class for lunar volatiles sim

import os
import argparse
import torch

class LvSimCfg():

    def __init__(self):

        # setup argument parser
        parser = argparse.ArgumentParser()

        # paths
        parser.add_argument('--outpath',
                            type=str,
                            default='../output',
                            help='Location for storing output')
        
        # args for lvsim
        parser.add_argument('--max_age',
                            type=float,
                            default=4.25,
                            help='Maximum age in gigayears (AKA start time of model)')
        parser.add_argument('--time_delta',
                            type=float,
                            default=0.001,
                            help='Time delta in gigayears')
        parser.add_argument('--d_to_D_threshold',
                            type=float,
                            default=0.04,
                            help='Depth to diameter ratio threshold for removing old craters')

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
                            default=[1, 1000],
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
                            default='../data/JPLHorizons/sun_position_2004_2024_values_only.txt',
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
        
        # parse args at the end and save
        self.args = parser.parse_args()

        # validate inputs
        self.validate_args()


    # check that args are correct
    def validate_args(self):

        if self.args.csfd not in ['Trask', 'VIPER_Env_Spec']:
            print("Invalid crater SFD provided, defaulting to VIPER_Env_Spec")
            self.args.csfd = 'VIPER_Env_Spec'

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
        theta = torch.deg2rad(theta)
        phi = torch.deg2rad(phi)
    x = r * torch.cos(theta) * torch.cos(phi)
    y = r * torch.cos(theta) * torch.sin(phi)
    z = r * torch.sin(theta)
    return x, y, z


# conversion from spherical coordinates to cartesian
# x, y, z = cartesian coordinates
# theta is elevation angle
def cartesian2spherical(x, y, z, deg=False):

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
        y = torch.tensor(y)
        z = torch.tensor(z)

    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arcsin(z / r)

    # define azimuth to be 0 if x and y are zero
    # extra pi added here to make azimuth match horizon database azimuth values
    azim_0 = (torch.abs(x) < 1e-15) * (torch.abs(y) < 1e-15)
    # print(azim_0)
    phi = torch.zeros_like(x)
    phi[~azim_0] = torch.sign(y[~azim_0]) * torch.arccos(x[~azim_0] / torch.sqrt(x[~azim_0]**2 + y[~azim_0]**2)) + torch.pi

    if deg:
        # return angles in degrees
        theta = torch.rad2deg(theta)
        phi = torch.rad2deg(phi)

    return r, theta, phi


# convert from latitude and longitude to local ENU frame
# lat, lon are the subsolar points
# range is the range of the sun
# lat0, lon0 are the local grid point(s)
def latlon2enu(lat, lon, range, lat0, lon0, deg=False, esu=False):

    if not isinstance(lat, torch.Tensor):
        lat = torch.tensor(lat)
        lon = torch.tensor(lon)
        range = torch.tensor(range)
        lat0 = torch.tensor(lat0)
        lon0 = torch.tensor(lon0)

    # if deg = true, convert inputs to radians
    if deg:
        lat = torch.deg2rad(lat)
        lon = torch.deg2rad(lon)
        lat0 = torch.deg2rad(lat0)
        lon0 = torch.deg2rad(lon0)

    # position of sun in cartesian coordinates centered on moon
    # elevation = latitude
    # azimuth = longitude
    x_sun, y_sun, z_sun = spherical2cartesian(range, lat, lon)

    # local grid points in cartesian coordinates centered on moon
    x_l, y_l, z_l = spherical2cartesian(1737400, lat0, lon0)

    # convert to local coordinates for each of the points in the grid 
    cos_th = torch.cos(lat0)
    sin_th = torch.sin(lat0)
    cos_phi = torch.cos(lon0)
    sin_phi = torch.sin(lon0)

    # # below definition is based on uvw2enu from https://github.com/geospace-code/pymap3d/blob/main/src/pymap3d/ecef.py#L365
    # n = len(lat0)
    # R = np.tile(np.eye(3).reshape((3,3,1)), (1,1,n))
    # R[0, 0, :] = -sin_phi
    # R[0, 1, :] = cos_phi
    # R[1, 0, :] = -sin_th * cos_phi
    # R[1, 1, :] = -sin_th * sin_phi
    # R[1, 2, :] = cos_th
    # R[2, 0, :] = cos_th * cos_phi
    # R[2, 1, :] = cos_th * sin_phi
    # R[2, 2, :] = sin_th

    # # transform point for the sun in cartesian coordinates from global to local ENU surface frame for each point in grid
    # m = len(lat)
    # local_t = np.array([x_l, y_l, z_l]).reshape((3,n))
    # p = np.transpose(np.dstack((x_moon, y_moon, z_moon)), (0,2,1))
    # a = np.tile(local_t.reshape((3,n,1)), (1,1,m))
    # b = np.tile(p.reshape((3,1,m)), (1,n,1))
    # diff = b - a
    # R_all = np.tile(R.reshape((3,3,n,1)), (1,1,1,m))
    # v_local = np.einsum('ijkl,jkl->ikl', R_all, diff)

    # expand dims --> first dim is sun position (ephemeris index), second dim is local coord (grid index)
    dx = torch.unsqueeze(x_sun,1) - torch.unsqueeze(x_l,0)
    dy = torch.unsqueeze(y_sun,1) - torch.unsqueeze(y_l,0)
    dz = torch.unsqueeze(z_sun,1) - torch.unsqueeze(z_l,0)

    t = cos_phi * dx + sin_phi * dy
    xs = -sin_phi * dx + cos_phi * dy
    ys = -sin_th * t + cos_th * dz
    zs = cos_th * t + sin_th * dz

    # free up some memory
    del(t)
    del(dx)
    del(dy)
    del(dz)
    del(cos_th)
    del(sin_th)
    del(cos_phi)
    del(sin_phi)

    v_local = torch.dstack((xs, ys, zs)).transpose(0,-1)

    # additional transform from ENU to ESU (left-handed frame with x east, y down) if desired
    if esu:
        v_local[1,...] *= -1

    return torch.squeeze(v_local)
