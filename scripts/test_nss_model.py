# date:     12-5-2025
# author:   margaret hansen
# purpose:  test nss measurement creation from depth and fraction from moonpies
#   note: default ice and regolith densities are those used in moonpies and are 
#   cited as being from Cannon (2020)

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from lvsim.nss import NSS


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', type=str, help='File path to data with depth and ice fraction to use')
    parser.add_argument('--nss_file', type=str, default='../data/leaf_LP.csv', help='File path to file with NSS calibration data')
    parser.add_argument('--map_res', type=float, default=1., help='Map resolution in meters per pixel')
    parser.add_argument('--ice_density', type=float, default=934., help='Ice density to use (kg / m^3)')
    parser.add_argument('--reg_density', type=float, default=1500., help='Regolith density to use (kg / m^3)')
    args = parser.parse_args()

    # load data on ice depth and fraction
    map_data = np.load(os.path.join(args.map_path, 'data.npz'))
    ice_depth = map_data['ice_depth']
    ice_frac = map_data['ice_frac']
    ice_tot = np.sum(map_data['ice_col_grid'], axis=0)
    ej_tot = np.sum(map_data['ej_col_grid'], axis=0)
    n,m = ice_depth.shape

    # convert fraction to wt %
    ice_wt = (ice_tot * (args.map_res**2)) * args.ice_density
    ej_wt = (ej_tot * (args.map_res**2)) * args.reg_density    
    ice_wt_pct = ice_wt / (ice_wt + ej_wt)

    # reshape inputs
    depth = ice_depth.reshape((n*m))
    wt_pct = ice_wt_pct.reshape((n*m))

    # generate NSS sensor observations
    nss = NSS(args.nss_file)
    obs1, obs2 = nss.inverse(wt_pct, depth)
    obs1_map = obs1.reshape((n,m))
    obs2_map = obs2.reshape((n,m))

    # display NSS observations
    fig, ax = plt.subplots(2,2,figsize=(20,20))

    ax[0,0].imshow(ice_depth, cmap='Blues')
    ax[0,1].imshow(ice_wt_pct, cmap='Oranges')
    ax[1,0].imshow(obs1_map, cmap='PuRd_r')
    ax[1,1].imshow(obs2_map, cmap='BuGn_r')

    ax[0,0].set_title('Depth')
    ax[0,1].set_title('Wt %')
    ax[1,0].set_title('Det 1 Count')
    ax[1,1].set_title('Det 2 Count')

    #plt.show()
    plt.savefig(os.path.join(args.map_path, 'nss_obs.png'), dpi=100, bbox_inches='tight')
