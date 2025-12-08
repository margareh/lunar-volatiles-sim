# date:     12-8-2025
# author:   margaret hansen
# purpose:  analyze results of sim

import os
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analyze_crater_list import analyze_crater_list


# function for analyzing moonpies data
def analyze_mp(datapath, mappath):

    # get directories to loop through
    dirs = [f for f in os.listdir(datapath) if f.find('png') < 0 and f.find('figs') < 0 and f.find('gif') < 0]
    # print(dirs)
    
    num_dirs = [int(f) for f in dirs]
    dirs_sort = [f for _,f in sorted(zip(num_dirs, dirs), reverse=True)]
    print(dirs_sort)
    num_dirs.sort(reverse=True)
    print(num_dirs)

    # make directory for storing the intermediate images
    img_path = os.path.join(datapath, 'figs')
    if os.path.exists(img_path) == False:
        os.mkdir(img_path)

    # loop through the directories
    psrs = []
    ice = []
    depths = []
    fracs = []
    ext = [0, 200, 0, 200]
    for i in range(len(dirs_sort)):

        print(dirs_sort[i])
        
        # load the data
        data = np.load(os.path.join(datapath, dirs_sort[i], 'data.npz'))
        ice_depth = data['ice_depth']
        ice_frac = data['ice_frac']
        ice_col = data['ice_col_grid']
        ej_col = data['ej_col_grid']

        # psr mask for this dataset
        dir_myr = int(num_dirs[i] / 1e6)
        map_data = np.load(os.path.join(mappath, 'maps_'+str(dir_myr)+'.npz'))
        psr = map_data['psr']

        # total ice and ejecta
        ice_tot = np.sum(ice_col, axis=0)
        ej_tot = np.sum(ej_col, axis=0)

        # PSR
        plt.imshow(psr, cmap='binary', extent=ext)
        plt.title(str(dir_myr) + ' Myr')
        plt.savefig(os.path.join(img_path, 'psr_tmp.png'), bbox_inches='tight', dpi=100)
        plt.close()

        new_psr = imageio.imread(os.path.join(img_path, 'psr_tmp.png'))
        psrs.append(new_psr)

        # total ice
        plt.imshow(ice_tot, cmap='Blues', vmin=0, vmax=17.5, extent=ext)
        plt.title(str(dir_myr) + ' Myr')
        plt.savefig(os.path.join(img_path, 'ice_tmp.png'), bbox_inches='tight', dpi=100)
        plt.close()

        new_ice = imageio.imread(os.path.join(img_path, 'ice_tmp.png'))
        ice.append(new_ice)

        # ice depth
        plt.imshow(ice_depth, cmap='Blues', vmin=0, vmax=11.7, extent=ext)
        plt.title(str(dir_myr) + ' Myr')
        plt.savefig(os.path.join(img_path, 'ice_depth_tmp.png'), bbox_inches='tight', dpi=100)
        plt.close()

        new_depth = imageio.imread(os.path.join(img_path, 'ice_depth_tmp.png'))
        depths.append(new_depth)

        # ice fraction
        plt.imshow(ice_frac, cmap='Oranges', vmin=0, vmax=1, extent=ext)
        plt.title(str(dir_myr) + ' Myr')
        plt.savefig(os.path.join(img_path, 'ice_frac_tmp.png'), bbox_inches='tight', dpi=100)
        plt.close()

        new_frac = imageio.imread(os.path.join(img_path, 'ice_frac_tmp.png'))
        fracs.append(new_frac)


    # make gifs
    imageio.mimsave(os.path.join(mappath, 'figs', 'psrs.gif'), psrs, duration = 1000, loop=0)
    imageio.mimsave(os.path.join(mappath, 'figs', 'total_ice.gif'), ice, duration = 1000, loop=0)
    imageio.mimsave(os.path.join(mappath, 'figs', 'ice_depth.gif'), depths, duration = 1000, loop=0)
    imageio.mimsave(os.path.join(mappath, 'figs', 'ice_frac.gif'), fracs, duration = 1000, loop=0)


# function for displaying nss observations
def display_nss(ice_depth, ice_wt_pct, obs1_map, obs2_map, datapath):
    
    # display last NSS observations
    fig, ax = plt.subplots(2,2,figsize=(20,20))

    im00 = ax[0,0].imshow(ice_depth, cmap='bone_r', vmin=0, vmax=2)
    im01 = ax[0,1].imshow(ice_wt_pct, cmap='Oranges', vmin=0, vmax=1)
    im10 = ax[1,0].imshow(obs1_map, cmap='PuRd_r', vmin=0, vmax=24)
    im11 = ax[1,1].imshow(obs2_map, cmap='BuGn_r', vmin=6, vmax=50)

    ax[0,0].set_title('Depth')
    ax[0,1].set_title('Wt %')
    ax[1,0].set_title('Det 1 Count')
    ax[1,1].set_title('Det 2 Count')

    div00 = make_axes_locatable(ax[0,0])
    div01 = make_axes_locatable(ax[0,1])
    div10 = make_axes_locatable(ax[1,0])
    div11 = make_axes_locatable(ax[1,1])

    cax00 = div00.append_axes('right', size='5%', pad=0.05)
    cax01 = div01.append_axes('right', size='5%', pad=0.05)
    cax10 = div10.append_axes('right', size='5%', pad=0.05)
    cax11 = div11.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(im00, cax=cax00, orientation='vertical')
    fig.colorbar(im01, cax=cax01, orientation='vertical')
    fig.colorbar(im10, cax=cax10, orientation='vertical')
    fig.colorbar(im11, cax=cax11, orientation='vertical')

    #plt.show()
    plt.savefig(os.path.join(datapath, 'figs', 'nss_obs.png'), dpi=100, bbox_inches='tight')
    plt.close()

# helper function for computing wt % from ice fraction
def get_wt_pct(ice_frac, ice_col_grid, args):
    ice_tot = np.sum(ice_col_grid, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ej_tot = (ice_tot * (1-ice_frac)) / ice_frac
    ice_wt = (ice_tot * (args.res**2)) * args.ice_density
    ej_wt = (ej_tot * (args.res**2)) * args.reg_density
    ice_wt_pct = ice_wt / (ice_wt + ej_wt)
    return ice_wt_pct

# main function
if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='Path to map data files to analyze')
    parser.add_argument('--mppath', type=str, help='Path to moonpies files')
    parser.add_argument('--dim', type=int, default=200, help='Side dimensions of map (number of pixels)')
    parser.add_argument('--res', type=float, default=1., help='Resolution of map (meters per pixel)')
    parser.add_argument('--plot', action='store_true', help='Flag to plot analysis figures instead of saving them')
    parser.add_argument('--age', type=float, nargs=2, default=[3.7, 0], help='Age of first and last non-flat terrain surface in Gyr')
    parser.add_argument('--haworth_dem', type=str, help='File path to Haworth DEM', default='/media/usb/ThesisWork/Volatiles/SouthPoleData/Haworth_DEM_1mpp/Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif')
    parser.add_argument('--ice_density', type=float, default=934., help='Ice density to use (kg / m^3)')
    parser.add_argument('--reg_density', type=float, default=1500., help='Regolith density to use (kg / m^3)')
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.datapath, 'figs')) == False:
        os.mkdir(os.path.join(args.datapath, 'figs'))

    first_step = str(int(args.age[0] * 1000))
    last_step= str(int(args.age[1] * 1000))

    # analyze crater list
    analyze_crater_list(args, first_step, last_step)

    # analyze moonpies output
    analyze_mp(args.mppath, args.datapath)

    # display final nss observations
    data = np.load(os.path.join(args.datapath, 'maps_'+last_step+'.npz'))
    mp_data = np.load(os.path.join(args.mppath, last_step, 'data.npz'))
    ice_wt_pct = get_wt_pct(mp_data['ice_frac'], mp_data['ice_col_grid'], args)
    display_nss(mp_data['ice_depth'], ice_wt_pct, data['det1'], data['det2'], args.datapath)

    # psr mask
    plt.imshow(data['psr'], cmap='binary')
    plt.savefig(os.path.join(args.datapath, 'figs', 'final_psr_mask.png'), bbox_inches='tight', dpi=100)
    plt.close()

    # surface age
    im = plt.imshow(data['age'] * 1e-6, cmap='plasma')
    plt.colorbar(im)
    plt.savefig(os.path.join(args.datapath, 'figs', 'final_surface_age_myr.png'), bbox_inches='tight', dpi=100)
    plt.close()

