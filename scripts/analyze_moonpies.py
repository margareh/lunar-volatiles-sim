# date:     11-25-2025
# author:   margaret hansen
# purpose:  make gif of moonpies output

import os
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='Path to data files to analyze')
    parser.add_argument('--mappath', type=str, help='Path to map files')
    args = parser.parse_args()

    # get directories to loop through
    dirs = [f for f in os.listdir(args.datapath) if f.find('png') < 0 and f.find('figs') < 0 and f.find('gif') < 0]
    # print(dirs)
    
    num_dirs = [int(f) for f in dirs]
    dirs_sort = [f for _,f in sorted(zip(num_dirs, dirs), reverse=True)]
    print(dirs_sort)
    num_dirs.sort(reverse=True)
    print(num_dirs)

    # make directory for storing the intermediate images
    img_path = os.path.join(args.datapath, 'figs')
    if os.path.exists(img_path) == False:
        os.mkdir(img_path)

    # data = np.load(os.path.join(args.datapath, dirs_sort[-1], 'data.npz'))
    # ice_col_grid = data['ice_col_grid']
    # ej_col_grid = data['ej_col_grid']
    # ice_depth = data['ice_depth']
    # ice_frac = data['ice_frac']

    # ice = np.sum(ice_col_grid, axis=0)
    # ej = np.sum(ej_col_grid, axis=0)

    # print(np.min(ice)) # 0
    # print(np.max(ice)) # 17.2

    # print(np.min(ej)) # 1.34
    # print(np.max(ej)) # 11.62

    # print(np.min(ice_frac)) # 0
    # print(np.max(ice_frac)) # 0.91

    # print(np.min(ice_depth)) # 0
    # print(np.max(ice_depth)) # 11.61

    # loop through the directories
    psrs = []
    ice = []
    depths = []
    fracs = []
    ext = [0, 200, 0, 200]
    for i in range(len(dirs_sort)):

        print(dirs_sort[i])
        
        # load the data
        data = np.load(os.path.join(args.datapath, dirs_sort[i], 'data.npz'))
        ice_depth = data['ice_depth']
        ice_frac = data['ice_frac']
        ice_col = data['ice_col_grid']
        ej_col = data['ej_col_grid']

        # psr mask for this dataset
        dir_myr = int(num_dirs[i] / 1e6)
        map_data = np.load(os.path.join(args.mappath, 'maps_'+str(dir_myr)+'.npz'))
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
    imageio.mimsave(os.path.join(args.datapath, 'psrs.gif'), psrs, duration = 1000, loop=0)
    imageio.mimsave(os.path.join(args.datapath, 'total_ice.gif'), ice, duration = 1000, loop=0)
    imageio.mimsave(os.path.join(args.datapath, 'ice_depth.gif'), depths, duration = 1000, loop=0)
    imageio.mimsave(os.path.join(args.datapath, 'ice_frac.gif'), fracs, duration = 1000, loop=0)


