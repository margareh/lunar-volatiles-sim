# date:     11-19-2025
# author:   margaret hansen
# purpose:  compute surface age based on last crater

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import box
from rasterio.transform import from_origin, rowcol
from rasterio.windows import from_bounds

# from lvsim.crater import in_crater

# Flag whether surface points are inside or outside of a crater
def in_crater(x, y, diam, dim, res, tf):

    # grid points for surface in meters
    xx, yy = np.meshgrid(np.arange(0, dim*res, res), np.arange(0, dim*res, res))

    # transform center of crater and grid points to row/column
    r_center, c_center = rowcol(tf, x, y)
    r_center /= res
    c_center /= res

    # compute whether or not each point is inside the crater
    r2 = np.power((yy - r_center), 2) + np.power((xx - c_center), 2)
    in_flag = (r2 <= np.power(diam / 2, 2))

    return in_flag


# main program
if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='Path to dataset to compute ages for')
    parser.add_argument('--bbox', type=float, default=[0, 200, 200, 0], help='Bounding box in meters for map')
    parser.add_argument('--res', type=float, default=1., help='Resolution of map in meters per pixel')
    parser.add_argument('--time_delta', type=float, default=0.1, help='Time delta in Gyr')
    parser.add_argument('--plot', action='store_true', help='Flag for showing plots of output')
    args = parser.parse_args()

    # convert time delta to years
    args.time_delta *= 1e9

    # compute dimension based on bbox and res
    dim = int(args.bbox[1] / args.res)

    # get files
    files = os.listdir(args.datapath)
    map_files = [f for f in files if f.find('npz') > 0]
    crater_files = [f for f in files if f.find('.csv') > 0]
    time_steps = [int(re.sub('\D', '', f)) for f in map_files]
    time_steps_c = [int(re.sub('\D', '', f)) for f in crater_files]
    # print(time_steps)
    map_files_sort = [f for _, f in sorted(zip(time_steps, map_files), reverse=True)]
    crater_files_sort = [f for _, f in sorted(zip(time_steps_c, crater_files), reverse=True)]
    T = len(map_files_sort)

    # print(crater_files_sort[0:10])
    # print(map_files_sort[0:10])
    # crater_df = pd.read_csv(os.path.join(args.datapath, crater_files_sort[1]), header=0, index_col=0)
    # data = np.load(os.path.join(args.datapath, map_files_sort[1]))
    # surf = data["surface"]

    poly = box(args.bbox[0], args.bbox[3], args.bbox[2], args.bbox[1])
    tf = from_origin(poly.bounds[0], poly.bounds[3], args.res, args.res)
    window = from_bounds(*poly.bounds, transform=tf)
    
    # crater_df_sort = crater_df.sort_values(by='diameter', ascending=False)
    # for i in range(len(crater_df)):
    #     in_flag = in_crater(crater_df_sort.x.values[i], crater_df_sort.y.values[i], crater_df_sort.diameter.values[i], surf.shape[0], args.res, tf)

    #     print(crater_df_sort.x.values[i])
    #     print(crater_df_sort.y.values[i])
        
    #     fig, ax = plt.subplots(1, 3, figsize=(30,10))
    #     ax[0].imshow(surf)
    #     ax[1].imshow(in_flag)
    #     ax[2].imshow(surf)
    #     ax[2].imshow(in_flag, cmap='gray', alpha=0.5)
    #     plt.show()

    # loop through files and compute age at each step
    surf_age = np.zeros((dim, dim))
    for t in range(T):
        
        print(t)

        # load the crater dataframe data for this time step
        crater_df = pd.read_csv(os.path.join(args.datapath, crater_files_sort[t]),
                                header=0,
                                index_col=0)
        nc = len(crater_df)

        if nc > 0:

            # get flags for whether points are inside a crater for all rows of dataframe
            flag = crater_df.apply(lambda row: in_crater(row["x"], row["y"], row["diameter"], dim, args.res, tf), axis=1)
            flags_np = np.transpose(np.array(flag.tolist()), (1, 2, 0))
            ages_np = flags_np * crater_df.age.values
            ages_np[~flags_np] = np.nan

            # compute ages based on last crater to hit that spot
            # or add in the time delta for this time step
            crater_cond = np.max(flags_np, axis=-1)
            surf_age[crater_cond] = np.nanmin(ages_np[crater_cond,:], axis=-1)
            surf_age[~crater_cond] += args.time_delta
            
            # # plot  things for each surface
            # data = np.load(os.path.join(args.datapath, map_files_sort[t]))
            # surf = data["surface"]
            
            # fig, ax = plt.subplots(1,2, figsize=(20,10))
            # ax[0].imshow(surf)
            # ax[1].imshow(surf_age)
            # plt.show()

    # plot the results at the end
    data = np.load(os.path.join(args.datapath, map_files_sort[-1]))
    surf = data["surface"]
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].imshow(surf)
    ax[1].imshow(surf_age * 1e-6)
    plt.show()


