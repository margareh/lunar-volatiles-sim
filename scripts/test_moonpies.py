# date:     11-20-2025
# author:   margaret hansen
# purpose:  run moonpies on crater dataset

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool
from shapely.geometry import box
from rasterio.transform import from_origin

from lvsim.crater import in_crater

from moonpies.moonpies import MoonPIES
from moonpies import config as mp_config


# update crater and psr mask data
def update_data(crater_file, psr_file, start_age, end_age, args):

    # load crater dataset and psr mask
    crater_df = pd.read_csv(os.path.join(args.datapath, crater_file), header=0, index_col=0)
    maps = np.load(os.path.join(args.datapath, psr_file))
    # surf = maps['surface']
    # illumin = maps['illumin_frac']
    psr = maps['psr']

    # transform for crater
    poly = box(args.bbox[0], args.bbox[3], args.bbox[2], args.bbox[1])
    tf = from_origin(poly.bounds[0], poly.bounds[3], args.res, args.res)

    # filter to new craters only
    print("Before filtering to new craters only: %d" % (len(crater_df)))
    # print(len(crater_df))
    age_diff = (start_age - end_age)
    crater_df = crater_df[crater_df["age"] <= age_diff]
    print("After filtering to new craters only: %d" % (len(crater_df)))

    # increase crater age based on length of sim
    crater_df['age'] += end_age

    # add columns to crater dataframe as needed for moonpies
    crater_df['cname'] = crater_df.index
    crater_df['age_upp'] = crater_df['age_low'] = crater_df['age']
    crater_df["rad"] = crater_df["diameter"] / 2

    with Pool() as p:
        args = [(crater_df.x.values[i], crater_df.y.values[i], crater_df.diameter.values[i], args.dim, args.res, tf) for i in len(crater_df)]
        in_crater_row = p.map(in_crater, args)

    crater_df['in_crater'] = in_crater_row

    with Pool() as p:
        args = [(crater_df.in_crater.values[i], psr, args.res) for i in len(crater_df)]
        psr_area = p.map(lambda args: np.sum(args[0] * args[1]) * (args[2]**2))

    crater_df['psr_area'] = psr_area

    # crater_df['in_crater'] = crater_df.apply(lambda row: in_crater(row["x"], row["y"], row["diameter"], args.dim, args.res, tf), axis=1)
    # crater_df['psr_flag'] = crater_df.apply(lambda row: row['in_crater'] * psr, axis=1)
    # crater_df['psr_area'] = crater_df.apply(lambda row: np.sum(row['psr_flag']) * (args.res**2), axis=1)

    return crater_df, psr


# main program
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='Path to dataset to use for running moonpies')
    parser.add_argument('--dim', type=int, default=200, help='Dimensions of map')
    parser.add_argument('--bbox', type=float, nargs=4, default=[0, 200, 200, 0], help='Bounding box in meters for map')
    parser.add_argument('--res', type=float, default=1., help='Resolution of map in meters per pixel')
    args = parser.parse_args()

    # make output directory
    if os.path.exists(os.path.join(args.datapath, 'moonpies')) == False:
        os.mkdir(os.path.join(args.datapath, 'moonpies'))

    # get ages from files
    files = os.listdir(args.datapath)
    map_files = [f for f in files if f.find('npz') > 0]
    crater_files = [f for f in files if f.find('.csv') > 0]
    time_steps = [int(re.sub('\D', '', f)) for f in map_files]
    time_steps_c = [int(re.sub('\D', '', f)) for f in crater_files]

    # sort things to make life easier
    map_files_sort = [f for _, f in sorted(zip(time_steps, map_files), reverse=True)]
    crater_files_sort = [f for _, f in sorted(zip(time_steps_c, crater_files), reverse=True)]
    time_steps.sort(reverse=True)
    # print(time_steps)

    # # load first file and show psrs within craters to make sure that's working properly
    # crater_df, psr_mask, surface = update_data(crater_files_sort[1], map_files_sort[1], time_steps[1] * 1e6, args)

    # for i in range(len(crater_df)):
    #     if crater_df.diameter.values[i] < 10:
    #         continue
    #     crater_flag = np.array(crater_df.in_crater.values[i])
    #     psr_flag = np.array(crater_df.psr_flag.values[i])
    #     # print(crater_flag.shape)
    #     # print(psr_flag.shape)
    #     psr_area = crater_df.psr_area.values[i]
    #     crater_area1 = np.sum(crater_flag) * (args.res**2)
    #     crater_area2 = np.pi * (crater_df.rad.values[i]**2)
    #     print(psr_area)
    #     print(crater_area1)
    #     print(crater_area2)
    #     print(np.abs(crater_area1 - crater_area2))
    #     print(psr_area / crater_area1)

    #     fig, ax = plt.subplots(1,3,figsize=(30,10))
    #     ax[0].imshow(surface)
    #     ax[1].imshow(psr_mask)
    #     ax[2].imshow(crater_flag, cmap='Greys', alpha=0.8)
    #     ax[2].imshow(psr_flag, cmap='Purples', alpha=0.5)
    #     plt.show()

    # loop through iterations
    for i in range(1, len(time_steps)):
    # for i in range(1, 3):

        print("Current time step (Myr): %4.4f" % (time_steps[i]))

        # load data
        crater_file = crater_files_sort[i]
        map_file = map_files_sort[i]
        if i == 1:
            start_time = (time_steps[i] + 100) * 1e6
        else:
            start_time = time_steps[i-1] * 1e6
        # print(start_time)

        end_time = time_steps[i] * 1e6
        # print(crater_file)
        # print(map_file)
        crater_df, psr_mask = update_data(crater_files_sort[i], map_files_sort[i], start_time, end_time, args)
        # print(crater_df[0:10])

        # print out the range of crater ages to check that they're correct
        print("Age range (Myr): (%4.4f, %4.4f) " % (np.min(crater_df.age.values) / 1e6, np.max(crater_df.age.values) / 1e6))
        # print(np.min(crater_df.age.values) / 1e6)
        # print(np.max(crater_df.age.values) / 1e6)

        # in_crater_np = np.array(crater_df.in_crater.tolist())
        # in_psr_np = np.array(crater_df.psr_flag.tolist())
        # print(in_crater_np.shape) # 214 x 200 x 200
        # plot the craters and psrs
        # fig, ax = plt.subplots(1,2,figsize=(20,10))
        # ax[0].imshow(np.any(in_crater_np, axis=0), cmap='Oranges')
        # ax[1].imshow(np.any(in_psr_np, axis=0), cmap='Blues')
        # plt.show()

        if i == 1:
            mp_cfg = mp_config.read_custom_cfg('../moonpies/moonpies/configs/lvsim_config.py',
                                               seed=9324712,
                                               gridsize=args.dim,
                                               gridres=args.res,
                                               outpath=os.path.join(args.datapath, 'moonpies'))
            mp_sim = MoonPIES(mp_cfg, crater_db=crater_df, psr_mask=psr_mask)
        else:
            mp_sim.update_crater_info(crater_db=crater_df, psr_mask=psr_mask)

        # run new moonpies iters
        mp_sim.run_between(start_time, end_time)

        # plot the results
        mp_sim.show()
