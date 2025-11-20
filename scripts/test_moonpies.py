# date:     11-20-2025
# author:   margaret hansen
# purpose:  run moonpies on crater dataset

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import box
from rasterio.transform import from_origin

from lvsim.crater import in_crater

from moonpies import MoonPIES
from moonpies import config as mp_config

def get_psr_area(in_crater_flag, rad, psr, px_area = 1):
    crater_area = np.pi * (rad **2)
    psr_area = np.sum(in_crater_flag * psr) * px_area / crater_area
    return psr_area

# update crater and psr mask data
def update_data(crater_file, psr_file, end_age, args):

    # load crater dataset and psr mask
    crater_df = pd.read_csv(os.path.join(args.datapath, crater_file), header=0, index_col=0)
    maps = np.load(os.path.join(args.datapath, psr_file))
    # surf = maps['surface']
    # illumin = maps['illumin_frac']
    psr = maps['psr']

    # transform for crater
    poly = box(args.bbox[0], args.bbox[3], args.bbox[2], args.bbox[1])
    tf = from_origin(poly.bounds[0], poly.bounds[3], args.res, args.res)

    # increase crater age based on length of sim
    crater_df['age'] += end_age

    # add columns to crater dataframe as needed for moonpies
    crater_df['cname'] = crater_df.index
    crater_df['age_upp'] = crater_df['age_low'] = crater_df['age']
    crater_df["rad"] = crater_df["diam"] / 2
    crater_df['in_crater'] = crater_df.apply(lambda row: in_crater(row["x"], row["y"], row["diameter"], args.dim, args.res, tf), axis=1)
    crater_df['psr_area'] = crater_df.apply(lambda row: get_psr_area(row["in_crater"], row["rad"], psr, px_area=args.res**2), axis=1)

    return crater_df, psr


# main program
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='Path to dataset to use for running moonpies')
    parser.add_argument('--dim', type=int, default=200, help='Dimensions of map')
    parser.add_argument('--bbox', type=float, default=[0, 200, 200, 0], help='Bounding box in meters for map')
    parser.add_argument('--res', type=float, default=1., help='Resolution of map in meters per pixel')
    args = parser.parse_args()

    # make output directory
    if ~os.path.exists(os.path.join(args.datapath, 'moonpies')):
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

    # loop through iterations
    init=False
    for i in range(1, len(time_steps)):

        # load data
        crater_df, psr_mask = update_data(crater_files_sort[i], map_files_sort[i], time_steps[i] * 1e6, args)

        if ~init:
            # initialize moonpies sim
            mp_cfg = mp_config.read_custom_cfg('../../moonpies/moonpies/configs/lvsim_config.py', 9324712)
            mp_cfg.grdxsize = args.dim
            mp_cfg.grdysize = args.dim
            mp_cfg.grdstep = args.res
            mp_cfg.out_path = os.path.join(args.datapath, 'moonpies')
            mp_sim = MoonPIES(mp_cfg, crater_db=crater_df, psr_mask=psr_mask)
            init=True

        # update time start and end in moonpies config
        mp_sim.cfg.timestart = time_steps[i-1]
        mp_sim.cfg.timeend = time_steps[i]

        # update data in moonpies
        mp_sim.update_crater_info(crater_db=crater_df, psr_mask=psr_mask)

        # run new moonpies iters
        mp_sim.run()

        # plot the results
        mp_sim.show()
