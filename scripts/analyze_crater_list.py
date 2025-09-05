# date: 9-5-2025
# author: margaret hansen
# purpose: analyze results of synthetic terrain sim

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

from synthterrain.crater import functions, determine_production_function
from synthterrain.crater.age import equilibrium_age

# define the size frequency distribution for VIPER
def viper_sfd(d):
    """
    CSFD( d <= 80 ) = (29174 / d^(1.92)) / (1000^2)
    CSFD( d > 80 ) = (156228 / d^(2.389)) / (1000^2)
    
    This is number per square meter for diameter in meters
    """
    small = (d <= 80)
    out = 156228 * np.float_power(d, -1.92)
    out[small] = 29174 * np.float_power(d[small], -2.389)
    return out / (1000 * 1000)

# plot the empirical size frequency distribution
def plot_sfd(file, args):

    print(file)

    # load the last crater list
    # has columns: x, y, diameter, age, d/D
    crater_df = pd.read_csv(os.path.join(args.datapath, file))
    min_d = np.min(crater_df.diameter.values)
    max_d = np.max(crater_df.diameter.values)
    print("Min diameter (m): %4.2f" % (min_d)) # 1.0
    print("Max diameter (m): %4.2f" % (max_d)) # 258.7757

    counts, bins, _ = plt.hist(crater_df.diameter, bins=np.arange(min_d, max_d))
    plt.close()

    cumul_counts = np.cumsum(counts[::-1])[::-1]
    area = (args.dim * args.res) ** 2
    viper_counts = viper_sfd(np.arange(min_d, max_d)) * area

    # plot SFD of craters
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(20,10)
    fig.suptitle(file)
    ax[0].hist(crater_df.diameter, bins=30)
    ax[0].set_title('Histogram')
    ax[0].set_xlabel('Crater Diameter (m)')
    ax[0].set_ylabel('Count')

    ax[1].plot(bins[:-1], cumul_counts, label='Simulated')
    ax[1].plot(bins, viper_counts, color='gray', linestyle='dashed', label='VIPER Spec')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_title('SFD')
    ax[1].set_xlabel('Crater Diameter (m)')
    ax[1].set_ylabel('Count')
    ax[1].legend(loc='upper right')

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.datapath, 'figs', file.replace('.csv', '_sfd.png')), dpi=100, bbox_inches='tight')
        plt.close()


# plot the crater diameter by age
def plot_diam_by_age(file, args):

    # load
    crater_df = pd.read_csv(os.path.join(args.datapath, file))
    diams = crater_df.diameter.values
    min_d = np.min(diams)
    max_d = np.max(diams)
    bins = np.arange(int(min_d), int(max_d))
    ages = crater_df["age"].values
    diam_age_hist, _ = np.histogram(diams, bins=bins, weights=ages)
    diam_hist, _ = np.histogram(diams, bins=bins)
    diam_ages = diam_age_hist / diam_hist

    # compute the equilibrium ages for each diameter
    crater_dist = getattr(functions, "VIPER_Env_Spec")(a=min_d, b=max_d)
    prod_fn = determine_production_function(crater_dist.a, crater_dist.b)
    eq_ages = equilibrium_age(diams, prod_fn.csfd, crater_dist.csfd)

    # plot
    plt.scatter(bins[:-1], diam_ages * 1e-9, label='Average Age per 1 m Bin')
    plt.scatter(diams, eq_ages * 1e-9, linestyle='dashed', color='gray', label='Equilibrium Age')
    plt.legend(loc="upper right")
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.datapath, 'figs', file.replace('.csv', '_age_dist.png')), dpi=100, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='Path to data to analyze')
    parser.add_argument('--dim', type=int, default=200, help='Side dimensions of map (number of pixels)')
    parser.add_argument('--res', type=float, default=1., help='Resolution of map (meters per pixel)')
    parser.add_argument('--plot', action='store_true', help='Flag to plot analysis figures instead of saving them')
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.datapath, 'figs')) == False:
        os.mkdir(os.path.join(args.datapath, 'figs'))

    # create gifs of maps
    out_files = os.listdir(args.datapath)
    plot_files = os.listdir(os.path.join(args.datapath, 'plots'))
    time_steps = [int(re.sub('\D', '', f)) for f in plot_files]
    # map_files = [f for f in out_files if f.find("maps") >= 0]
    # crater_files = [f for f in out_files if f.find("crater") >= 0]
    
    plot_files_sort = [f for _, f in sorted(zip(time_steps, plot_files), reverse=True)]
    # crater_files_sort = [f for _, f in sorted(zip(time_steps, crater_files), reverse=True)]
    T = len(plot_files_sort)

    imgs = []
    for t in range(T):
        print(t)
        new_img = imageio.imread(os.path.join(args.datapath, 'plots', plot_files_sort[t]))
        imgs.append(new_img)

    # make gif of heightmaps
    imageio.mimsave(os.path.join(args.datapath, 'figs', 'hmaps.gif'), imgs, duration = 500, loop=0)

    # also look at the crater ages and diameters over time to make sure these are trending correctly
    plot_diam_by_age('crater_list_4240.csv', args)
    plot_diam_by_age('crater_list_0.csv', args)

    # plot the SFD for the first and last time steps
    plot_sfd('crater_list_4240.csv', args) # first one with craters
    plot_sfd('crater_list_0.csv', args) # last one
