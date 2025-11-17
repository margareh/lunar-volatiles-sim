# date: 9-5-2025
# author: margaret hansen
# purpose: analyze results of synthetic terrain sim

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LightSource
import imageio
import rasterio as rs
import richdem as rd

# from synthterrain.crater import functions, determine_production_function
# from synthterrain.crater.age import equilibrium_age

# define the size frequency distribution for VIPER
def viper_sfd(d):
    """
    CSFD( d <= 80 ) = (29174 / d^(1.92)) / (1000^2)
    CSFD( d > 80 ) = (156228 / d^(2.389)) / (1000^2)
    
    This is number per square meter for diameter in meters
    """
    small = (d <= 80)
    out = 156228 * np.float_power(d, -2.389)
    out[small] = 29174 * np.float_power(d[small], -1.92)
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


# # plot the crater diameter by age
# def plot_diam_by_age(file, args):

#     # load
#     crater_df = pd.read_csv(os.path.join(args.datapath, file))
#     diams = crater_df.diameter.values
#     min_d = np.min(diams)
#     max_d = np.max(diams)
#     bins = np.arange(int(min_d), int(max_d))
#     ages = crater_df["age"].values
#     diam_age_hist, _ = np.histogram(diams, bins=bins, weights=ages)
#     diam_hist, _ = np.histogram(diams, bins=bins)
#     diam_ages = diam_age_hist / diam_hist

#     # compute the equilibrium ages for each diameter
#     crater_dist = getattr(functions, "VIPER_Env_Spec")(a=min_d, b=max_d)
#     prod_fn = determine_production_function(crater_dist.a, crater_dist.b)
#     eq_ages = equilibrium_age(diams, prod_fn.csfd, crater_dist.csfd)

#     # plot
#     plt.scatter(bins[:-1], diam_ages * 1e-9, label='Average Age per 1 m Bin')
#     plt.scatter(diams, eq_ages * 1e-9, linestyle='dashed', color='gray', label='Equilibrium Age')
#     plt.legend(loc="upper right")
#     if args.plot:
#         plt.show()
#     else:
#         plt.savefig(os.path.join(args.datapath, 'figs', file.replace('.csv', '_age_dist.png')), dpi=100, bbox_inches='tight')
#         plt.close()


# plot slope histogram of surface vs that of haworth DEM
def plot_slope_hist(file, args):

    # load haworth DEM
    haworth_dem_f = rs.open(args.haworth_dem)
    haworth_dem = haworth_dem_f.read(1)
    haworth_proj = haworth_dem_f.crs.to_string()
    haworth_dem_f.close()

    # compute slope of haworth DEM
    haworth_dem_rd = rd.rdarray(haworth_dem, no_data=haworth_dem_f.nodata)
    haworth_dem_rd.projection = haworth_proj
    haworth_slope = rd.TerrainAttribute(haworth_dem_rd, attrib='slope_radians')
    haworth_max = np.nanmax(haworth_slope) * (180 / np.pi)
    # print(np.nanmin(haworth_slope) * (180 / np.pi))
    # print(haworth_max)
    # print(haworth_slope.shape)

    # load surface
    maps = np.load(os.path.join(args.datapath, file))
    surf = maps['surface']

    # compute slope of surface
    surf_rd = rd.rdarray(surf, no_data=np.nan)
    # surf_rd.projection = '+proj=utm +zone=11 +ellps=WGS84 + datum=WGS84 +units=m +no_defs +type=crs'
    surf_rd.projection = haworth_proj
    surf_slope = rd.TerrainAttribute(surf_rd, attrib='slope_radians')
    surf_max = np.nanmax(surf_slope) * (180 / np.pi)
    # print(np.nanmin(surf_slope) * (180 / np.pi))
    # print(surf_max)
    # print(surf_slope.shape)
    
    # plot both beside one another
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(10, 5)
    fig.suptitle(file)

    max_slope = max(haworth_max, surf_max)
    print("Max slope:")
    print(max_slope)
    surf_slope_sample = surf_slope[::2, ::2] * (180 / np.pi)
    haworth_slope_sample = haworth_slope[::50, ::50] * (180 / np.pi)
    ax[0].imshow(surf_slope_sample, cmap='Spectral_r', vmin=0, vmax=max_slope)
    im = ax[1].imshow(haworth_slope_sample, cmap='Spectral_r', vmin=0, vmax=max_slope)

    ax[0].set_title('Synthetic')
    ax[1].set_title('Haworth DEM')

    div = make_axes_locatable(ax[1])
    cax = div.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.datapath, 'figs', file.replace('.npz', '_slope_imgs.png')), dpi=100, bbox_inches='tight')
        plt.close()

    # plot both on same histogram
    bins = np.linspace(0, max_slope, 30)
    fig, ax = plt.subplots()
    ax.hist(surf_slope_sample.flatten(), bins, alpha=0.5, label='Synthetic', color='tab:blue')
    ax2 = ax.twinx()
    ax2.hist(haworth_slope_sample.flatten(), bins, alpha=0.5, label='Haworth', color='tab:orange')
    ax.set_ylabel('Synthetic')
    ax2.set_ylabel('Haworth')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    hs = h1+h2
    labs = l1+l2
    ax.legend(hs, labs, loc='upper right')
    # plt.legend(loc='upper right')

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.datapath, 'figs', file.replace('.npz', '_slope_hist.png')), dpi=100, bbox_inches='tight')
        plt.close()

    return surf, haworth_dem

# get FFT shifted to center for an image
def get_fft(img):
    ft = np.fft.ifftshift(img)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return ft


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='Path to data to analyze')
    parser.add_argument('--dim', type=int, default=200, help='Side dimensions of map (number of pixels)')
    parser.add_argument('--res', type=float, default=1., help='Resolution of map (meters per pixel)')
    parser.add_argument('--plot', action='store_true', help='Flag to plot analysis figures instead of saving them')
    parser.add_argument('--age', type=float, nargs=2, default=[3.79, 0], help='Age of first and last non-flat terrain surface in Gyr')
    parser.add_argument('--haworth_dem', type=str, help='File path to Haworth DEM', default='/media/usb/ThesisWork/Volatiles/SouthPoleData/Haworth_DEM_1mpp/Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif')
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.datapath, 'figs')) == False:
        os.mkdir(os.path.join(args.datapath, 'figs'))

    first_step = str(int(args.age[0] * 1000))
    last_step= str(int(args.age[1] * 1000))

    # crater ages and diameters over time to make sure these are trending correctly
    # plot_diam_by_age('crater_list_'+first_step+'.csv', args)
    # plot_diam_by_age('crater_list_0.csv', args)

    # plot the SFD for the first and last time steps
    plot_sfd('crater_list_'+first_step+'.csv', args) # first one with craters
    plot_sfd('crater_list_'+last_step+'.csv', args) # last one

    # get slope histogram and compare to Haworth DEM
    surf_dem_first, haworth_dem = plot_slope_hist('maps_'+first_step+'.npz', args)
    surf_dem_last, _ = plot_slope_hist('maps_'+last_step+'.npz', args)

    # hillshaded DEMs and ffts of DEMs
    haworth_ft = get_fft(haworth_dem)
    first_ft = get_fft(surf_dem_first)
    last_ft = get_fft(surf_dem_last)

    ls = LightSource(azdeg=315, altdeg=6) # 6 degree altitude (like lunar south pole), from NW
    fig, ax = plt.subplots(2, 3, figsize=(30, 20))

    ax[0][0].imshow(ls.hillshade(haworth_dem), cmap='gray')
    ax[0][1].imshow(ls.hillshade(surf_dem_first), cmap='gray')
    ax[0][2].imshow(ls.hillshade(surf_dem_last), cmap='gray')

    ax[0][0].set_title('Haworth')
    ax[0][1].set_title('First Surface')
    ax[0][2].set_title('Last Surface')

    ax[1][0].imshow(abs(haworth_ft), cmap='gray')
    ax[1][1].imshow(abs(first_ft), cmap='gray')
    ax[1][2].imshow(abs(last_ft), cmap='gray')

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.datapath, 'figs', 'hillshade_dems_ffts.png'), dpi=100, bbox_inches='tight')
        plt.close()

    # # create gifs of maps
    # out_files = os.listdir(args.datapath)
    # plot_files = os.listdir(os.path.join(args.datapath, 'plots'))
    # time_steps = [int(re.sub('\D', '', f)) for f in plot_files]
    # # map_files = [f for f in out_files if f.find("maps") >= 0]
    # # crater_files = [f for f in out_files if f.find("crater") >= 0]
    
    # plot_files_sort = [f for _, f in sorted(zip(time_steps, plot_files), reverse=True)]
    # # crater_files_sort = [f for _, f in sorted(zip(time_steps, crater_files), reverse=True)]
    # T = len(plot_files_sort)

    # imgs = []
    # for t in range(T):
    #     print(t)
    #     new_img = imageio.imread(os.path.join(args.datapath, 'plots', plot_files_sort[t]))
    #     imgs.append(new_img)

    # make gif of heightmaps
    # imageio.mimsave(os.path.join(args.datapath, 'figs', 'hmaps.gif'), imgs, duration = 500, loop=0)
