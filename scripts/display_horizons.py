# date:     7-31-2025   
# author:   margaret hansen
# purpose:  show horizons data

import os
import argparse
import imageio
import rasterio as rs
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection

import matplotlib
matplotlib.use('TkAgg')

# function to display data
def display_data(elev_db, azim_res=1.):

    plt.ion()
    fig, ax = plt.subplots()

    for i in range(0, 360, azim_res):

        # load the current dataset
        data = elev_db[i,...]

        # update the plot
        if i == 0:
            img = plt.imshow(data, cmap='Blues')
        else:
            img.set_data(data)
        ax.set_title("Azimuth: " + str(i))
        plt.pause(0.05)


# function to make and save images
def make_images(elev_db, outpath, azim_res=1.):

    if ~os.path.exists(os.path.join(outpath, "images")):
        os.mkdir(os.path.join(outpath, "images"))

    for i in range(0, 360, azim_res):

        fig, ax = plt.subplots()

        # load the current dataset
        data = elev_db[i,...]

        # add data to the plot and save
        plt.imshow(data, cmap='Blues')
        ax.set_title("Azimuth: " + str(i))
        plt.savefig(os.path.join(outpath, "images/horizon_" + str(i) + ".png"), dpi=100, bbox_inches="tight")
        plt.close()

    # now make the gif of the images
    imgs = []
    for i in range(0, 360, azim_res):

        # load the image
        name = os.path.join(outpath, "images/horizon_" + str(i) + ".png")
        img = imageio.imread(name)
        imgs.append(img)

    # save the output
    # duration is in ms
    imageio.mimsave(os.path.join(outpath, "images/horizons.gif"), imgs, duration = 100, loop=0)


# function for interactive selection of points to generate horizons
def horizon_picker(elev_db, hmap, hmap_res, max_range, azim_res=1.):

    # load heightmap
    # tif_file = [f for f in os.listdir(hmap_path) if f.find('tif') >= 0]
    # hmap_f = rs.open(os.path.join(hmap_path, tif_file[0]))
    # hmap = hmap_f.read(1)

    # limit heightmap and elevation maps to border
    r = int(np.floor(max_range * 1000 / hmap_res))
    h,w = hmap.shape
    dem_data_limit = hmap[r:(h-r), r:(w-r)]
    # elev_db_limit = elev_db[:, r:(h-r), r:(w-r)]
    elev_db_limit = elev_db

    del(hmap)
    del(elev_db)
    # hmap_f.close()
    
    # load elevation database
    azims = np.arange(0, 360, azim_res)

    # display 3 plots
    
    # plt.ion()
    fig = plt.figure(layout="constrained")
    gs = GridSpec(2,2, figure=fig)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
    
    # add some titles
    ax1.set_title('Horizon profile for selected point')
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Elevation (degrees)')

    ax2.set_title('Terrain height (m)')
    ax3.set_title('Horizon elevation map for selected azimuth')

    # initial plots
    azim = 0
    pt = np.array([int(h/2), int(w/2)])
    heights = elev_db_limit[:, pt[1], pt[0]]
    
    l1, = ax1.plot(azims, heights, c='tab:blue')
    l2 = ax1.axvline(azim, ymin=0, ymax=np.max(heights), c='gray', linestyle='dashed')
    ax1.set_ylim(np.min(elev_db_limit)-0.25, np.max(elev_db_limit)+0.25)

    ax2.imshow(dem_data_limit, cmap='terrain')
    pt1 = ax2.scatter(pt[0], pt[1], c='red', s=2)

    im1 = ax3.imshow(elev_db_limit[azim, ...], cmap='Blues')
    pt2 = ax3.scatter(pt[0], pt[1], c='red', s=2)

    # hook for mouse selection of points in image
    def onclick(event):

        # switch between selections based on which plot was clicked
        curr_ax = event.inaxes
        if len(curr_ax.get_images()) > 0:
            
            x = event.xdata
            y = event.ydata

            # update points displayed in plot
            pt1.set_offsets(np.c_[x, y])
            pt2.set_offsets(np.c_[x, y])

            # update horizon line
            heights = elev_db_limit[:, int(y), int(x)]
            l1.set_data(azims, heights)
        
        elif len(curr_ax.get_lines()) > 0:

            # update the axis for the horizon elevation plot
            azim = event.xdata
            im1.set_data(elev_db_limit[int(azim), ...])
            l2.set_xdata([azim, azim])
        
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

