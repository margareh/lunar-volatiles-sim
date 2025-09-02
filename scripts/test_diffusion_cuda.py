# date: 8-21-2025
# author: margaret hansen
# purpose: test diffusion with CUDA

import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from lvsim.lvsim import profile, stopar_fresh_dd, make_heightmap
from lvsim.utils import LvSimCfg

from synthterrain import crater
from synthterrain.crater import functions
from synthterrain.crater.diffusion import make_crater_field

from diffusion.diffusion import diffusion_cuda

if __name__ == "__main__":

    # create list of craters
    cfg = LvSimCfg()

    # set seed
    np.random.seed(cfg.args.seed)
    random.seed(cfg.args.seed)

    # set up initial list of craters
    poly = box(cfg.args.bbox[0], cfg.args.bbox[3], cfg.args.bbox[2], cfg.args.bbox[1])
    transform = from_origin(
        poly.bounds[0], poly.bounds[3], cfg.args.res, cfg.args.res
    )
    window = from_bounds(*poly.bounds, transform=transform)
    crater_dist = getattr(functions, cfg.args.csfd)(a=cfg.args.d_lim[0], b=cfg.args.d_lim[1])

    # diffusion with synthterrain
    print("Running synthterrain diffusion")
    crater_df = crater.synthesize(
        crater_dist,
        polygon=poly,
        min_d=cfg.args.d_lim[0],
        max_d=cfg.args.d_lim[1],
        return_surfaces=True,
        by_bin=False,
    )
    # print(self.crater_df)

    # surface from synthterrain crater list
    start = time.time()
    surface = make_crater_field(
        crater_df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform
    )
    end = time.time()
    print("Elapsed time from make_crater_field: %4.2f" % (end-start))

    # diffusion with binned synthterrain
    print("Running binned synthterrain diffusion")
    np.random.seed(cfg.args.seed)
    random.seed(cfg.args.seed)
    crater2_df = crater.synthesize(
        crater_dist,
        polygon=poly,
        min_d=cfg.args.d_lim[0],
        max_d=cfg.args.d_lim[1],
        return_surfaces=True,
        by_bin=True
    )
    surface_bin = make_crater_field(crater2_df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform)

    # diffusion with cuda
    print("Running CUDA diffusion")
    init_df = pd.DataFrame(columns=['x','y','diameter','age','d/D','surface'])
    init_df['x'] = crater_df['x']
    init_df['y'] = crater_df['y']
    init_df['diameter'] = crater_df['diameter']
    init_df['age'] = crater_df['age']
    init_df['d/D'] = stopar_fresh_dd(crater_df['diameter'])
    init_df['surface'] = init_df.apply(lambda row: profile(row["d/D"], row["diameter"], D=cfg.args.domain_size), axis=1)

    # plt.imshow(init_df['surface'][0])
    # plt.show()

    surfs_np = np.array(init_df["surface"].tolist())
    start = time.time()
    new_ratios, new_surfs = diffusion_cuda(init_df["diameter"], init_df["d/D"], init_df["age"], surfs_np, D=cfg.args.domain_size)
    end = time.time()
    print("Elapsed time from crater diffusion with CUDA: %4.2f" % (end-start))

    init_df['d/D'] = new_ratios
    init_df["surface"] = [s for s in new_surfs[:,...]]

    # surface_cuda = make_crater_field(init_df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform)
    start = time.time()
    surface_cuda = make_heightmap(crater_df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform)
    end = time.time()
    print("Elapsed time from make_heightmap: %4.2f" % (end-start))

    # comparison
    ratio_compare = np.array([crater_df['d/D'].values.tolist(), new_ratios.tolist(), crater2_df['d/D'].values.tolist()]).T
    diff = ratio_compare[:,0] - ratio_compare[:,1]
    diff2 = ratio_compare[:,0] - ratio_compare[:,2]
    # print(ratio_compare)
    print("d/D MAE between synthterrain and diffusion_cuda: %4.4e" % (np.mean(np.abs(diff))))
    print("d/D MAE between synthterrain and binned synthterrain: %4.4e" % (np.mean(np.abs(diff2))))

    height_diff = np.mean(np.abs(surface - surface_cuda))
    height_diff2 = np.mean(np.abs(surface - surface_bin))
    print("Height MAE between synthterrain and diffusion_cuda: %4.4e" % (height_diff))
    print("Height MAE between synthterrain and binned synthterrain: %4.4e" % (height_diff2))

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(surface, cmap='terrain')
    ax[1].imshow(surface_cuda, cmap='terrain')
    ax[2].imshow(surface_bin, cmap='terrain')
    ax[0].set_title('synthterrain')
    ax[1].set_title('diffusion_cuda')
    ax[2].set_title('synthterrain binned')

    plt.show()
    # plt.savefig('../figs/diffusion_comparison.png', dpi=100, bbox_inches='tight')
    # plt.close()
