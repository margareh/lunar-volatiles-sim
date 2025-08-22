# date: 8-21-2025
# author: margaret hansen
# purpose: test diffusion with CUDA

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from lvsim.lvsim import profile, stopar_fresh_dd
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
        by_bin=False
    )
    # print(self.crater_df)

    # surface from synthterrain crater list
    surface = make_crater_field(
        crater_df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform
    )

    # diffusion with cuda
    print("Running CUDA diffusion")
    init_df = pd.DataFrame(columns=['x','y','diameter','age','d/D','surface'])
    init_df['x'] = crater_df['x']
    init_df['y'] = crater_df['y']
    init_df['diameter'] = crater_df['diameter']
    init_df['age'] = crater_df['age']
    init_df['d/D'] = stopar_fresh_dd(crater_df['diameter'])
    init_df['surface'] = init_df.apply(lambda row: profile(row["d/D"], row["diameter"], D=cfg.args.domain_size), axis=1)

    surfs_np = np.array(init_df["surface"].tolist())
    new_ratios, new_surfs = diffusion_cuda(init_df["diameter"], init_df["d/D"], init_df["age"], surfs_np, D=cfg.args.domain_size)

    init_df['d/D'] = new_ratios
    init_df["surface"] = [s for s in new_surfs[:,...]]

    surface_cuda = make_crater_field(init_df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform)

    # comparison
    ratio_compare = np.array([crater_df['d/D'].values.tolist(), new_ratios.tolist()]).T
    diff = ratio_compare[:,0] - ratio_compare[:,1]
    print(ratio_compare)
    print(diff)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(surface, cmap='terrain')
    ax[1].imshow(surface_cuda, cmap='terrain')
    ax[0].set_title('synthterrain')
    ax[1].set_title('diffusion_cuda')

    plt.show()
