# date:     7-28-2025
# author:   margaret hansen
# purpose:  generate synthetic data of lunar volatiles distributions

import os
import raytrace

from shapely.geometry import box
from synthterrain import crater
from synthterrain.crater import functions
from moonpies import moonpies

from utils import LvSimCfg


# class for lunar volatiles sim
class LvSim():

    def __init__(self, cfg):

        self.cfg = cfg

        # set up list of craters
        poly = box(cfg.args.bbox[0], cfg.args.bbox[3], cfg.args.bbox[2], cfg.args.bbox[1])

        crater_dist = getattr(functions, cfg.args.csfd)(a=cfg.args.d_lim[0], b=cfg.args.d_lim[1])

        self.crater_df = crater.synthesize(
            crater_dist,
            polygon=poly,
            by_bin=False,
            min_d=cfg.args.d_lim[0],
            max_d=cfg.args.d_lim[1],
        )

        # save the list of craters
        crater.to_file(self.crater_df, cfg.args.outfile, False)

        # initialize list of times based on crater ages



    # Apply synthterrain diffusion model to current time step to get height map
    def gen_terrain(self, time):
        pass


    # Use illumination maps and moonpies to generate ice distribution
    def gen_ice_dist(self, time):
        pass    

    # Produce sensor observations for a specific location
    def gen_sensor_obs(self, time):
        pass


    # Run through all steps
    def run_all(self):
        pass



if __name__ == "__main__":

    # parse args
    cfg = LvSimCfg()

    # init sim
    lvsim = LvSim(cfg)

    # run sim
    lvsim.run_all()

