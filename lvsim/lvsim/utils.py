# date:     7-29-2025
# author:   margaret hansen
# purpose:  input arguments class for lunar volatiles sim

import os
import argparse

class LvSimCfg():

    def __init__(self):

        # setup argument parser
        parser = argparse.ArgumentParser()

        # paths
        parser.add_argument('--outpath',
                            type=str,
                            default='../../output',
                            help='Location for storing output')
        
        # args for lvsim
        parser.add_argument('--max_age',
                            type=float,
                            default=4.25,
                            help='Maximum age in gigayears (AKA start time of model)')
        parser.add_argument('--time_delta',
                            type=float,
                            default=0.01,
                            help='Time delta in gigayears')

        # args for synthterrain
        parser.add_argument('--bbox',
                            type=float,
                            nargs=4,
                            default=[0, 1000, 1000, 0],
                            help='Bounding box in m, ordered as min-x, may-y, max-x, min-y')
        parser.add_argument('--res',
                            type=float,
                            default=1.,
                            help='Resolution of terrain model to generate')
        parser.add_argument('--d_lim',
                            type=float,
                            nargs=2,
                            default=[1, 1000],
                            help='Minimum and maximum allowed crater diameters in m')
        parser.add_argument('--csfd',
                            type=str,
                            default='VIPER_Env_Spec',
                            help='Name of crater size-frequency distribution to use (only supports Trask or VIPER_Env_Spec)')

        # behavior flags
        parser.add_argument('--plot',
                            action='store_true',
                            help='Flag for plotting results')
        
        # other args
        parser.add_argument('--seed',
                            type=int,
                            default=12957973,
                            help='Random seed')
        
        # parse args at the end and save
        self.args = parser.parse_args()

        # validate inputs
        self.validate_args()


    # check that args are correct
    def validate_args(self):

        if self.args.csfd not in ['Trask', 'VIPER_Env_Spec']:
            print("Invalid crater SFD provided, defaulting to VIPER_Env_Spec")
            self.args.csfd = 'VIPER_Env_Spec'

        # add seed to output file path and make directory
        self.args.outpath = os.path.join(self.args.outpath, str(self.args.seed))

        if os.path.exists(self.args.outpath) == False:
            os.mkdir(self.args.outpath)

