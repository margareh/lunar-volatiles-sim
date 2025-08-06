# date:     7-29-2025
# author:   margaret hansen
# purpose:  input arguments class for lunar volatiles sim

import argparse

class LvSimCfg():

    def __init__(self):

        # setup argument parser        
        parser = argparse.ArgumentParser()

        # paths
        parser.add_argument('--outpath',
                            type=str,
                            help='Location for storing output')

        # args for synthterrain
        parser.add_argument('--bbox',
                            type=float,
                            nargs=4,
                            default=[0, 1000, 1000, 0],
                            help='Bounding box in m, ordered as min-x, may-y, max-x, min-y')
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
        
        # parse args at the end and save
        self.args = parser.parse_args()

        # validate inputs
        self.validate_args()


    # check that args are correct
    def validate_args(self):

        if self.args.csfd not in ['Trask', 'VIPER_Env_Spec']:
            print("Invalid crater SFD provided, defaulting to VIPER_Env_Spec")
            self.args.csfd = 'VIPER_Env_Spec'

