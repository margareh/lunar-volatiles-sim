"""
Command-line interface for lunar volatiles sim

Usage: 
    lvsim N [options]

Arguments:
    N = number of iterations to run

Options (most useful ones are listed below):
    seed            Random seed
    bbox            Bounding box for map in meters [left top right bottom]
    res             Map resolution in meters
    max_range       Maximum range for horizon searching
    d_lim           Diameter limits for craters
    csfd            Type of crater SFD to use
    outpath         Path for saving results
    eph_file        Path to ephemeris file for illumination calculations
    mp_cfg          Path to moonpies config file
    nss_file        Path to NSS calibration file
    haworth_dem     Path to Haworth DEM for results comparison
    plot            Flag for showing plots instead of storing
    surface_age     Flag for computing surface age at each iteration [DO NOT USE WITH LARGE MAPS]

Example:
    lvsim 1 --bbox 0 1000 1000 0 --max_range 0.2 --d_lim 2 1000
    lvsim 50 --bbox 0 200 200 0 --max_range 0.04 --d_lim 2 200

"""

import os
from lvsim.sim import LvSim
from lvsim.utils import LvSimCfg
from lvsim.analysis import analyze_sim

def adjust_args(args):

    args.seed += 1

    orig_outpath, _ = os.path.split(args.outpath)
    args.outpath = os.path.join(orig_outpath, str(args.seed))

    if os.path.exists(args.outpath) == False:
        os.mkdir(args.outpath)
    
    if os.path.exists(os.path.join(args.outpath, 'plots')) == False:
        os.mkdir(os.path.join(args.outpath, 'plots'))
    
    args.mppath = os.path.join(args.outpath, 'moonpies')

    return args


def run():
    
    # initialize config and sim
    cfg = LvSimCfg()
    lvsim = LvSim(cfg)

    # loop through and run the sim for the correct number of times
    for i in range(cfg.args.loop):

        print("#---------------------- Now on simulation %d / %d ----------------------#" % (i+1, cfg.args.loop))

        # Override print to always display iteration number in print messages
        def print(*args, **kwargs):
            __builtins__.print('%d / %d: ' % (i+1, cfg.args.loop), end='')
            return __builtins__.print(*args, **kwargs)

        # change args if necessary
        if i > 0:
            cfg.args = adjust_args(cfg.args)

        # run sim
        lvsim.run_all()

        # analyze results
        analyze_sim(cfg.args)
    

if __name__ == "__main__":
    run()
