"""
Command-line interface for lunar volatiles sim

Usage: 
    lvsim [options]

Options:
    

Example: 

"""

import argparse
from lvsim import LvSim
from lvsim.utils import LvSimCfg

def run():
    
    # initialize config and sim
    cfg = LvSimCfg()
    lvsim = LvSim(cfg)

    # run sim
    lvsim.run_all()
    

if __name__ == "__main__":
    run()
