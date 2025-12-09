"""
Command-line interface for lunar volatiles sim

Usage: 
    lvsim [options]

Options:
    

Example: 

"""

from lvsim.lvsim import LvSim
from lvsim.utils import LvSimCfg
from lvsim.analysis import analyze_sim

def run():
    
    # initialize config and sim
    cfg = LvSimCfg()
    lvsim = LvSim(cfg)

    # run sim
    lvsim.run_all()

    # analyze results
    analyze_sim(cfg)
    

if __name__ == "__main__":
    run()
