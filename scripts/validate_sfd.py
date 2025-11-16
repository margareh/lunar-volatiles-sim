# date: 11-15-2025
# author: margaret hansen
# purpose: plot crater distribution

import argparse
import numpy as np
import matplotlib.pyplot as plt

from numbers import Number
from analyze_crater_list import viper_sfd

# SFD from synthterrain
def csfd(d):
    """
    CSFD( d <= 80 ) = (29174 / d^(1.92)) / (1000^2)
    CSFD( d > 80 ) = (156228 / d^(2.389)) / (1000^2)

    """
    if isinstance(d, Number):
        # Convert to numpy array, if needed.
        diam = np.array(
            [
                d,
            ]
        )
    else:
        diam = d
    c = np.empty_like(diam, dtype=np.dtype(float))
    c[diam <= 80] = 29174 * np.float_power(diam[diam <= 80], -1.92)
    c[diam > 80] = 156228 * np.float_power(diam[diam > 80], -2.389)
    out = c / (1000 * 1000)
    if isinstance(d, Number):
        return out.item()

    return out


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=200, help='Number of pixels in map')
    parser.add_argument('--res', type=float, default=1, help='Resolution of map pixels (meters per pixel)')
    parser.add_argument('--diam_range', type=float, nargs=2, default=[1, 1000], help='Ranges of diameters of craters to sample')
    args = parser.parse_args()

    # get crater SFD
    side_length = args.dim * args.res
    area = side_length * side_length

    diams = np.arange(args.diam_range[0], args.diam_range[1], dtype=np.float32)
    sfd = viper_sfd(diams)

    sfd_old = np.zeros_like(diams)
    for i in range(len(diams)):
        sfd_old[i] = csfd(diams[i])

    print(sfd[0:10])
    # print(sfd_old[0:10])

    # multiply by area of map
    sfd *= area
    sfd_old *= area
    print(sfd[0:10])
    # print(sfd_old[0:10])

    # display crater SFD
    fig, ax = plt.subplots()
    ax.bar(diams, sfd)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
