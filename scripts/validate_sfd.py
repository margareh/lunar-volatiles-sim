# date: 11-15-2025
# author: margaret hansen
# purpose: plot crater distribution

import argparse
import numpy as np
import matplotlib.pyplot as plt

from numbers import Number
from analyze_crater_list import viper_sfd
from synthterrain.crater.age import equilibrium_age
from synthterrain.crater import functions, determine_production_function
from synthterrain.crater.functions import VIPER_Env_Spec
from synthterrain.crater.diffusion import kappa_diffusivity

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
    # fig, ax = plt.subplots()
    # ax.bar(diams, sfd)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # plt.show()

    # get equilibrium age for each of these craters
    crater_dist = VIPER_Env_Spec(a=args.diam_range[0], b=args.diam_range[1])
    prod_fn = prod_fn = determine_production_function(crater_dist.a, crater_dist.b)
    eq_ages = equilibrium_age(diams, prod_fn.csfd, crater_dist.csfd) * 1e-6

    # alpha = (2 * age - 1) / (age - 1)
    alphas = (2 * eq_ages - 1) / (eq_ages - 1)
    print(eq_ages[0:10])
    print(alphas[0:10]) # these are pretty much 2 for everything (ages are large enough)
    # get larger differences from 2 when scaling ages to Myr
    alphas2 = eq_ages / (eq_ages - 1) # in this instance, will pretty much be 1 for everything

    # # plot these and attempt to compute a power law distribution using them as means
    # # "expected max age"
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(diams, eq_ages)
    # ax[0].set_yscale('log')
    # ax[0].set_xscale('log')
    # ax[1].plot(diams, alphas)
    # plt.show()

    # get equilibrium lifetimes to compare with Fassett et al (2022)
    diams2 = np.array([1, 1.41, 2, 2.83, 4, 5.66, 8, 10, 11.31, 16, 20, 22.63, 32, 40, 45.25, 64, 80, 90.51, 128, 160, 181])
    eq_ages = equilibrium_age(diams2, prod_fn.csfd, crater_dist.csfd) * 1e-6
    print(eq_ages) # Myr

    # diffusivities
    k = kappa_diffusivity(diams2) * 1e6
    print(k) # m^2 / Myr

