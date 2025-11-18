# date:     11-18-2025
# author:   margaret hansen
# purpose:  plot the diffusivity coefficient versus diameter for craters from synthterrain

import numpy as np
import matplotlib.pyplot as plt

from synthterrain.crater.age import equilibrium_age
from synthterrain.crater import determine_production_function
from synthterrain.crater.functions import VIPER_Env_Spec


# kappa diffusivity edited to work with arrays
def kappa_diffusivity(diams):

    k = np.ones_like(diams) * 0.0155
    k[diams > 11.2] = 1.55e-3 * np.power(diams[diams > 11.2], 0.974)
    k[diams >= 45] = 1.23e-3 * np.power(diams[diams >= 45], 0.8386)
    k[diams >= 125] = 5.2e-3 * np.power(diams[diams >= 125], 1.3)

    # if diameter <= 11.2:
    #     k = 0.0155  # m2/myr
    # elif diameter < 45:
    #     k = 1.55e-3 * math.pow(diameter, 0.974)
    # elif diameter < 125:
    #     k = 1.23e-3 * math.pow(diameter, 0.8386)
    # else:  # UNCONSTRAINED BY EQUILIBRIUM ABOVE 125m!!!!!!!
    #     k = 5.2e-3 * math.pow(diameter, 1.3)

    return k  # m2/Myr

# diffusion length scale edited to work with arrays
def diffusion_length_scale(diam, domain_size):
    # return math.pow(diameter * 2 / domain_size, 2) / 4
    return np.power(2 * diam / domain_size, 2) / 4 # resolution squared / 4?



# main program
if __name__ == "__main__":

    # diameters to test (in m)
    d = np.arange(2, 200, 10)

    # call to diffusivity coeff calc
    k = kappa_diffusivity(d)
    ls = diffusion_length_scale(d, 100)

    # number of steps to take
    eq = VIPER_Env_Spec(a=2, b=1000)
    pd = determine_production_function(eq.a, eq.b)
    eq_age = equilibrium_age(d, pd.csfd, eq.csfd, eps=0.001)
    nsteps = (k / 1e6) * eq_age / ls
    
    # old kappa
    kappa0 = 5.5e-6
    kappa_corr = 0.9
    k_old = kappa0 * np.power(d / 1000, kappa_corr)

    # plot
    fig, ax = plt.subplots(2, 2)

    ax[0][0].plot(d, k)
    ax[0][1].plot(d, k_old)
    ax[1][0].plot(d, eq_age / 1e6)
    ax[1][1].plot(d, nsteps)

    ax[0][0].set_title('Kappa Diffusivity (m^2 / Myr)')
    # ax[0][1].set_title('Diffusion Length Scale')
    ax[0][1].set_title('Old Kappa Diffusivity (m^2 / Myr)')
    ax[1][0].set_title('Equilibrium Age (Myr)')
    ax[1][1].set_title('Number of Steps')

    plt.show()

    # components for equilibrium age
    upper_diameters = np.float_power(10, np.log10(d) + 0.001)
    eq = eq.csfd(d) - eq.csfd(upper_diameters)
    pf = pd.csfd(d) - pd.csfd(upper_diameters)
    age = 1e9 * (eq / pf)
    print(age)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(d, eq)
    ax[1].plot(d, pf)
    ax[0].set_title('Equilibrium Count')
    ax[1].set_title('Prod Function Count')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    plt.show()


