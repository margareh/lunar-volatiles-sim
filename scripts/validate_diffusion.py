# date: 11-12-2025
# author: margaret hansen
# purpose: validate diffusion model against Farrell et al results

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lvsim.crater import profile, stopar_fresh_dd
from diffusion.diffusion import diffusion_cuda

# diameters in m to generate diffusion profiles for
D = [300, 1000, 3000]

# time step and length in Gyr
TSTEP = 0.5
TSTART = 0
TEND = 3

# domain size for crater surfaces
DOMSIZE = 101
HALFSIZE = int(DOMSIZE / 2)
print(HALFSIZE)

# main function
if __name__ == "__main__":

    # create crater dataframe with craters of known diameter
    A = np.arange(TSTART, TEND+TSTEP, TSTEP) * 1e9
    ages, diams = np.meshgrid(A, D)
    nd = len(D)
    na = len(A)
    np_df = np.dstack((ages, diams)).reshape((na*nd, 2))

    crater_df = pd.DataFrame(data=np_df, columns=['age', 'diameter'])
    crater_df['d/D'] = stopar_fresh_dd(crater_df['diameter'])
    # print(crater_df)

    # generate crater profiles
    crater_df['surface'] = crater_df.apply(lambda row: profile(row["d/D"], row["diameter"], D=DOMSIZE), axis=1)

    # run diffusion for set amounts of time
    surfs_np = np.array(crater_df["surface"].tolist())
    new_ratios, new_surfs = diffusion_cuda(crater_df["diameter"], crater_df["d/D"], crater_df["age"], surfs_np, D=DOMSIZE)

    # save numpy file
    np.savez('../output/diffusion_profile_out.npz',new_surfs=new_surfs)

    # generate new crater profiles
    crater_df['d/D'] = new_ratios
    crater_df["surface"] = [s for s in new_surfs[:,...]]

    craters_300m = crater_df[crater_df.diameter == 300]
    craters_1km = crater_df[crater_df.diameter == 1000]
    craters_3km = crater_df[crater_df.diameter == 3000]
    # print(len(craters_300m))
    # print(len(craters_1km))
    # print(len(craters_3km))

    # plot the profiles against what's expected
    fig, ax = plt.subplots(1,3, figsize=(30, 10))

    ax[0].set_title('D = 300 m')
    ax[1].set_title('D = 1 km')
    ax[2].set_title('D = 3 km')

    for i in range(3):
        ax[i].set_xlabel('Distance (m)')
        ax[i].set_ylabel('Elevation (m)')
        
        if i == 0:
            craters = copy.copy(craters_300m)
        elif i == 1:
            craters = copy.copy(craters_1km)
        else:
            craters = copy.copy(craters_3km)
        # print(len(craters))

        diam = D[i]
        dists = np.arange(0, DOMSIZE) * (2 * diam / (DOMSIZE))

        for j in range(len(A)):
            curr_crater = craters[craters.age == A[j]]
            surf_np = curr_crater.surface.to_numpy()[0]
            surf_profile = surf_np[HALFSIZE-1, :]
            ax[i].plot(dists, surf_profile)

    plt.savefig('../output/diffusion_profiles.png', dpi=100, bbox_inches='tight')
    plt.close()
