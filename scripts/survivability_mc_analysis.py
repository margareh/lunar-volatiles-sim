# date:     10-17-2025
# author:   margaret hansen
# purpose:  run monte carlo simulations to figure out survivability distribution of 
#       craters in the face of degradation and impacts

import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool
from lvsim.sim import LvSim, remove_old_craters
from lvsim.utils import LvSimCfg


# run simulation and compute number of craters destroyed due to diffusion vs impacts
# at each time step
def run_sim(lvsim):

    # run sim through all time steps
    i = 0
    while lvsim.t-lvsim.cfg.args.time_delta >= 0:

        # Take a step
        lvsim.t -= lvsim.cfg.args.time_delta

        # print progress and store initial set of craters
        print("Now on " + str(lvsim.t) + " Ga")
        old_craters_df = copy.copy(lvsim.crater_df)

        # only need to evolve terrain here, don't care about illumination / volatiles
        # so we can skip those steps
        lvsim.evolve_terrain()

        # compute number of craters removed
        new_rows = compute_crater_ages(old_craters_df, lvsim.crater_df)
        print(new_rows)

        # add to database of deleted craters (age, diameter, d/D, type)
        if i > 1:
            age_df = pd.concat([age_df, new_rows])
        else:
            age_df = copy.copy(new_rows)
        
        i += 1

    return age_df, lvsim.crater_df


# compute crater ages for craters removed during one time step
def compute_crater_ages(old_df, new_df):
    # crater dataframes have columns: x, y, diameter, age, d/D

    # filter to craters that were in old but not new
    removed_craters = old_df.copy()[~old_df.index.isin(new_df.index)]
    # print(old_df.index)
    # print(new_df.index)
    # print(removed_craters.index)
    removed_craters_np = np.array([removed_craters.index, removed_craters.age.values, removed_craters.diameter.values, removed_craters.x.values, removed_craters.y.values]).T
    new_craters_np = np.array([new_df.index, new_df.age.values, new_df.diameter.values, new_df.x.values, new_df.y.values]).T

    # determine which new(er) crater removed the older craters
    # and compute age at which crater was removed
    with Pool() as p:
        args = [(i, removed_craters_np, new_craters_np, True) for i in range(removed_craters.shape[0])]
        out = p.map(remove_old_craters, args) # this produces a list of results
        # out is a list of tuples: (index of removed crater, age at removal)

    # create dataframe of new rows of removed craters
    # new_ages = [a for (_,a) in out]
    new_ages = pd.DataFrame(out, columns=['index', 'age_rm'])
    new_ages.set_index('index', inplace=True)
    
    removed_craters.drop(['x', 'y', 'surface', 'new'], inplace=True, axis=1)
    new_ages_all = pd.merge(removed_craters, new_ages, left_index=True, right_index=True)
    
    new_ages_all['type'] = 'IMPACT'
    new_ages_all.loc[new_ages_all['age_rm'] < 0, 'type'] = 'DIFFUSION'
    
    new_ages_all['age_at_rm'] = new_ages_all['age_rm']
    new_ages_all.loc[new_ages_all['age_rm'] < 0, 'age_at_rm'] = new_ages_all['age']
    
    new_ages_all.drop(['age', 'age_rm'], inplace=True, axis=1)
    new_ages_all.rename(columns={'age_at_rm' : 'age'}, inplace=True)
    
    return new_ages_all


# summarize data on removed craters by type of removal (impact or diffusion) and age bin
def get_summary(age_df):

    # add bin column to group by
    age_df['diam_bin'] = '1-2.5'
    age_df.loc[age_df['diameter'] > 2.5, 'diam_bin'] = '2.5-5'
    age_df.loc[age_df['diameter'] > 5,   'diam_bin'] = '5-10'
    age_df.loc[age_df['diameter'] > 10,  'diam_bin'] = '10-25'
    age_df.loc[age_df['diameter'] > 25,  'diam_bin'] = '25-50'
    age_df.loc[age_df['diameter'] > 50,  'diam_bin'] = '50-75'
    age_df.loc[age_df['diameter'] > 75,  'diam_bin'] = '75-100'
    age_df.loc[age_df['diameter'] > 100, 'diam_bin'] = '100-150'
    age_df.loc[age_df['diameter'] > 150, 'diam_bin'] = '150-200'
    age_df.loc[age_df['diameter'] > 200, 'diam_bin'] = '200-250'
    age_df.loc[age_df['diameter'] > 250, 'diam_bin'] = '250-300'
    age_df.loc[age_df['diameter'] > 300, 'diam_bin'] = '300-350'
    age_df.loc[age_df['diameter'] > 350, 'diam_bin'] = '350-400'
    age_df.loc[age_df['diameter'] > 400, 'diam_bin'] = '400-450'
    age_df.loc[age_df['diameter'] > 450, 'diam_bin'] = '450-500'
    age_df.loc[age_df['diameter'] > 500, 'diam_bin'] = '500-600'
    age_df.loc[age_df['diameter'] > 600, 'diam_bin'] = '600-700'
    age_df.loc[age_df['diameter'] > 700, 'diam_bin'] = '700-800'
    age_df.loc[age_df['diameter'] > 800, 'diam_bin'] = '800-900'
    age_df.loc[age_df['diameter'] > 900, 'diam_bin'] = '900-1000'
    age_df.set_index([age_df.index, 'diam_bin', 'type'], inplace=True)

    age_df.drop('diameter', axis=1, inplace=True)

    # make age be myr instead of yr
    age_df['age'] /= 1e6

    # compute aggregate statistics
    summ_df = age_df.groupby(['diam_bin', 'type']).agg(
        count = pd.NamedAgg(column='age', aggfunc='count'),
        avg_age = pd.NamedAgg(column='age', aggfunc='mean'),
        std_age = pd.NamedAgg(column='age', aggfunc='std'),
        avg_dD = pd.NamedAgg(column='d/D', aggfunc='mean'),
        std_dD = pd.NamedAgg(column='d/D', aggfunc='std')
    )
    return summ_df


# main program
if __name__ == "__main__":

    # print(sys.argv)

    # addtl_args = {'test3': 'C',
    #               'test4': 'D'}

    # args_to_parse = copy.copy(sys.argv)
    # for k, v in addtl_args.items():
    #     args_to_parse.append('--'+k)
    #     args_to_parse.append(v)
    # print(args_to_parse)

    # add to arguments to make sure sim has everything it needs
    #addtl_args = {'iters' : '100'}
    addtl_args = {'iters' : '1'}
    cfg = LvSimCfg(addtl_args=addtl_args) # this defines a lot of things for us!
    # cfg.args.use_prod_fn = True # want to make sure we use the production function here
    
    # run simulations
    init_seed = 3567897
    for i in range(cfg.args.iters):
        
        # setup for this iteration
        print("Now on " + str(i+1) + " / " + str(cfg.args.iters))
        cfg.args.seed = init_seed + i
        lvsim = LvSim(cfg)

        # run simulation and get results of survivability
        age_df, crater_df = run_sim(lvsim)
        # print(len(age_df))

        # plot the distribution of craters that were removed vs remaining by age
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
        ax[0].hist(age_df['age'] / 1e6, bins=30, color='tab:blue')
        ax[1].hist(crater_df['age'] / 1e6, bins=30, color='tab:orange')
        ax[0].set_title("Removed Craters by Age (Myr)")
        ax[1].set_title("Remaining Craters by Age (Myr)")
        plt.savefig(os.path.join(cfg.args.outpath, 'crater_age_dist_'+str(i)+'.png'), dpi=100, bbox_inches='tight')
        plt.close()

        # # bin by diameter and compute avg and std dev of ages
        # new_age_rows = get_summary(age_df)
        # print(new_age_rows)
        # new_age_rows['iter'] = i
        # new_age_rows.set_index([new_age_rows.index, 'iter'], inplace=True)

        # # append to overall dataset
        # if i > 0:
        #     age_summ_df = pd.concat([age_summ_df, new_age_rows])
        # else:
        #     age_summ_df = copy.copy(new_age_rows)
        # # print(len(age_summ_df))

        # save the results
        # age_summ_df.to_csv(os.path.join(cfg.args.outpath, 'mc_crater_ages.csv'))

