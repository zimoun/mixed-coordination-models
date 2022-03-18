from tqdm import tqdm
import matplotlib.pyplot as plt
from definitions import ROOT_FOLDER
import os
import pandas as pd
from hippocampus.agents import CombinedAgent
from hippocampus.environments import HexWaterMaze
import numpy as np
import seaborn as sns
from datetime import datetime
import argparse

np.random.seed(0)

# save location
results_folder = os.path.join(ROOT_FOLDER, 'results', 'cue_vs_place_watermaze')
figure_folder = os.path.join(results_folder, 'figures')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    os.makedirs(figure_folder)


g = HexWaterMaze(6)
g.starting_state = 42
g.current_state = 42

possible_platform_states = np.array([51, 57])

for group in ['dls', 'hpc', 'sham', 'double']:

    tqdm.write('\nRunning {} lesioned group... \n'.format(group))

    if group == 'dls':
        l_dls = True
        l_hpc = False
    elif group == 'hpc':
        l_dls = False
        l_hpc = True
    elif group == 'sham':
        l_dls = False
        l_hpc = False
    elif group == 'double':
        l_dls = True
        l_hpc = True

    for agent_n in tqdm(range(15)):

        g.other_terminals = []
        filename = os.path.join(results_folder, '{}_agent{}.csv'.format(group,  agent_n))
        if os.path.exists(filename):
            continue

        platform_sequence = possible_platform_states
        np.random.shuffle(platform_sequence)

        agent = CombinedAgent(g, init_sr='rw',
                              lesion_dls=l_dls,
                              lesion_hpc=l_hpc,
                              inv_temp=10.,
                              gamma=.99,
                              learning_rate=.01,
                              eta=.03)
        agent_results = []
        agent_ets = []
        session = 0

        total_trial_count = 0
        n_training_trials = 15

        for trial in tqdm(range(n_training_trials), leave=False):
            # every first trial of a session, change the platform location
            if trial == 0:
                g.set_platform_state(platform_sequence[0])

            res = agent.one_episode(random_policy=False, random_start_loc=False)
            res['trial'] = trial
            res['escape time'] = res.time.max()
            res['total trial'] = total_trial_count
            res['trial type'] = 'train'
            agent_results.append(res)
            agent_ets.append(res.time.max())

            total_trial_count += 1

        # probe trial
        g.set_platform_state(platform_sequence[1])  # set new platform state
        g.add_terminal(platform_sequence[0])  # set previous platform to also be terminal
        res = agent.one_episode(random_policy=False, random_start_loc=False)
        res['trial'] = trial
        res['escape time'] = res.time.max()
        res['total trial'] = total_trial_count
        res['trial type'] = 'probe'
        agent_results.append(res)
        agent_ets.append(res.time.max())


        agent_df = pd.concat(agent_results)
        agent_df['total time'] = np.arange(len(agent_df))
        agent_df['agent'] = 0

        agent_df.to_csv(filename)


