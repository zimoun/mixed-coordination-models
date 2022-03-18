from tqdm.auto import trange
import matplotlib.pyplot as plt
# MODIF
#from definitions import ROOT_FOLDER
import os
import pandas as pd
from hippocampus.agents import CombinedAgent
from hippocampus.environments import HexWaterMaze
import numpy as np
import seaborn as sns
from datetime import datetime
import argparse
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.image as mpimg

import warnings
warnings.filterwarnings('ignore')

def launch_pearce(path, n_agents=100, lesion_hippocampus=False, plot=False):
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gamma', type=float)
    # parser.add_argument('--inv_temp', type=float)
    # parser.add_argument('--n_agents')
    # parser.add_argument('--lesion_hippocampus')
    # parser.add_argument('--learning_rate', type=float)
    #
    # args = parser.parse_args()

    # set parameters
    gamma = .95
    inv_temp = 16.
    #lesion_hippocampus = False
    learning_rate = .07
    # if args.gamma is None:
    #     gamma = .99
    # else:
    #     gamma = args.gamma
    # if args.n_agents is None:
    #     n_agents = 100
    # else:
    #     n_agents = args.n_agents
    # if args.inv_temp is None:
    #     inv_temp = 5.
    # else:
    #     inv_temp = args.inv_temp
    # if args.lesion_hippocampus is None:
    #     lesion_hippocampus = False
    # else:
    #     lesion_hippocampus = True
    # if args.learning_rate is None:
    #     learning_rate = .07
    # else:
    #     learning_rate = args.learning_rate

    lesion_striatum = False

    # tqdm.write('Running {} agents with lr {}, gamma {}, inv temp {} and HPC lesioned {}'.format(n_agents,
    #                                                                                             learning_rate,
    #                                                                                             gamma,
    #                                                                                             inv_temp,
    #                                                                                             lesion_hippocampus))

    if lesion_hippocampus and not lesion_striatum:
        group = 'lesion'
    elif not lesion_hippocampus and not lesion_striatum:
        group = 'control'
    elif lesion_striatum and not lesion_hippocampus:
        group = 'lesion_DLS'
    else:
        group = 'other'

    # save location
    #MODIF
    #results_folder = os.path.join(ROOT_FOLDER, 'results', 'pearce', group, str(datetime.now()))
    results_folder = os.path.join('results', path, group)
    figure_folder = os.path.join(results_folder, 'figures')
    print("Figure folder path: ", figure_folder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    params = pd.DataFrame({'n_agents': [n_agents],
                           'inv_temp': [inv_temp],
                           'gamma': [gamma],
                           'lesion HPC': [lesion_hippocampus],
                           'lesion DLS': [lesion_striatum]})
    params.to_csv(os.path.join(results_folder, 'params.csv'))

    # initialise environment
    g = HexWaterMaze(10)

    # determine platform sequence
    possible_platform_states = np.array([192, 185, 181, 174, 216, 210, 203, 197])  # for the r = 10 case


    def determine_platform_seq(platform_states):
        indices = np.arange(len(platform_states))
        usage = np.zeros(len(platform_states))

        plat_seq = [np.random.choice(platform_states)]
        for sess in range(1, 11):
            distances = np.array([g.grid.distance(plat_seq[sess - 1], s) for s in platform_states])
            candidates = indices[np.logical_and(usage < 2, distances > g.grid.radius)]
            platform_idx = np.random.choice(candidates)
            plat_seq.append(platform_states[platform_idx])
            usage[platform_idx] += 1.

        return plat_seq


    for n_agent in trange(n_agents, position=0,leave=False,desc='agents: '):
        # set random seed
        np.random.seed(n_agent)

        # determine sequence of platform locations
        platform_sequence = determine_platform_seq(possible_platform_states)

        # intialise agent
        agent = CombinedAgent(g, init_sr='rw',
                              lesion_dls=lesion_striatum,
                              lesion_hpc=lesion_hippocampus,
                              inv_temp=inv_temp,
                              gamma=gamma,
                              learning_rate=learning_rate)
        agent_results = []
        agent_ets = []
        session = 0

        total_trial_count = 0
        if not os.path.exists(os.path.join(figure_folder, 'agent{}.png'.format(n_agent))):
            for ses in trange(11, position=1,leave=False, desc='sessions: '):
                for trial in trange(4, position=2, leave=False, desc='trials: '):

                    # every first trial of a session, change the platform location
                    if trial == 0:
                        g.set_platform_state(platform_sequence[ses])

                    res = agent.one_episode(random_policy=False)
                    res['trial'] = trial
                    res['escape time'] = res.time.max()
                    res['session'] = ses
                    res['total trial'] = total_trial_count
                    agent_results.append(res)
                    agent_ets.append(res.time.max())

                    total_trial_count += 1

            agent_df = pd.concat(agent_results)
            agent_df['total time'] = np.arange(len(agent_df))
            agent_df['agent'] = n_agent

            agent_df.to_csv(os.path.join(results_folder, 'agent{}.csv'.format(n_agent)))

            # plot and save a prelim figure
            first_and_last = agent_df[np.logical_or(agent_df.trial == 0, agent_df.trial == 3)]

            fig = plt.figure()
            ax = sns.lineplot(data=first_and_last, x='session', y='escape time', hue='trial')
            plt.title('Agent n {}'.format(n_agent))
            plt.savefig(os.path.join(figure_folder, 'agent{}.png'.format(n_agent)))
            plt.close()

    # plot averages

    all_data = []
    for ag in trange(n_agents, desc='loading data...'):
        df = pd.read_csv(os.path.join(results_folder, 'agent{}.csv'.format(ag)))
        summary = df.pivot_table(index=['agent', 'session', 'trial'], aggfunc='mean')
        all_data.append(summary)

    df = pd.concat(all_data)
    df['platform location'] = df['platform'].astype('category')
    df.to_csv(os.path.join(results_folder, 'summary.csv'))

    # Plot the average escape time per platform
    plt.figure()
    sns.barplot(data=df.loc[(list(range(n_agents)), list(range(11)), 0)], x='platform location', y='escape time')
    plt.savefig(os.path.join(figure_folder, 'et_per_platform.png'))
    plt.close()

    # plot the escape time per session for trials 1 and 4
    plt.figure()
    first_last = df.loc[(list(range(n_agents)), list(range(11)), (0, 3))]
    sns.lineplot(data=first_last.reset_index(), x='session', y='escape time', hue='trial', ci=None)
    plt.savefig(os.path.join(figure_folder, 'escape_time_firstlast.png'))
    plt.close()

    if plot:
        plt.figure()
        first_last = df.loc[(list(range(n_agents)), list(range(11)), (0, 3))]
        if lesion_hippocampus:
            sns.lineplot(data=first_last.reset_index(), x='session', y='escape time', hue='trial', ci=None).set_title("lesioned HPC group")
        else:
            sns.lineplot(data=first_last.reset_index(), x='session', y='escape time', hue='trial', ci=None).set_title("control group")
        plt.show()
        plt.close()

    # plot the escape time per session for all trials
    plt.figure()
    sns.lineplot(data=df.reset_index(), x='session', y='escape time', hue='trial', ci=None)
    plt.savefig(os.path.join(figure_folder, 'escape_time.png'))
    plt.close()

    # plot the escape time per trial with vlines indicating new sessions
    plt.figure()
    sns.lineplot(data=df, x='total trial', y='escape time')
    for i in range(44):
        if (i % 4) == 0:
            plt.axvline(x=i, ymin=0, ymax=1, linewidth=1, color='r', alpha=.3)
    plt.savefig(os.path.join(figure_folder, 'escape_time_pertrial.png'))
    plt.close()


def fuse_plot_and_save(path):

    control_df = pd.read_csv('results/'+path+'/control/summary.csv')
    lesioned_df = pd.read_csv('results/'+path+'/lesion/summary.csv')

    def plot_fusioned_escape_time():
        df_trial1 = control_df[control_df["trial"]==0]
        df_trial4 = control_df[control_df["trial"]==3]
        df_lesion_trial1 = lesioned_df[lesioned_df["trial"]==0]
        df_lesion_trial4 = lesioned_df[lesioned_df["trial"]==3]

        fig, axs = plt.subplots(1, 3, figsize=(18,5))
        fig.suptitle("Escape time as a function of sessions and trials", fontsize=20, y = 1.1)

        # first, plots the original results of Pearce and its replication by Geerts 2020
        axs[0].imshow(mpimg.imread("../images/results_pearce.jpg"),aspect="auto")
        axs[1].imshow(mpimg.imread("../images/geerts_exp_1.jpg"),aspect="auto")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[0].set_title("Pearce results")
        axs[1].set_title("Geerts results")
        axs[0].set_frame_on(False)
        axs[1].set_frame_on(False)

        # then plot our results
        pal = sns.color_palette()
        # HPC lesion - trial 1
        sns.lineplot(data=df_lesion_trial1, x=df_lesion_trial1['session']+1, y='escape time', label="HPC lesion - trial 1", linewidth = 5, color=pal[1], ci=None)
        # HPC lesion - trial 4
        sns.lineplot(data=df_lesion_trial4, x=df_lesion_trial4['session']+1, y='escape time',  label="HPC lesion - trial 4", linewidth = 5, color=pal[1],  style=False, dashes=[(2,1)], ci=None)
        # Control - trial 1
        sns.lineplot(data=df_trial1, x=df_trial1['session']+1, y='escape time', label="Control - trial 1", linewidth = 5, color=pal[0], ci=None)
        # Control - trial 1
        sns.lineplot(data=df_trial4, x=df_trial4['session']+1, y='escape time', label="Control - trial 4", linewidth = 5,  color=pal[0], style=False, dashes=[(2,1)], ci=None)

        axs[2].set_title("Our results")
        axs[2].set(ylabel="Escape time (steps)")
        axs[2].set(xlabel="Session")
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['top'].set_visible(False)
        handles, labels = axs[2].get_legend_handles_labels()
        axs[2].legend(handles=[handles[0],handles[1],handles[3],handles[4]], labels=[labels[0],labels[1],labels[3],labels[4]])
        axs[2].plot(aspect="auto")
        axs[2].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    plt.figure()
    plot_fusioned_escape_time()
    plt.savefig(os.path.join("results/"+path+"/", 'fused_escape_time.png'))
    plt.close()

    plt.figure()
    plot_fusioned_escape_time()
    plt.show()
    plt.close()
