from agents.dolle_agent import DolleAgent
from agents.fusion_agent import CombinedAgent
from environments.HexWaterMaze import HexWaterMaze
from utils import create_path, get_mean_preferred_dirs, plot_mean_arrows, create_df, get_MSLE, charge_agents, save_params_in_txt

from IPython.display import clear_output
from statsmodels.formula.api import ols
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import random
import pickle
import shutil
import time
import os


def perform_group_pearce(env_params, ag_params, show_plots = True, save_plots = True, save_agents=True, directory = None, verbose = True):
    """
    Run multiple simulations of the main experiment of Pearce 1998, a Morris water-maze derived task where a rat has
    to navigate through a circular maze filled with water, to find a submerged platform indicated by a visual landmark
    hovering at proximity of the platform.
    In this experiment there are 11 sessions of 4 trials. The platform and visual landmark move at each new session.
    The rat might be released from 4 different locations at the edge of the pool, with each release point being selected once
    at each session, in a random order. The platform might be positionned at 8 different points in the maze.
    See Pearce 1998 and our code for more informations about the original and replicated protocol.
    In the original experiment, a significant decrease of the escape time with both sessions and trials was found.
    In this work we want to check whether Dolle's and Geerts's coordination models are able to replicate this result.

    :param env_params: Contains all the parameters to set the water-maze RL environment (see EnvironmentParams for details)
    :type env_params: EnvironmentParams
    :param ag_params: Contains all the parameters to set the different RL modules of the agent (see AgentsParams for details)
    :type ag_params: AgentsParams
    :param show_plots: whether to display any created plot at the end of the simulations
    :type show_plot: boolean
    :param save_plots: whether to save any created plots in the results folder at the end of the simulations
    :type save_plots: boolean
    :param save_agents: whether to save Agents object in the result folder (take a lot of memory)
    :type save_agents: boolean
    :param directory: optional directory where to store all the results (used for grid-search)
    :type directory: str
    :param verbose: whether to print the progress of the simulations
    :type verbose: boolean
    """

    # create environment
    possible_platform_states, envi = get_maze_pearce(env_params.maze_size, env_params.landmark_dist, env_params.starting_states)

    # get results directory path
    results_folder = create_path_main_pearce(env_params, ag_params, directory=directory)

    saved_results_folder = "../saved_results/"+results_folder # never erased
    results_folder = "../results/"+results_folder # erased if an identical simulation is run

    if not os.path.isdir(saved_results_folder):
        if os.path.isdir(results_folder): # delete previous identical simulation data (as it wasn't saved)
            shutil.rmtree(results_folder)
        # creates results folders
        figure_folder = os.path.join(results_folder, 'figs')
        os.makedirs(results_folder)
        os.makedirs(figure_folder)

        save_params_in_txt(results_folder, env_params, ag_params)

        agents = []
        for n_agent in range(env_params.n_agents):

            # np.random.seed(n_agent) # uncomment to make every simulation identical

            # determine sequence of platform locations
            platform_sequence = determine_platform_seq(envi, possible_platform_states, env_params.n_sessions)

            # intialise agent (either Geerts' model or Dolle's)
            if not ag_params.dolle:
                agent = CombinedAgent(envi, ag_params, init_sr=env_params.init_sr)
            else:
                agent = DolleAgent(envi, ag_params, init_sr=env_params.init_sr)

            total_trial_count = 0
            agent_df = pd.DataFrame() # to create a log file keeping track of the agents performances

            for ses in range(env_params.n_sessions):
                for trial in range(env_params.n_trials):

                    if verbose:
                        print("agent: "+str(n_agent)+", session: "+str(ses)+", trial: "+str(trial)+"                              ", end="\r")

                    # every first trial of a session, change the platform and landmark locations
                #     if trial == 0:
                # envi.set_platform_state(platform_sequence[ses])
                # envi.set_proximal_landmark() # put a landmark at the predefined distance of the platform
                # envi.set_distal_landmark2()
                    if ses % 2 == 0:
                        envi.set_platform_state(platform_sequence[ses])
                        envi.set_proximal_landmark() # put a landmark at the predefined distance of the platform
                        envi.set_distal_landmark3()


                    if ses % 2 == 1:
                        envi.set_platform_state(random.choice(possible_platform_states))
                        envi.set_proximal_landmark() # put a landmark at the predefined distance of the platform
                        envi.set_distal_landmark2()


                    # simulate one episode, res is a dataframe keeping track of agent's and environment's variables at each timesteps
                    res = envi.one_episode(agent, env_params.time_limit)

                    # add infos for each trial
                    res['trial'] = trial
                    res['escape time'] = res.time.max()
                    res['session'] = ses
                    res['total trial'] = total_trial_count

                    total_trial_count += 1
                    agent_df = agent_df.append(res, ignore_index=True)

            if n_agent < 100:
                envi.set_platform_state(126)
                envi.set_proximal_landmark() # put a landmark at the predefined distance of the platform
                envi.set_distal_landmark2()
                # simulate one episode, res is a dataframe keeping track of agent's and environment's variables at each timesteps
                res = envi.one_episode(agent, env_params.time_limit)
            else:
                envi.set_platform_state(126)
                envi.set_proximal_landmark() # put a landmark at the predefined distance of the platform
                envi.set_distal_landmark3()
                # simulate one episode, res is a dataframe keeping track of agent's and environment's variables at each timesteps
                res = envi.one_episode(agent, env_params.time_limit)

            res['trial'] = 0
            res['escape time'] = res.time.max()
            res['session'] = 12
            res['total trial'] = total_trial_count

            total_trial_count += 1
            agent_df = agent_df.append(res, ignore_index=True)


            # add infos for each simulation
            agent_df['total time'] = np.arange(len(agent_df))
            agent_df['agent'] = n_agent

            agent_df.to_csv(os.path.join(results_folder, 'agent{}.csv'.format(n_agent)))
            agents.append(agent)

            # show single agent performances
            if show_plots or save_plots:
                create_single_agent_plot(results_folder, agent, agent_df, n_agent, save_plots)

            clear_output() # erase standard output

        # create plots showing mean performances of all the agents
        if show_plots or save_plots:
            plot_main_pearce_perfs(results_folder, env_params.n_trials, env_params.n_agents, env_params.n_sessions, show_plots, save_plots)
            plot_main_pearce_quivs(results_folder, agents, show_plots, save_plots) # show preferred heading-vectors of each strategy

        # take a lot of memory
        if save_agents:
            file_to_store = open(results_folder+"/agents.p", "wb")
            pickle.dump(agents, file_to_store)
            file_to_store.close()

    # if an identical simulation has already been saved
    else:
        if show_plots or save_plots:
            plot_main_pearce_perfs(saved_results_folder, env_params.n_trials, env_params.n_agents, env_params.n_sessions, show_plots, save_plots)
            agents = charge_agents(saved_results_folder+"/agents.p")
            plot_main_pearce_quivs(saved_results_folder, agents, show_plots, save_plots) # show heading-vectors for each state in the water-maze

    # delete all agents, to prevent memory error
    if 'agents' in locals():
        for agent in agents:
            del agent
        del agents


def plot_main_pearce_perfs(results_folder, n_trials, n_agents, n_sessions, show_plots, save_plots):
    """
    Plot four figures showing the mean performances of the agents which data is saved at path results_folder.
    Figure 1: Escape time per session for all trials
    Figure 2: Escape time per trial with vlines indicating new sessions
    Figure 3: Arbitrator's choice evolution across session
    Figure 4: Plot the average escape time per platform

    :param results_folder: path of the results folder
    :type results_folder: str
    :param n_trials: number of trials to run
    :type n_trials: int
    :param n_agents: number of simulations (agents) stored in results_folder
    :type n_agents: int
    :param n_sessions: number of sessions to run
    :type n_sessions: int
    :param show_plots: whether to display any created plot
    :type show_plot: boolean
    :param save_plots: whether to save any created plots in the results folder
    :type save_plots: boolean
    """

    print("Computing the mean performances of the agents...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17,11))
    fig.suptitle("Mean performances of the agents", fontsize = 14)

    figure_folder = os.path.join(results_folder, 'figs')
    df = create_df(results_folder, n_agents, grouped=True)

    # 200 rows, 100*trial 1, 100*trial 4
    df_anova_trial = df.reset_index()
    df_anova_trial = df_anova_trial[np.logical_or(df_anova_trial["trial"]==0, df_anova_trial["trial"]==3)]
    df_anova_trial = df_anova_trial.pivot_table(index=['agent', 'trial'], aggfunc='mean')
    df_anova_trial = df_anova_trial[["escape time"]]

    # 200 rows, 100*session 1, 100*session 4
    df_anova_session = df.reset_index()
    df_anova_session = df_anova_session[np.logical_or(df_anova_session["session"]==0, df_anova_session["session"]==10)]
    df_anova_session = df_anova_session.pivot_table(index=['agent','session'], aggfunc='mean')
    df_anova_session = df_anova_session[["escape time"]]

    # perform ANOVA (IV -> trial, DV -> escape time)
    try:
        df_anova_trial["escape_time"] = df_anova_trial["escape time"]
        model = ols('escape_time ~ C(trial) + C(trial)', data=df_anova_trial.reset_index().sample(8)).fit()
        print()
        print("Computing ANOVA on trial...")
        print(sm.stats.anova_lm(model, typ=2))
        print()
        f = open(results_folder+"/anova_results.txt",'w')
        print(sm.stats.anova_lm(model, typ=2), file=f)
        print("\n(IV -> trial and session, DV -> escape time)", file=f)
        f.close()
    except:
        print("Anova failed (there might not be enough data)")
        print()
        f = open(results_folder+"/anova_results.txt",'w')
        print("Anova failed (there might not be enough data)", file=f)
        f.close()

    # perform ANOVA (IV -> session, DV -> escape time)
    try:
        df_anova_session["escape_time"] = df_anova_session["escape time"]
        model = ols('escape_time ~  C(session) + C(session)', data=df_anova_session.reset_index().sample(8)).fit()
        print()
        print("Computing ANOVA on session...")
        print(sm.stats.anova_lm(model, typ=2))
        print()
        f = open(results_folder+"/anova_results.txt",'w')
        print(sm.stats.anova_lm(model, typ=2), file=f)
        print("\n(IV -> trial and session, DV -> escape time)", file=f)
        f.close()
    except:
        print("Anova failed (there might not be enough data)")
        print()
        f = open(results_folder+"/anova_results.txt",'w')
        print("Anova failed (there might not be enough data)", file=f)
        f.close()

    ###############PLOTS#################

    # Figure 1: Escape time per session for all trials
    plt.figure()
    sns.lineplot(data=df.reset_index(), x=df.reset_index()['session']+1, y='escape time', hue='trial', ci=None, ax=ax1).set_title("Escape time as a function of sessions and trials")
    ax1.set(ylabel="Escape time (steps number)")
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    plt.close()


    # Figure 2: Escape time per trial with vlines indicating new sessions
    plt.figure()
    sns.lineplot(data=df, x='total trial', y='escape time',ax=ax2).set_title("Escape time per trial")
    for i in range(n_sessions*n_trials):
        if (i % n_trials) == 0:
            ax2.axvline(x=i, ymin=0, ymax=1, linewidth=1, color='r', alpha=.3)
    ax2.set(ylabel="Escape time (steps number)")
    plt.close()

    # Figure 3: Arbitrator choice evolution across session
    # try with dolle arbitrator, then geerts arbitrator if it fail
    try: # Dolle arbitrator
        plt.figure()
        sns.lineplot(data=df.reset_index(), x=df.reset_index()['session']+1, y='arbitrator_choice', hue='trial', ci=None, ax=ax3).set_title("Evolution of P(Goal-Directed) across sessions and trials")
        ax3.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.close()

    except Exception: # Geerts arbitrator
        plt.figure()
        sns.lineplot(data=df.reset_index(), x=df.reset_index()['session']+1, y='P(SR)', hue='trial', ci=None, ax=ax3).set_title("Evolution of P(Goal-Directed) across sessions and trials")
        ax3.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.close()

    # Figure 4: Plot the average escape time per platform
    plt.figure()
    sns.barplot(data=df.loc[(list(range(n_agents)), list(range(n_sessions)), 0)], x='platform location', y='escape time', ax=ax4).set_title("Escape time per platform")
    ax4.set(ylabel="Escape time (steps number)")

    fig.savefig(os.path.join(figure_folder, 'mean_performances.png'))
    plt.close()


def plot_main_pearce_quivs(results_folder, agents, show_plots, save_plots):
    """
    Plot eight quivers of the heading vectors of the navigation strategies of a group of agents.
    Each quiver display 270 arrows showing the preferred direction of a given strategy at each state of the maze
    Plot a first set of quivers with platform at state 18 and 4 trials of additional training:
    One quiver for egocentric MF strategy
    One quiver for allocentric MF strategy
    One quiver for goal-directed strategy
    One quiver for the coordination model strategy
    Then a second set of four quivers with platform at state 48 and no additional training.
    This is to see the effect of training and previous platform state on heading-vectors

    :param results_folder: path of the results folder
    :type results_folder: str
    :param agents: all the agents that were created to produce data stored in results_folder
    :type agents: Agent list
    :param show_plots: whether to display any created plot
    :type show_plot: boolean
    :param save_plots: whether to save any created plots in the results folder
    :type save_plots: boolean
    """

    print("Computing the heading-vectors of each strategy...")
    print()



    figure_folder = os.path.join(results_folder, 'figs')


    for  i in agents:
        i.env.set_distal_landmark2()
    # plot a first set of quivers, showing the heading-vectors after a 4 trials training
    if str(type(agents[0])) != "<class 'agents.dolle_agent.DolleAgent'>" :
        hv_mf, hv_allo, hv_sr, hv_combined,_,_,_,_,_ = get_mean_preferred_dirs(agents, platform_idx=126, nb_trials=4)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, hv_mf, hv_allo, hv_sr, hv_combined, nb_trials=4)
    else:
        hv_mf, hv_allo, hv_sr, hv_combined, decisions_arbi = get_mean_preferred_dirs(agents, platform_idx=126, nb_trials=4)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, hv_mf, hv_allo, hv_sr, hv_combined, nb_trials=4, decisions_arbi=decisions_arbi)
    fig.suptitle("Mean strategies after 11 sessions of training + 4 additional trials with platform on state 126", fontsize=14)
    if save_plots:
        plt.savefig(os.path.join(figure_folder, 'mean_heading_vectors_4trainingtrials.png'))
    if show_plots:
        plt.show()

    for  i in agents:
        i.env.set_distal_landmark3()
    # plot a first set of quivers, showing the heading-vectors after a 4 trials training
    if str(type(agents[0])) != "<class 'agents.dolle_agent.DolleAgent'>" :
        hv_mf, hv_allo, hv_sr, hv_combined,_,_,_,_,_ = get_mean_preferred_dirs(agents, platform_idx=126, nb_trials=0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, hv_mf, hv_allo, hv_sr, hv_combined, nb_trials=0)
    else:
        hv_mf, hv_allo, hv_sr, hv_combined, decisions_arbi = get_mean_preferred_dirs(agents, platform_idx=126, nb_trials=0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, hv_mf, hv_allo, hv_sr, hv_combined, nb_trials=0, decisions_arbi=decisions_arbi)
    fig.suptitle("Mean strategies after 11 sessions of training + 4 additional trials with platform on state 126", fontsize=14)
    if save_plots:
        plt.savefig(os.path.join(figure_folder, 'mean_heading_vectors_4trainingtrials.png'))
    if show_plots:
        plt.show()

    for  i in agents:
        i.env.set_distal_landmark2()

    # plot a second set of quivers, showing the heading-vectors after a platform location shift (state 18 to 48) and no training
    if str(type(agents[0])) != "<class 'agents.dolle_agent.DolleAgent'>" :
        hv_mf, hv_allo, hv_sr, hv_combined,_,_,_,_,_ = get_mean_preferred_dirs(agents, platform_idx=48, nb_trials=0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, hv_mf, hv_allo, hv_sr, hv_combined, nb_trials=0)
    else:
        hv_mf, hv_allo, hv_sr, hv_combined, decisions_arbi = get_mean_preferred_dirs(agents, platform_idx=48, nb_trials=0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, hv_mf, hv_allo, hv_sr, hv_combined, nb_trials=0, decisions_arbi=decisions_arbi)
    fig.suptitle("Mean strategies after 11 sessions of training + 4 additional trials with platform on state 18 \n and final platform switch to state 48 with no further training", fontsize=14)
    if save_plots:
        plt.savefig(os.path.join(figure_folder, 'mean_heading_vectors_0trainingtrials.png'))
    if show_plots:
        plt.show()

    for i in agents:
        i.env.set_distal_landmark3()

    # plot a second set of quivers, showing the heading-vectors after a platform location shift (state 18 to 48) and no training
    if str(type(agents[0])) != "<class 'agents.dolle_agent.DolleAgent'>" :
        hv_mf, hv_allo, hv_sr, hv_combined,_,_,_,_,_ = get_mean_preferred_dirs(agents, platform_idx=48, nb_trials=0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, hv_mf, hv_allo, hv_sr, hv_combined, nb_trials=0)
    else:
        hv_mf, hv_allo, hv_sr, hv_combined, decisions_arbi = get_mean_preferred_dirs(agents, platform_idx=48, nb_trials=0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, hv_mf, hv_allo, hv_sr, hv_combined, nb_trials=0, decisions_arbi=decisions_arbi)
    fig.suptitle("Mean strategies after 11 sessions of training + 4 additional trials with platform on state 18 \n and final platform switch to state 48 with no further training", fontsize=14)
    if save_plots:
        plt.savefig(os.path.join(figure_folder, 'mean_heading_vectors_0trainingtrials.png'))
    if show_plots:
        plt.show()

    plt.close()


def create_single_agent_plot(results_folder, agent, agent_df, n_agent, save_plots):
    """
    Saves figures showing the performances of a single agent

    :param results_folder: path of the results folder to store the plots
    :type results_folder: str
    :param agent: the Agent object
    :type agent: Agent
    :param agent_df: Dataframe containing agent's and environment's variable
    :type agent_df: pandas dataframe
    :param n_agent: the id of the agent
    :type n_agent: int
    :param save_plots: whether to save any created plots in the results folder at the end of the simulations
    :type save_plots: boolean
    """

    figure_folder = os.path.join(results_folder, 'figs')
    first_and_last = agent_df[np.logical_or(agent_df.trial == 0, agent_df.trial == 3)]

    # save a plot of the evolution of escape time across sessions (two lines, for first and last trials)
    fig = plt.figure()
    ax = sns.lineplot(data=first_and_last, x='session', y='escape time', hue='trial')
    plt.title('Agent n {}'.format(n_agent))
    plt.savefig(os.path.join(figure_folder, 'escape_time_agent{}.png'.format(n_agent)))
    plt.close()

    # saves quivers showing the heading-vectors of the preferred direction of the agent at each state of the maze
    # four quivers / four strategies : egocentric MF, allocentric MF, Goal-Directed, coordination model
    if str(type(agent)) != "<class 'agents.dolle_agent.DolleAgent'>" :
        res2_mf, res2_allo, res2_sr, res2_combined, _,_,_,_,_ = get_mean_preferred_dirs([agent])
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows([agent], res2_mf, res2_allo, res2_sr, res2_combined)
    else:
        res2_mf, res2_allo, res2_sr, res2_combined, decisions_arbi = get_mean_preferred_dirs([agent])
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows([agent], res2_mf, res2_allo, res2_sr, res2_combined, decisions_arbi=decisions_arbi)
    plt.savefig(os.path.join(figure_folder, 'heading_vectors_agent{}.png'.format(n_agent)))
    plt.close()


def plot_pearce(env_params, ag_params, ci=None, experimental_data=None, show_plots=True, save_plots=False, directory=None):
    """
    Plot the evolution of the escape time across sessions, for both the control and HPC-lesiond group.
    For each simulated group, plot a line for the first trial and another for the last.
    Option to show the confidence interval (see param ci)
    Display the summed Mean Square Error of the four lines relative to the original data from Pearce

    :param env_params: contains all the environment's parameters, (see EnvironmentParams for details)
    :type env_params: EnvironmentParams
    :param ag_params: contains all the agent's parameters, (see AgentsParams for details)
    :type ag_params: AgentsParams
    :param ci: confidence interval (between 0. and 1.)
    :type ci: float
    :param experimental_data_pearce: The original data obtained in Pearce's main experiment, to compare to our simulation data
    :type experimental_data_pearce: dict of {str:list}
    """

    if directory is not None:
        directory_control = directory+"/control_group"
        directory_lesion = directory+"/lesioned_group"
    else:
        directory_control = None
        directory_lesion = None

    ag_params.lesion_HPC = False
    results_folder_normal = create_path_main_pearce(env_params, ag_params, directory_control)
    ag_params.lesion_HPC = True
    results_folder_lesion = create_path_main_pearce(env_params, ag_params, directory_lesion)
    ag_params.lesion_HPC = False

    if os.path.exists("../results/"+results_folder_normal) and os.path.exists("../results/"+results_folder_lesion): # if results has not been saved
        saved_results_folder_normal = "../results/"+results_folder_normal
        saved_results_folder_lesion = "../results/"+results_folder_lesion
    else:
        saved_results_folder_normal = "../saved_results/"+results_folder_normal
        saved_results_folder_lesion = "../saved_results/"+results_folder_lesion

    # data retrieving for normal rats
    all_data = []
    for ag in range(env_params.n_agents):
        df = pd.read_csv(os.path.join(saved_results_folder_normal, 'agent{}.csv'.format(ag)))
        summary = df.pivot_table(index=['agent', 'session', 'trial'], aggfunc='mean')
        all_data.append(summary)

    df = pd.concat(all_data)
    df['platform location'] = df['platform'].astype('category')
    df.to_csv(os.path.join(saved_results_folder_normal, 'summary.csv'))

    # data retrieving for HPC-lesioned rats
    all_data = []
    for ag in range(env_params.n_agents):
        df_lesion = pd.read_csv(os.path.join(saved_results_folder_lesion, 'agent{}.csv'.format(ag)))
        summary = df_lesion.pivot_table(index=['agent', 'session', 'trial'], aggfunc='mean')
        all_data.append(summary)

    df_lesion = pd.concat(all_data)
    df_lesion['platform location'] = df_lesion['platform'].astype('category')
    df_lesion.to_csv(os.path.join(saved_results_folder_lesion, 'summary.csv'))

    # plotting
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
    df_lesion = df_lesion.reset_index()
    df_lesion_trial1 = df_lesion[df_lesion['trial']==0]
    sns.lineplot(data=df_lesion_trial1, x=df_lesion_trial1['session']+1, y='escape time', label="HPC lesion - trial 1", linewidth = 5, color=pal[1], ci=ci)
    # HPC lesion - trial 4
    df_lesion = df_lesion.reset_index()
    df_lesion_trial4 = df_lesion[df_lesion['trial']==3]
    sns.lineplot(data=df_lesion_trial4, x=df_lesion_trial4['session']+1, y='escape time',  label="HPC lesion - trial 4", linewidth = 5, color=pal[1],  style=False, dashes=[(2,1)], ci=ci)
    # Control - trial 1
    df = df.reset_index()
    df_trial1 = df[df['trial']==0]
    sns.lineplot(data=df_trial1, x=df_trial1['session']+1, y='escape time', label="Control - trial 1", linewidth = 5, color=pal[0], ci=ci)
    # Control - trial 1
    df = df.reset_index()
    df_trial4 = df[df['trial']==3]
    sns.lineplot(data=df_trial4, x=df_trial4['session']+1, y='escape time', label="Control - trial 4", linewidth = 5,  color=pal[0], style=False, dashes=[(2,1)], ci=ci)

    axs[2].set_title("Our results")
    axs[2].set(ylabel="Escape time (steps)")
    axs[2].set(xlabel="Session")
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    handles, labels = axs[2].get_legend_handles_labels()
    axs[2].legend(handles=[handles[0],handles[1],handles[3],handles[4]], labels=[labels[0],labels[1],labels[3],labels[4]])
    axs[2].plot(aspect="auto")
    axs[2].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    # Plot the MSE of the four lines relative to Pearce original data
    if experimental_data is not None:
        # Pearce data
        real_data = experimental_data
        real_data = [real_data["cont1"], real_data["cont4"], real_data["hip1"], real_data["hip4"]]
        # Simulation data
        control_1bis = axs[2].get_lines()[3].get_xydata()[:,1]
        control_4bis = axs[2].get_lines()[4].get_xydata()[:,1]
        hippo_1bis = axs[2].get_lines()[0].get_xydata()[:,1]
        hippo_4bis = axs[2].get_lines()[1].get_xydata()[:,1]
        expected_data = [control_1bis, control_4bis, hippo_1bis, hippo_4bis]
        se = get_MSLE(real_data, expected_data, relative=True)
        print("Mean Square Error: ", se)
    if show_plots:
        plt.show()
    if save_plots:
        try:
            fig.savefig(os.path.join("../results/"+directory, 'full_results.png'))
        except:
            fig.savefig(os.path.join("../saved_results/"+directory, 'full_results.png'))
    plt.close()

    # BELOW : print ANOVA analyses comparing groups performances at trials 1 and 4

    # 200 rows, 100*trial 1, 100*trial 4
    df_anova_trial1 = df.reset_index()
    df_anova_trial1 = df_anova_trial1[df_anova_trial1["trial"]==0]
    df_anova_trial1 = df_anova_trial1.pivot_table(index=['agent', 'trial'], aggfunc='mean')
    df_anova_trial1 = df_anova_trial1[["escape time"]]

    df_anova_trial4 = df.reset_index()
    df_anova_trial4 = df_anova_trial4[df_anova_trial4["trial"]==3]
    df_anova_trial4 = df_anova_trial4.pivot_table(index=['agent', 'trial'], aggfunc='mean')
    df_anova_trial4 = df_anova_trial4[["escape time"]]

    # 200 rows, 100*trial 1, 100*trial 4
    df_anova_trial_lesion1 = df_lesion.reset_index()
    df_anova_trial_lesion1 = df_anova_trial_lesion1[df_anova_trial_lesion1["trial"]==0]
    df_anova_trial_lesion1 = df_anova_trial_lesion1.pivot_table(index=['agent', 'trial'], aggfunc='mean')
    df_anova_trial_lesion1 = df_anova_trial_lesion1[["escape time"]]

    df_anova_trial_lesion4 = df_lesion.reset_index()
    df_anova_trial_lesion4 = df_anova_trial_lesion4[df_anova_trial_lesion4["trial"]==3]
    df_anova_trial_lesion4 = df_anova_trial_lesion4.pivot_table(index=['agent', 'trial'], aggfunc='mean')
    df_anova_trial_lesion4 = df_anova_trial_lesion4[["escape time"]]

    df_anova_trial1["group"] = "normal"
    df_anova_trial_lesion1["group"] = "lesioned"
    df_anova_trial4["group"] = "normal"
    df_anova_trial_lesion4["group"] = "lesioned"

    df_both_1 = pd.concat([df_anova_trial1, df_anova_trial_lesion1])
    df_both_4 = pd.concat([df_anova_trial4, df_anova_trial_lesion4])

    # perform ANOVA (IV -> group, DV -> escape time)
    try:
        df_both_1["escape_time"] = df_both_1["escape time"]
        model = ols('escape_time ~ C(group) + C(group)', data=df_both_1.reset_index()).fit()
        print()
        print("Computing ANOVA to test difference between trial 1 of group normal and group lesioned...")
        print(sm.stats.anova_lm(model, typ=2))
        print()
    except:
        print("Anova failed (there might not be enough data)")
        print()

    # perform ANOVA (IV -> group, DV -> escape time)
    try:
        df_both_4["escape_time"] = df_both_4["escape time"]
        model = ols('escape_time ~ C(group) + C(group)', data=df_both_4.reset_index()).fit()
        print()
        print("Computing ANOVA to test difference between trial 4 of group normal and group lesioned...")
        print(sm.stats.anova_lm(model, typ=2))
        print()
    except:
        print("Anova failed (there might not be enough data)")
        print()


def determine_platform_seq(env, platform_states, n_sessions):
    """
    Recursive function that create a platform random sequence but respecting certain rules.
    The rules are that a platform should not occur too much in the sequence (see code for exact rule), and
    that shift of the platform between sessions should only occur between far enough platforms.

    :param env: the environment
    :type env: HexWaterMaze
    :param platform_states: the eligible platform states
    :type platform_states: int list
    :param n_sessions: the number of shift from a platform to another
    :type n_sessions: int

    :returns: The random platform sequence
    :return type: int list
    """

    plat_seq = None # to be returned
    try:
        indices = np.arange(len(platform_states))
        usage = np.zeros(len(platform_states)) # keep track of platform occurence in the sequence to return
        plat_seq = [np.random.choice(platform_states)]
        for sess in range(1, n_sessions):
            distances = np.array([env.grid.distance(plat_seq[sess - 1], s) for s in platform_states])
            # keep only platform apart from last platform and with a low number of occurence in the sequence
            candidates = indices[np.logical_and(usage < n_sessions/8+1, distances > env.grid.radius*0.8)]
            platform_idx = np.random.choice(candidates)
            plat_seq.append(platform_states[platform_idx])
            usage[platform_idx] += 1.
    except Exception: # can happen if no candidate is encountered (far enough platforms were all selected too much in the past)
        return determine_platform_seq(env, platform_states, n_sessions)

    return plat_seq


def get_maze_pearce(maze_size, landmark_dist, edge_states):
    """
    Create the environment to simulate Pearce's task
    :param maze_size: diameter of the Morris pool (number of states). 10 is used in Geerts' and our work
    :type maze_size: int
    :param landmark_dist: number of states separating the landmark from the platform
    :type landmark_dist: int
    :param edge_states: list of eligible starting states
    :type edge_states: list of int
    """
    if maze_size == 6:
        possible_platform_states = np.array([48, 51, 54, 57, 60, 39, 42, 45])
        envi = HexWaterMaze(6, landmark_dist, edge_states)
        return possible_platform_states, envi
    if maze_size == 10:
        # [192, 185, 181, 174, 216, 210, 203, 197] is Geerts version
        possible_platform_states = np.array([48, 52, 118, 122, 126, 94, 98, 44])
        envi = HexWaterMaze(10, landmark_dist, edge_states)
        return possible_platform_states, envi
    if maze_size == 12:
        possible_platform_states = np.array([300, 292, 284, 277, 329, 321, 313, 306])
        envi = HexWaterMaze(12, landmark_dist, edge_states)
        return possible_platform_states, envi


def create_path_main_pearce(env_params, ag_params, directory=None):
    """
    Create a path to the directory where the data of all agents from a group of
    simulations with identical parameters has been stored

    :param env_params: contains all the environment's parameters, to concatenate in the path (see EnvironmentParams for details)
    :type env_params: EnvironmentParams
    :param ag_params: contains all the agent's parameters, to concatenate in the path (see AgentsParams for details)
    :type ag_params: AgentsParams
    :param directory: optional directory where to store all the results
    :type directory: str

    :returns: the path of the results folder
    :return type: str
    """
    path = create_path(env_params, ag_params)
    path = "pearce_"+str(env_params.maze_size)+str(env_params.n_trials)+str(env_params.n_sessions)+path
    if directory is not None:  # used for the grid-search, where the user gives a name to a hierarchically higher directory
        path = directory+"/"+path
    return path
