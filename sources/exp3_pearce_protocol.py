import os
import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pearce_protocol import create_path_main_pearce
from utils import get_mean_preferred_dirs, plot_mean_arrows, charge_agents


def perform_group_exp3_pearce(env_params, ag_params, dist = 4, alt_dist = -4, directory=None, show_plots=True, save_plots=False):
    """
    Subject a group of agents to the third experiment of Pearce 1998.
    In accord with Pearce 1998 protocol, each agents used for the third experiment has previously been trained on
    the full Pearce 1998 first experiment (perform_group_pearce() in the main_pearce module in our work).
    Agents are subdivided in two experimental groups: control and inverted landmark distance group.
    Both groups are then subjected to one episode of the first Pearce 1998 task where the agent must find a submerged
    platform, in a limited amount of time in a water-maze, with only a proximal landmark indicating the presence of the platform.
    The difference in this third experiment is that in this additional episode, the platform to landmark distance is reversed
    for the non-control group.
    Pearce et al, in their original article found a significant difference of escape time between
    control and inverted landmark groups.
    We want to check whether the Geert 2020 and Dolle 2010 coordination models are able to replicate this result.
    See Pearce 1998 for the exact protocol.

    IMPORTANT : pass lesion_HPC as True to simulate and visualize data for hpc-lesioned agents and lesion_HPC as False for normal agents

    :param env_params: Contains all the parameters to set the water-maze RL environment (see EnvironmentParams for details)
    :type env_params: EnvironmentParams
    :param ag_params: Contains all the parameters to set the different RL modules of the agent (see AgentsParams for details)
    :type ag_params: AgentsParams
    :param dist: the platform-landmark distance for the control group
    :type dist: int
    :param alt_dist: the platform-landmark distance for the inverted group
    :type alt_dist: int
    :param directory: optional directory where to store all the results
    :type directory: str
    :param show_plots: whether to display any created plot at the end of the simulations
    :type show_plot: boolean
    :param save_plots: whether to save any created plots in the results folder at the end of the simulations
    :type save_plots: boolean
    """
    # Original results
    fig,(ax0, ax1)=plt.subplots(1,2, figsize=(15,6))
    if not ag_params.lesion_HPC:
        fig.suptitle("Mean escape time for normal and inverted landmark distances (Healthy rats)")
        ax0.bar(["normal", "inverted"], height=[10.2, 54.6], color="gray", edgecolor="black")
    else:
        fig.suptitle("Mean escape time for normal and inverted landmark distances (HPC lesioned rats)")
        ax0.bar(["normal", "inverted"], height=[9.8, 59.4], color="gray", edgecolor="black")

    ax0.title.set_text("Original results")
    ax0.set_xlabel('landmark position')
    ax0.set_ylabel('mean escape time (s)')

    df_normal = []
    df_inverted = []

    # run the additional episode for the control group, then the inverted group
    try:
        df_normal.append(run_test(env_params, ag_params, dist, directory))
        df_inverted.append(run_test(env_params, ag_params, alt_dist, directory))
    except FileNotFoundError:
        print()
        print("Error: Experience 3 of Pearce 1998 can only be conducted with agents trained on the first experiment")
        print("Please train rats on the first experiment of Pearce using perform_full_pearce script, ")
        print("then execute perform_full_exp_pearce using the same parameters and directory")
        print()
        print("command example:")
        print("'python perform_full_pearce.py mydirectory best_geerts'")
        print("'python perform_full_exp3_pearce.py mydirectory best_geerts'")
        print()
        print("check that the first experiment and third experiment parameters are identical, if an error persist")
        print()
        sys.exit(1)

    df_normal = pd.concat(df_normal)
    df_inverted = pd.concat(df_inverted)
    x = ["normal", "inverted"]
    y = [df_normal["escape time"].mean(),df_inverted["escape time"].mean()]
    
    # if show_plots:
    #     print("normal: ", df_normal["escape time"].mean(), " steps")
    #     print("inverted: ", df_inverted["escape time"].mean(), " steps")

    ax1.bar(x, height=y, yerr=[df_normal["escape time"].std()/ np.sqrt(df_normal["escape time"].shape[0])*1.96, df_inverted["escape time"].std()/np.sqrt(df_inverted["escape time"].shape[0])*1.96])
    ax1.title.set_text("Our results")
    ax1.set_xlabel('landmark position')
    ax1.set_ylabel('mean escape time (steps)')

    if show_plots:
        plt.show()
    if save_plots:
        try:
            plt.savefig(os.path.join("../results/"+directory, 'exp3_pearce_mean_escape_time.png'))
        except:
            plt.savefig(os.path.join("../saved_results/"+directory, 'exp3_pearce_mean_escape_time.png'))

    # charge agents (not the one subjected to the third exp of Pearce)
    # to plot their heading-vectors at different locations of the maze
    path = create_path_main_pearce(env_params, ag_params)
    if directory is not None:
        path = directory+"/"+path
    try:
        agents = charge_agents("../saved_results/"+path+"/agents.p")
    except:
        agents = charge_agents("../results/"+path+"/agents.p")

    for ag in agents:
        ag.env.landmark_dist = dist
    if str(type(ag)) != "<class 'agents.dolle_agent.DolleAgent'>" :
        res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs(agents, 0, 0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined)
    else:
        res2_mf, res2_allo, res2_sr, res2_combined, decisions_arbi = get_mean_preferred_dirs(agents, 0, 0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined, 0, decisions_arbi)

    fig.suptitle("Strategies used with normal landmarks directions", fontsize=14)

    if save_plots:
        try:
            plt.savefig(os.path.join("../results/"+directory, 'exp3_pearce_normal_mean_heading_vectors.png'))
        except:
            plt.savefig(os.path.join("../saved_results/"+directory, 'exp3_pearce_normal_mean_heading_vectors.png'))

    for ag in agents:
        ag.env.landmark_dist = alt_dist
    if str(type(ag)) != "<class 'agents.dolle_agent.DolleAgent'>" :
        res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs(agents, 0, 0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined)
    else:
        res2_mf, res2_allo, res2_sr, res2_combined, decisions_arbi = get_mean_preferred_dirs(agents, 0, 0)
        ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined, 0, decisions_arbi)

    fig.suptitle("Strategies used with inverted landmarks directions", fontsize=14)

    if show_plots:
        plt.show()
    if save_plots:
        try:
            plt.savefig(os.path.join("../results/"+directory, 'exp3_pearce_inverted_mean_heading_vectors.png'))
        except:
            plt.savefig(os.path.join("../saved_results/"+directory, 'exp3_pearce_inverted_mean_heading_vectors.png'))
    plt.close()


def run_test(env_params, ag_params, ld_dist, directory=None):
    """
    Build a path to an existing results folder. Charge an already trained group of agents from a previous Pearce 1998
    experiment simulation. Launch the third experiment of Pearce 1998 with the charged agents.

    :param env_params: Contains all the parameters to set the water-maze RL environment (see EnvironmentParams for details)
    :type env_params: EnvironmentParams
    :param ag_params: Contains all the parameters to set the different RL modules of the agent (see AgentsParams for details)
    :type ag_params: AgentsParams
    :param ld_dist: the platform-landmark distance for the experimental group
    :type ld_dist: int
    :param directory: optional directory where to store all the results
    :type directory: str

    :returns: A dataframe containing the logs of simulations of the third experiment of pearce
    :return type: pandas DataFrame
    """

    path = create_path_main_pearce(env_params, ag_params)
    if directory is not None:
        path = directory+"/"+path

    try:
        agents = charge_agents("../saved_results/"+path+"/agents.p")
    except:
        agents = charge_agents("../results/"+path+"/agents.p")

    df = get_mean_escape_time(agents, ld_dist, ag_params.mf_allo, env_params.time_limit)
    del agents
    return df


def get_mean_escape_time(agents, landmark_dist, allo, time_limit):
    """
    Subject a group of agents to the third experiment of Pearce 1998. (either control group
    or inverted landmark distance group, all depends on the landmark_dist parameter).

    :param agents: the group of agents to subject to the third experiment of Pearce
    :type agents: Agent list
    :param landmark_dist: distance (in states) between the platform and the landmark (positive if normal, negative for inverted group)
    :type landmark_dist: int
    :param allo: whether the agent MF model is in an egocentric or allocentric frame of reference
    :type allo: boolean
    :param time_limit: max number of steps authorized before the end of the episode is forced
    :type time_limit: int

    :returns: A dataframe containing the logs of simulations of the third experiment of pearce
    :return type: pandas DataFrame
    """
    df_agents = []
    id_agent = 0

    for agent in agents:

        agent.env.landmark_dist = landmark_dist
        agent.env.set_platform_state(0)
        agent.env.delete_distal_landmark() # not used in Pearce experiment
        agent.env.set_proximal_landmark()

        res = agent.env.one_episode(agent,time_limit)
        res['escape time'] = res.time.max()
        res['id_agent'] = id_agent
        df_agents.append(res)
        id_agent += 1

    df_agents = pd.concat(df_agents)
    result_df = df_agents.pivot_table(index=["id_agent"], aggfunc='mean')

    return result_df
