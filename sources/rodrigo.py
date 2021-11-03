from agents.dolle_agent import DolleAgent
from agents.fusion_agent import CombinedAgent
from environments.HexWaterMaze import HexWaterMaze
from utils import create_path, create_df, get_coords, isinoctant

from IPython.display import clear_output
from statsmodels.formula.api import ols
from patsy.contrasts import Helmert
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np
import random
import pickle
import shutil
import time
import os


def perform_rodrigo(n_agents, mf_allo, sr_lr, q_lr, gamma, eta, alpha1, beta1, A_alpha, A_beta, landmark_dist, HPCmode, time_limit, edge_states, lesion_HPC, lesion_DLS, dolle=False, create_plots = True, save_agents=True, inv_temp=None, inv_temp_gd=None, inv_temp_mf=None, arbi_inv_temp = None, directory=None, verbose=True):
    """
    Run multiple simulations of the first experiment of Rodrigo 2006, a Morris water-maze derived task where a rat has
    to navigate through a circular maze filled with water, to find a submerged platform indicated by both a
    proximal visual landmark hovering directly over the platform and a distal visual landmark located at a
    fixed distance of the platform.
    During test trials, the proximal landmark is rotated to a given angle respecting to the center of the maze, while the
    distal landmark remain at the same location. There are five different possible angles (0°, 45°, 90°, 135° and 180°),
    each tested during a different test trial.
    With increasing angle of the rotation of the proximal beacon, the mean occupation time of both the distal
    beacon and proximal beacon octants were observed to decrease in Rodrigo's study. In this work, we want to check
    whether Dolle's and Geerts' coordination models are able to reproduce this effect, known as generalization gradient.

    Roughly, the experiment is composed of three different stages. First a pre-exploration phase of 5 trials where no landmarks
    are presents but the platform is present. Then a second training stage of 4 sessions of 8 trials where both the proximal
    and distal landmarks are present. And finally a third stage of 10 sessions of 9 trials where one trial of each
    session is either an extinction trial (uneven sessions) or test trial (even sessions).
    During extinction trials, no platform and proximal beacon are present. During test trials, no platform is present and
    the proximal landmark is rotated.
    See Rodrigo 2006 for original results and exact protocol.

    :param n_agents: number of simulations (with a different agent for each) to run
    :type n_agents: int
    :param mf_allo: whether the DLS module is in the allocentric or egocentric frame of reference
    :type mf_allo: boolean
    :param q_lr: learning rate of the DLS model
    :type q_lr: float
    :param sr_lr: learning rate of the HPC model (SR or MB)
    :type sr_lr: float
    :param gamma: discount factor of value propagation (shared between coordination, SR, MB and MF models)
    :type gamma: float
    :param eta: used to update the HPC and DLS models' reliability (Geerts coordination model only)
    :type eta: float
    :param alpha1: used to compute the transition rate from MF to SR (Geerts coordination model only)
    :type alpha1: float
    :param beta1: used to compute the transition rate from SR to MF (Geerts coordination model only)
    :type beta1: float
    :param A_alpha: steepness of transition curve MF to SR (Geerts coordination model only)
    :type A_alpha: float
    :param A_beta: steepness of transition curve SR to MF (Geerts coordination model only)
    :type A_beta: float
    :param landmark_dist: number of states separating the landmark from the platform
    :type landmark_dist: int
    :param HPCmode: to choose the model of the HPC, either "SR" or "MB"
    :type HPCmode: str
    :param time_limit: max number of timestep to find the reward, the episode is forced to end if reached
    :type time_limit: int
    :param edge_states: list of eligible starting states
    :type edge_states: list of int
    :param lesion_HPC: Whether the DLS module is inactivated or not (full control of HPC if True)
    :type lesion_HPC: boolean
    :param lesion_DLS: Whether the HPC module is inactivated or not (full control of DLS if True)
    :type lesion_DLS: boolean
    :param dolle: whether the coordination model is based on associative learning (Dolle) or fusion (Geerts)
    :type dolle: boolean
    :param create_plots: whether to display any plot at all at the end of the simulations
    :type create_plot: boolean
    :param save_agents:  whether to save Agents object in the result folder (take a lot of memory)
    :type save_agents: boolean
    :param inv_temp: DLS inverse temperature for softmax exploration (Geerts model only)
    :type inv_temp: int
    :param inv_temp_gd: HPC inverse temperature for softmax exploration (Dolle model only)
    :type inv_temp_gd: int
    :param inv_temp_mf: DLS inverse temperature for softmax exploration (Dolle model only)
    :type inv_temp_mf: int
    :param arbi_inv_temp: Coordination model inverse temperature for softmax exploration (Dolle model only)
    :type arbi_inv_temp: int
    :param directory: optional directory where to store all the results (used for grid-search)
    :type directory: str
    :param verbose: whether to print the progress of the simulations
    :type verbose: boolean
    """

    possible_platforms = [75, 99, 90, 117] # eligible platform states
    maze_size = 10 # diameter of the hexagonal maze (in states)
    pretraining_trial_nbr = 5
    n_sessions1 = 4 # phase 1
    n_sessions2 = 10 # phase 2
    n_trials1 = 8 # number of trials at session 1
    n_trials2 = 9 # number of trials at session 2
    # 120 seconds in rodrigo's (2006) paper
    time_limit = 500
    landmark_dist = 0

    # create environment
    possible_platform_states, envi = get_maze_rodrigo(maze_size, landmark_dist, edge_states)
    # get results directory path
    results_folder = create_path_rodrigo(n_agents, mf_allo, sr_lr, q_lr, gamma, eta, alpha1, beta1, A_alpha, A_beta, landmark_dist, HPCmode, time_limit, edge_states, lesion_HPC, lesion_DLS, dolle, inv_temp=inv_temp, inv_temp_gd=inv_temp_gd, inv_temp_mf=inv_temp_mf, arbi_inv_temp = arbi_inv_temp, directory=directory)

    saved_results_folder = "../saved_results/"+results_folder # never erased
    results_folder = "../results/"+results_folder # erased if an identical simulation is run

    if not os.path.isdir(saved_results_folder):
        if os.path.isdir(results_folder): # delete previous identical simulation data (as it wasn't saved)
            shutil.rmtree(results_folder)
        # creates results folders
        figure_folder = os.path.join(results_folder, 'figs')
        os.makedirs(results_folder)
        os.makedirs(figure_folder)

        agents = []
        for n_agent in range(n_agents):

            # np.random.seed(n_agent) # uncomment to make every simulation identical

            # intialise agent (either Geerts' model or Dolle's)
            if not dolle:
                agent = CombinedAgent(envi, init_sr='zero',
                                      inv_temp=inv_temp,
                                      gamma=gamma,
                                      q_lr=q_lr,
                                      hpc_lr=sr_lr,
                                      eta=eta,
                                      alpha1=alpha1,
                                      beta1=beta1,
                                      A_alpha = A_alpha,
                                      A_beta = A_beta,
                                      mf_allo = mf_allo,
                                      HPCmode = HPCmode,
                                      lesion_hpc = lesion_HPC,
                                      lesion_dls = lesion_DLS,
                                    )
            else:
                agent = DolleAgent(envi, init_sr='zero',
                                      inv_temp_gd=inv_temp_gd,
                                      inv_temp_mf= inv_temp_gd,
                                      gamma=gamma,
                                      q_lr=q_lr,
                                      hpc_lr=sr_lr,
                                      learning_rate=eta,
                                      mf_allo = mf_allo,
                                      HPCmode = HPCmode,
                                      lesion_hpc = lesion_HPC,
                                      lesion_dls = lesion_DLS,
                                      arbi_inv_temp = arbi_inv_temp)

            total_trial_count = 0

            # PRETRAINING
            # list of eligible starting states in a random order
            pretraining_platforms = get_rodrigo_platforms_pretraining(possible_platforms)
            agent_df = pd.DataFrame() # to create a log file keeping track of the agents performances
            for trial in range(pretraining_trial_nbr):
                if verbose:
                    print("agent: "+str(n_agent)+", session: pretraining, trial: "+str(trial)+"                        ", end="\r")

                envi.set_platform_state(pretraining_platforms[trial])
                envi.delete_landmarks()
                # simulate one episode, res is a dataframe keeping track of agent's and environment's variables at each timesteps
                res = envi.one_episode(agent, time_limit)

                # add infos for each trial
                res['trial'] = trial
                res['escape time'] = res.time.max()
                res['session'] = "pretraining"
                res['stage'] = "pretraining"
                res['total trial'] = total_trial_count
                res["cond"] = "pretraining"
                res['angle'] = "pretraining"
                res["proximal_posx"] = "pretraining"
                res["proximal_posy"] = "pretraining"
                res["distal_posx"] = "pretraining"
                res["distal_posy"] = "pretraining"
                agent_df=agent_df.append(res, ignore_index=True)

                total_trial_count += 1

            # FIRST STAGE
            # the platform is only chosen once at the beginning of session 1
            platform_state = random.choice(possible_platforms)
            envi.set_platform_state(platform_state)
            envi.set_proximal_landmark()
            envi.set_distal_landmark()
            for ses in range(n_sessions1):
                for trial in range(n_trials1):
                    if verbose:
                        print("agent: "+str(n_agent)+", stage: 1, session: "+str(ses)+", trial: "+str(trial)+"                              ", end="\r")

                    res = envi.one_episode(agent, time_limit)

                    res['trial'] = trial
                    res['escape time'] = res.time.max()
                    res['session'] = ses
                    res['stage'] = "first"
                    res['total trial'] = total_trial_count
                    res["cond"] = "escape"
                    res['angle'] = "session1"
                    res["proximal_posx"] = envi.proximal_landmark_location[0]
                    res["proximal_posy"] = envi.proximal_landmark_location[1]
                    res["distal_posx"] = envi.distal_landmark_location[0]
                    res["distal_posy"] = envi.distal_landmark_location[1]
                    agent_df = agent_df.append(res, ignore_index=True)

                    total_trial_count += 1

            # SECOND STAGE
            possible_angles = [[0,45,90,135,180],[0,-45,-90,-135,-180]] # angles of the proximal beacon during test trials
            angles = random.sample(random.choice(possible_angles), 5) # select either positive or negative angle list and shuffle
            angles = dict(zip([2,4,6,8,10], angles)) # associate each angle to a trial

            for ses in range(1, n_sessions2+1):

                cond = None
                special_trial = -1
                if ses in [1,3,5,7,9]:
                    cond = "extinction"
                    special_trial = random.choice(range(1,9))
                elif ses in [2,4,6,8,10]:
                    cond = "test"
                    special_trial = 8
                else:
                    raise Exception("cond is neither extinction nor test")

                for trial in range(n_trials2):
                    if verbose:
                        print("agent: "+str(n_agent)+", stage: 2, session: "+str(n_sessions1 + ses)+", trial: "+str(trial)+"                              ", end="\r")

                    if special_trial == trial and cond == "extinction":
                        envi.delete_proximal_landmark()
                        envi.delete_plaform()
                        res = envi.one_episode(agent, time_limit/2)
                        # put platform and proximal landmark at normal again
                        envi.set_platform_state(platform_state)
                        envi.set_proximal_landmark()
                        res["proximal_posx"] = "extinction"
                        res["proximal_posy"] = "extinction"
                        res["distal_posx"] = "extinction"
                        res["distal_posy"] = "extinction"
                        res["cond"] = "extinction"
                        res['angle'] = "extinction"

                    elif special_trial == trial and cond == "test":
                        envi.delete_plaform()
                        envi.set_angle_proximal_beacon(angles[ses]) # rotate the proximal landmark
                        res = envi.one_episode(agent, time_limit/2)
                        res["proximal_posx"] = envi.proximal_landmark_location[0]
                        res["proximal_posy"] = envi.proximal_landmark_location[1]
                        res["distal_posx"] = envi.distal_landmark_location[0]
                        res["distal_posy"] = envi.distal_landmark_location[1]
                        res["cond"] = "test"
                        res['angle'] = angles[ses]
                        # put platform and proximal landmark at normal again
                        envi.set_platform_state(platform_state)
                        envi.set_proximal_landmark()

                    else: # normal trial
                        res = envi.one_episode(agent, time_limit)
                        res["proximal_posx"] = envi.proximal_landmark_location[0]
                        res["proximal_posy"] = envi.proximal_landmark_location[1]
                        res["distal_posx"] = envi.distal_landmark_location[0]
                        res["distal_posy"] = envi.distal_landmark_location[1]
                        res["cond"] = "escape"
                        res['angle'] = "escape"

                    res['trial'] = trial
                    res['escape time'] = res.time.max()
                    res['session'] = ses
                    res['stage'] = "second"
                    res['total trial'] = total_trial_count
                    agent_df = agent_df.append(res, ignore_index=True)

                    total_trial_count += 1

            # add infos for each simulation
            agent_df['total time'] = np.arange(len(agent_df))
            agent_df['agent'] = n_agent

            agent_df.to_csv(os.path.join(results_folder, 'agent{}.csv'.format(n_agent)))
            agents.append(agent)

            # erase standard output
            clear_output()

        # take a lot of memory
        if save_agents:
            file_to_store = open(results_folder+"/agents.p", "wb")
            pickle.dump(agents, file_to_store)
            file_to_store.close()

        # plot a histogram of the mean occupancy of distal landmark octant and proximal landmark octant for each angle condition
        if create_plots:
            plot_rodrigo(results_folder, n_agents)

    # if an identical simulation has already been saved
    else:
        # plot a histogram of the mean occupancy of distal landmark octant and proximal landmark octant for each angle condition
        plot_rodrigo(saved_results_folder, n_agents)

    # delete all agents, to prevent memory error
    if 'agents' in locals():
        for i in agents:
            del i
        del agents

def run_statistical_tests_rodrigo(path, n_agents):
    """
    Check and print if a group of agents validate multiple statistical test,
    originally performed in rodrigo 2006 (Helmert contrasts, ANOVAs, TTests)

    :param path: path where the simulation data to analyse is stored
    :type path: str
    """
    coords = get_coords() # associate each state of the water-maze with a cartesian coordinate
    df_analysis = pd.DataFrame()
    agents_df_lst = []

    for agent_ind in range(n_agents):
        print("agent: "+ str(agent_ind), end="\r")
        one_agent_df = pd.read_csv(path+"/agent"+str(agent_ind)+".csv")
        one_agent_df["agent"] = agent_ind
        one_agent_df = one_agent_df[one_agent_df["cond"] == "test"]
        agents_df_lst.append(one_agent_df)

    print("Concatenating all data")
    df_analysis = pd.concat(agents_df_lst)

    print("Computing proximal and distal octants mean occupation on test episodes")
    dist0 = get_mean_occupation_octant(0, df_analysis, coords)
    dist45 = get_mean_occupation_octant(45, df_analysis, coords)
    dist90 = get_mean_occupation_octant(90, df_analysis, coords)
    dist135 = get_mean_occupation_octant(135, df_analysis, coords)
    dist180 = get_mean_occupation_octant(180, df_analysis, coords)

    octant_occup_df = pd.concat([dist0, dist45, dist90, dist135, dist180], ignore_index=False)

    helmert_rodrigo_0vs_results_p = []
    helmert_rodrigo_45vs_results_p = []
    helmert_rodrigo_90vs_results_p = []
    helmert_rodrigo_0vs_results_d = []
    helmert_rodrigo_45vs_results_d = []
    helmert_rodrigo_90vs_results_d = []
    anova_rodrigo_results = []
    ttest_rodrigo_results = []

    p = 0.05

    print("Performing statistical analyses")

    cluster_df = octant_occup_df
    cluster_df = cluster_df.reset_index()

    # HELMERT TESTS
    helmert_p = lambda lst : ols("isinoctant_proximal ~ C(angle, Helmert)", data=cluster_df[cluster_df["angle"].isin(lst)]).fit()
    helmert_rodrigo_0vs_results_p.append(helmert_p([0,45,90,135,180]).f_pvalue < p)
    helmert_rodrigo_45vs_results_p.append(helmert_p([45,90,135,180]).f_pvalue < p)
    helmert_rodrigo_90vs_results_p.append(helmert_p([90,135,180]).f_pvalue < p)

    # HELMERT TESTS
    helmert_d = lambda lst : ols("isinoctant_distal ~ C(angle, Helmert)", data=cluster_df[cluster_df["angle"].isin(lst)]).fit()
    helmert_rodrigo_0vs_results_d.append(helmert_d([0,45,90,135,180]).f_pvalue < p)
    helmert_rodrigo_45vs_results_d.append(helmert_d([45,90,135,180]).f_pvalue < p)
    helmert_rodrigo_90vs_results_d.append(helmert_d([90,135,180]).f_pvalue < p)

    # ANOVA
    model = ols('isinoctant_proximal ~ C(angle) + C(angle) + C(angle):C(angle)', data=cluster_df).fit()
    tmp = sm.stats.anova_lm(model, typ=2)
    model2 = ols('isinoctant_distal ~ C(angle) + C(angle) + C(angle):C(angle)', data=cluster_df).fit()
    tmp2 = sm.stats.anova_lm(model2, typ=2)
    anova_rodrigo_results.append(tmp["PR(>F)"]["C(angle)"]<p and tmp2["PR(>F)"]["C(angle)"]<p)

    # TTESTS
    ttest = lambda ang : stats.ttest_1samp(cluster_df[cluster_df["angle"] == ang].groupby("agent").mean()["isinoctant_proximal"],0.125).pvalue
    ttest2 = lambda ang : stats.ttest_1samp(cluster_df[cluster_df["angle"] == ang].groupby("agent").mean()["isinoctant_distal"],0.125).pvalue
    ttest_rodrigo_results.append(ttest(0) < p and ttest(45) < p and ttest(90) < p and ttest(135) < p and ttest(180) < p and ttest2(0) < p and ttest2(45) < p)
    print("Helmert tests")
    print("p < 0.05 on 0° versus others (proximal beacon): ", helmert_p([0,45,90,135,180]).f_pvalue < p)
    print("p < 0.05 on 45° versus others (proximal beacon): ",helmert_p([45,90,135,180]).f_pvalue < p)
    print("p < 0.05 on 90° versus others (proximal beacon): ",helmert_p([90,135,180]).f_pvalue < p)
    print("p < 0.05 on 0° versus others (distal beacon): ",helmert_d([0,45,90,135,180]).f_pvalue < p)
    print("p < 0.05 on 45° versus others (distal beacon): ",helmert_d([45,90,135,180]).f_pvalue < p)
    print("p < 0.05 on 90° versus others (distal beacon): ",helmert_d([90,135,180]).f_pvalue < p)
    print()
    print("ANOVAS")
    print("Effect of angle on proximal beacon's octant occupation: ",tmp["PR(>F)"]["C(angle)"]<p)
    print("Effect of angle on distal beacon's octant occupation: ",tmp2["PR(>F)"]["C(angle)"]<p)
    print()
    print("TTESTS")
    print("Proximal beacon's octant occupation different from chance: ", ttest(0) < p)
    print("Proximal beacon's octant occupation different from chance: ", ttest(45) < p)
    print("Proximal beacon's octant occupation different from chance: ", ttest(90) < p)
    print("Proximal beacon's octant occupation different from chance: ", ttest(135) < p)
    print("Proximal beacon's octant occupation different from chance: ", ttest(180) < p)
    print("Distal beacon's octant occupation different from chance: ", ttest2(0) < p)
    print("Distal beacon's octant occupation different from chance: ", ttest2(45) < p)
    print("Distal beacon's octant occupation different from chance: ", ttest2(90) < p)
    print("Distal beacon's octant occupation different from chance: ", ttest2(135) < p)
    print("Distal beacon's octant occupation different from chance: ", ttest2(180) < p)


def plot_rodrigo(results_folder, n_agents):
    """
    Plot a histogram of the mean proportion of occupancy of distal landmark octant and proximal landmark octant
    for each angle condition.

    :param results_folder: path of the results folder where the data to plot is stored
    :type results_folder: str
    :param n_agents: number of simulations stored in results_folder
    :type n_agents: int
    """
    # get the mean occupancy of octants in each 10 conditions + the confidence interval (y) for each condition
    (dist0, dist45, dist90, dist135, dist180, prox0, prox45, prox90, prox135, prox180, ydist0, ydist45, ydist90, ydist135, ydist180, yprox0, yprox45, yprox90, yprox135, yprox180) = get_values_rodrigo(results_folder, n_agents)

    fig, axs = plt.subplots(2, 2, figsize=(15,12))

    axs[0,0].set_title("Original results")
    axs[0,0].imshow(mpimg.imread("../images/results_rodrigo_proximal.jpg"))
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,0].set_frame_on(False)
    axs[0,0].plot(aspect="auto")

    axs[1,0].imshow(mpimg.imread("../images/results_rodrigo_distal.jpg"))
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])
    axs[1,0].set_frame_on(False)
    axs[1,0].plot(aspect="auto")

    axs[0,1].bar(["0°", "45°", "90°", "135°", "180°"], [prox0, prox45, prox90, prox135, prox180], yerr=[yprox0, yprox45, yprox90, yprox135, yprox180], color='gray', edgecolor="black", )
    axs[0,1].set_title("Our results")
    axs[0,1].set_ylabel("Proportion of steps searching in the Beacon octant")
    axs[0,1].set_xlabel("Tests")

    axs[1,1].bar(["0°", "45°", "90°", "135°", "180°"], [dist0, dist45, dist90, dist135, dist180], yerr=[ydist0, ydist45, ydist90, ydist135, ydist180], color='gray', edgecolor="black")
    axs[1,1].set_ylabel("Proportion of steps searching in the Frame octant")
    axs[1,1].set_xlabel("Tests")

    plt.show()
    plt.close()
    run_statistical_tests_rodrigo(results_folder, n_agents)
    # check and print if the group of simulations at results_folder validate the multiple required statistical test, originally performed in rodrigo 2006
    # try:
    #     run_statistical_tests_rodrigo(results_folder)
    # except:
    #     print("Statical analysis failed (there might not be enough data)")


def get_mean_occupation_octant(angle, df_analysis, coords):
    """
    Compute and return the mean proportion of occupancy of both the proximal beacon and distal beacon octants, on test trials,
    at a specific angle of the proximal landmark rotation.

    :param angle: the angle of the proximal landmark rotation on the test condition. Either 0°, 45°, 90°, 135° or 180°
    :type angle: int
    :param df_analysis: the DataFrame containing data of all agents simulated
    :type df_analysis: pandas DataFrame
    :param coords: a dictionary linking states to cartesian coordinates
    :type coords: dict
    :return type: pandas DataFrame
    """
    df = df_analysis[np.logical_or(df_analysis["angle"]==str(angle), df_analysis["angle"]==str(-angle))]
    df['angle'] = angle
    df['isinoctant_distal'] = df.apply(lambda row: isinoctant(coords[row.state], [float(row.distal_posx), float(row.distal_posy)]), axis=1)
    df['isinoctant_proximal'] = df.apply(lambda row: isinoctant(coords[row.state], [float(row.proximal_posx), float(row.proximal_posy)]), axis=1)
    df_res = df.groupby("agent").mean()
    df_res['isinoctant_distal'] = df_res['isinoctant_distal'].replace(np.inf, 0)
    df_res['isinoctant_proximal'] = df_res['isinoctant_proximal'].replace(np.inf, 0)
    return df_res


def get_meanin_octant(df, coords):
    """
    Compute and returns the mean proportion of occupation of the proximal and distal beacons at a specific angle of
    rotation of the proximal beacons, and the confidence interval for both measurements
    :param df: DataFrame of logs recorded during the simulations, containing only data from test trials at a specific angle
    :type df: pandas DataFrame
    :param coords: Associate each state of the water-maze with a cartesian coordinate
    :type coords: dict
    :return type: tuple of four floats
    """

    df['isinoctant_distal'] = df.apply(lambda row: isinoctant(coords[row.state], [float(row.distal_posx), float(row.distal_posy)]), axis=1)
    df['isinoctant_proximal'] = df.apply(lambda row: isinoctant(coords[row.state], [float(row.proximal_posx), float(row.proximal_posy)]), axis=1)

    df_dist = df.groupby("agent")['isinoctant_distal'].apply(lambda x: np.sum(x)/250)
    df_dist = df_dist.replace(np.inf, 0)
    mean_distal = np.asarray(df_dist, dtype=np.float64).mean()
    yerr_distal = df_dist.std()/ np.sqrt(df_dist.shape[0])*1.96

    df_prox = df.groupby("agent")['isinoctant_proximal'].apply(lambda x: np.sum(x)/250)
    df_prox = df_prox.replace(np.inf, 0)
    mean_proximal = np.asarray(df_prox, dtype=np.float64).mean()
    yerr_proximal = df_prox.std()/ np.sqrt(df_prox.shape[0])*1.96

    return mean_distal, mean_proximal, yerr_distal, yerr_proximal


def get_values_rodrigo(results_folder, n_agents):
    """
    Returns the mean proportion of occupancy of distal landmark octant and proximal landmark octant for each angle condition.
    Also returns the confidence interval for each measurement.

    :param results_folder: path of the results folder where the data is stored
    :type results_folder: str
    :param n_agents: number of simulations stored in results_folder
    :type n_agents: int
    :return type: tuple of floats
    """
    df = create_df(results_folder, n_agents)
    coords = get_coords()

    df0 = df[df["angle"]=="0"]
    df45 = df[np.logical_or(df["angle"]=="45", df["angle"]=="-45")]
    df90 = df[np.logical_or(df["angle"]=="90", df["angle"]=="-90")]
    df135 = df[np.logical_or(df["angle"]=="135", df["angle"]=="-135")]
    df180 = df[np.logical_or(df["angle"]=="180", df["angle"]=="-180")]
    dist0, prox0, ydist0, yprox0 = get_meanin_octant(df0, coords)
    dist45, prox45, ydist45, yprox45 = get_meanin_octant(df45, coords)
    dist90, prox90, ydist90, yprox90 = get_meanin_octant(df90, coords)
    dist135, prox135, ydist135, yprox135 = get_meanin_octant(df135, coords)
    dist180, prox180, ydist180, yprox180 = get_meanin_octant(df180, coords)

    return dist0, dist45, dist90, dist135, dist180, prox0, prox45, prox90, prox135, prox180, ydist0, ydist45, ydist90, ydist135, ydist180, yprox0, yprox45, yprox90, yprox135, yprox180


def get_rodrigo_platforms_pretraining(possible_platforms):
    """
    Get a random sequence of five platforms, one for each trial of the pretraining stage.
    :param possible_platforms: the eligible platform states (there are only four)
    :type possible_platforms: int list
    :return type: int list
    """
    return  random.sample(possible_platforms, 4) + [random.choice(possible_platforms)]


def get_maze_rodrigo(maze_size, landmark_dist, edge_states):
    """
    Create the environment to simulate Rodrigo's task
    :param maze_size: diameter of the Morris pool (number of states)
    :type maze_size: int
    :param landmark_dist: number of states separating the landmark from the platform
    :type landmark_dist: int
    :param edge_states: list of eligible starting states
    :type edge_states: list of int
    """
    if maze_size == 10:
        possible_platform_states = np.array([75, 99, 90, 117])
        g = HexWaterMaze(10, landmark_dist, edge_states)
        return possible_platform_states, g


def create_path_rodrigo(n_agents, mf_allo, sr_lr, q_lr, gamma, eta, alpha1, beta1, A_alpha, A_beta, landmark_dist, HPCmode, time_limit, edge_states, lesion_HPC, lesion_DLS, dolle, inv_temp=None, inv_temp_gd=None, inv_temp_mf=None, arbi_inv_temp = None, directory=None):
    """
    Create a path to the directory where the data of all agents from a group of
    simulations with identical parameters has been stored

    Identical parameters to perform_rodrigo()
    :returns: the path of the results folder
    :return type: str
    """

    path = create_path(n_agents, mf_allo, sr_lr, q_lr, gamma, eta, alpha1, beta1, A_alpha, A_beta, landmark_dist, HPCmode, time_limit, edge_states, lesion_HPC, lesion_DLS, dolle, inv_temp=inv_temp, inv_temp_gd=inv_temp_gd, inv_temp_mf=inv_temp_mf, arbi_inv_temp = arbi_inv_temp)
    path = "rodrigo_"+path
    if directory is not None: # used for the grid-search, where the user gives a name to a hierachically higher directory
        path = directory+"/"+path
    return path
