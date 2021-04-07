from environments.HexWaterMaze import HexWaterMaze
from added import *
import os
from tqdm import tqdm
import numpy as np
from agents import CombinedAgent
import pandas as pd
from IPython.display import clear_output
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import shutil
from matplotlib.ticker import MaxNLocator
import matplotlib
from mdp import RTDP
import statsmodels.api as sm
from statsmodels.formula.api import ols



def perform_pearce(exp, maze_size, n_trials, n_sessions, n_agents, mf_allo, sr_lr, q_lr, inv_temp, gamma, eta, mpe, A_alpha, A_beta, landmark_dist, HPCmode, time_limit, edge_states, lesion_HPC, lesion_DLS, save_data = True, create_plots = True, plot=True, dland=True, show_quiv=True, show_perfs=True):

    possible_platform_states, g = get_maze(maze_size, landmark_dist, edge_states)

    # save location
    mydate = datetime.datetime.now()
    date = mydate.strftime("%d%B")
    if exp == "first_exp_pearce":
        results_folder = os.path.join("1pearce_mazesize"+str(maze_size)+"_ntrials"+str(n_trials)+"_nsessions"+str(n_sessions)+"_nagents"+str(n_agents)+"_mfallo"+str(mf_allo)+"_srlr"+str(sr_lr)+"_qlr"+str(q_lr)+"_explo"+str(inv_temp)+"_gamma"+str(gamma)+"_eta"+str(eta)+"_mpe"+str(mpe)+"_Aalpha"+str(A_alpha)+"_Abeta"+str(A_beta)+"_landmarkdist"+str(landmark_dist)+"_HPCmode"+str(HPCmode)+"_timelimit"+str(time_limit)+"_noHPC"+str(lesion_HPC)+"_noDLS"+str(lesion_DLS))
    if exp == "second_exp_pearce":
        results_folder = os.path.join("2pearce_mazesize"+str(maze_size)+"_ntrials"+str(n_trials)+"_nsessions"+str(n_sessions)+"_nagents"+str(n_agents)+"_mfallo"+str(mf_allo)+"_srlr"+str(sr_lr)+"_qlr"+str(q_lr)+"_explo"+str(inv_temp)+"_gamma"+str(gamma)+"_eta"+str(eta)+"_mpe"+str(mpe)+"_Aalpha"+str(A_alpha)+"_Abeta"+str(A_beta)+"_dland-"+str(dland)+"_landmarkdist"+str(landmark_dist)+"_HPCmode"+str(HPCmode)+"_timelimit"+str(time_limit)+"_noHPC"+str(lesion_HPC)+"_noDLS"+str(lesion_DLS))

    saved_results_folder = "saved_res/"+results_folder
    results_folder = "results/"+results_folder
    if not os.path.isdir(saved_results_folder):
        if os.path.isdir(results_folder):
            shutil.rmtree(results_folder)
        figure_folder = os.path.join(results_folder, 'figs')
        os.makedirs(results_folder)
        os.makedirs(figure_folder)
        agents = []

        for n_agent in range(n_agents):


            # set random seed
            np.random.seed(n_agent)
            #np.random.seed(np.random.randint(10000))

            # determine sequence of platform locations
            platform_sequence = None

            if exp == "first_exp_pearce":
                platform_sequence = determine_platform_seq(g, possible_platform_states, n_sessions)
            elif exp == "second_exp_pearce":
                platform_sequence = determine_platform_seq(g, possible_platform_states, n_sessions*n_trials)
            else:
                raise Exception

            # intialise agent
            agent = CombinedAgent(g, init_sr='zero',
                                  inv_temp=inv_temp,
                                  gamma=gamma,
                                  eta=eta,
                                  A_alpha = A_alpha,
                                  A_beta = A_beta,
                                  mf_allo = mf_allo,
                                  HPCmode = HPCmode,
                                  lesion_hpc = lesion_HPC,
                                  lesion_dls = lesion_DLS)

            agent.HPC.learning_rate = sr_lr
            agent.DLS.learning_rate = q_lr


            agent_results = []
            agent_ets = []
            session = 0

            total_trial_count = 0

            for ses in range(n_sessions):

                for trial in range(n_trials):
                    print("agent: "+str(n_agent)+", session: "+str(ses)+", trial: "+str(trial)+"      ", end="\r")

                    # every first trial of a session, change the platform location
                    if exp == "first_exp_pearce":
                        if trial == 0:
                            g.set_platform_state(platform_sequence[ses])
                    elif exp == "second_exp_pearce":
                        g.set_platform_state(platform_sequence[ses*n_trials+trial], dland)

                    # if mf_allo:
                    #     res = agent.one_episode_allo(time_limit, random_policy=False)
                    # else:
                    res = agent.one_episode(time_limit, random_policy=False)

                    res['trial'] = trial
                    res['escape time'] = res.time.max()
                    res['session'] = ses
                    res['total trial'] = total_trial_count
                    agent_results.append(res)

                    total_trial_count += 1

            agent_df = pd.concat(agent_results)
            agent_df['total time'] = np.arange(len(agent_df))
            agent_df['agent'] = n_agent

            agent_df.to_csv(os.path.join(results_folder, 'agent{}.csv'.format(n_agent)))
            agents.append(agent)
            if create_plots:
                # plot and save a prelim figure
                first_and_last = agent_df[np.logical_or(agent_df.trial == 0, agent_df.trial == 3)]

                fig = plt.figure()
                ax = sns.lineplot(data=first_and_last, x='session', y='escape time', hue='trial')
                plt.title('Agent n {}'.format(n_agent))
                plt.savefig(os.path.join(figure_folder, 'agent{}.png'.format(n_agent)))
                plt.close()

                res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs([agent])
                ax1, ax2, ax3, ax4, fig = plot_mean_arrows([agent], res2_mf, res2_allo, res2_sr, res2_combined)
                plt.savefig(os.path.join(figure_folder, 'agent_strats{}.png'.format(n_agent)))
                plt.close()

        clear_output()

        file_to_store = open(results_folder+"/agents.p", "wb")
        pickle.dump(agents, file_to_store)
        file_to_store.close()

        if create_plots:
            if show_perfs:
                create_plts(results_folder, n_trials, n_agents, n_sessions, plot)

            if show_quiv:
                res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs(agents, 18, 4)
                ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined, 4)
                fig.suptitle("Mean strategies after 11 sessions of training + 4 additional trials with platform on state 18", fontsize=14)
                plt.savefig(os.path.join(figure_folder, 'mean_strats_trial4.png'))

                res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs(agents, 48, 0)
                ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined, 0)
                fig.suptitle("Mean strategies after 11 sessions of training + 4 additional trials with platform on state 18 \n and final platform switch to state 48 with no further training", fontsize=14)
                plt.savefig(os.path.join(figure_folder, 'mean_strats_trial0.png'))

                plt.show()
                plt.close()



        #return agents

    else:
        saved_figure_folder = os.path.join(results_folder, 'figs')
        if create_plots:
            if show_perfs:
                create_plts(saved_results_folder, n_trials, n_agents, n_sessions, plot)

            if show_quiv:
                agents = charge_agents(saved_results_folder+"/agents.p")
                # plot the preferred direction of agents when the platform is in the state 18
                # (and the landmark supposedly in the state 0)
                res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs(agents, 18, 4)
                ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined, 4)
                fig.suptitle("Mean strategies after 11 sessions of training + 4 additional trials with platform on state 18", fontsize=14)
                #plt.savefig(os.path.join(saved_figure_folder, 'mean_strats_trial4.png'))

                res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs(agents, 48, 0)
                ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined, 0)
                fig.suptitle("Mean strategies after 11 sessions of training + 4 additional trials with platform on state 18 \n and final platform switch to state 48 with no further training", fontsize=14)
                #plt.savefig(os.path.join(saved_figure_folder, 'mean_strats_trial0.png'))

                plt.show()
                plt.close()

        #return agents


def create_plts(results_folder, n_trials, n_agents, n_sessions, plot):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17,11))
    fig.suptitle("Mean performances of the agents", fontsize=14)

    figure_folder = os.path.join(results_folder, 'figs')
    # plot averages

    all_data = []
    for ag in range(n_agents):
        df = pd.read_csv(os.path.join(results_folder, 'agent{}.csv'.format(ag)))
        summary = df.pivot_table(index=['agent', 'session', 'trial'], aggfunc='mean')
        all_data.append(summary)

    df = pd.concat(all_data)
    df['platform location'] = df['platform'].astype('category')
    df.to_csv(os.path.join(results_folder, 'summary.csv'))

    #perform two-way ANOVA
    df["escape_time"] = df["escape time"]

    model = ols('escape_time ~ C(session) + C(trial) + C(session):C(trial)', data=df.reset_index()).fit()
    print("Two-way ANOVA on trial and session")
    print()
    print(sm.stats.anova_lm(model, typ=2))
    ###############PLOTS#################

    # plot the escape time per session for trials 1 and 4
    # plt.figure()
    # first_last = df.loc[(list(range(n_agents)), list(range(n_sessions)), (0, 3))]
    # sns.lineplot(data=first_last.reset_index(), x='session', y='escape time', hue='trial', ci=None, ax=ax1).set_title("Escape time as a function of sessions and trials")
    # ax1.set(ylabel="Escape time (steps number)")
    # plt.savefig(os.path.join(figure_folder, 'escape_time_firstlast.png'))
    # plt.close()

    # plot the escape time per session for all trials
    plt.figure()

    sns.lineplot(data=df.reset_index(), x=df.reset_index()['session']+1, y='escape time', hue='trial', ci=None).set_title("Escape time as a function of sessions and trials")
    #plt.set(ylabel="Escape time (steps number)")
    plt.savefig(os.path.join(figure_folder, 'escape_time.png'))

    sns.lineplot(data=df.reset_index(), x=df.reset_index()['session']+1, y='escape time', hue='trial', ci=None, ax=ax1).set_title("Escape time as a function of sessions and trials")

    ax1.set(ylabel="Escape time (steps number)")
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    plt.close()


    # plot the escape time per trial with vlines indicating new sessions
    plt.figure()

    sns.lineplot(data=df, x='total trial', y='escape time').set_title("Escape time per trial")
    for i in range(n_sessions*n_trials):
        if (i % n_trials) == 0:
            plt.axvline(x=i, ymin=0, ymax=1, linewidth=1, color='r', alpha=.3)


    sns.lineplot(data=df, x='total trial', y='escape time',ax=ax2).set_title("Escape time per trial")
    for i in range(n_sessions*n_trials):
        if (i % n_trials) == 0:
            ax2.axvline(x=i, ymin=0, ymax=1, linewidth=1, color='r', alpha=.3)
    ax2.set(ylabel="Escape time (steps number)")
    try:
        plt.savefig(os.path.join(figure_folder, 'escape_time_pertrial.png'))
    except Exception:
        pass

    #plt.set(ylabel="Escape time (steps number)")

    plt.close()


    # plot the sr model weight on the decision process as a function of sessions and trials
    plt.figure()
    sns.lineplot(data=df.reset_index(), x=df.reset_index()['session']+1, y='P(SR)', hue='trial', ci=None, ax=ax3).set_title("Evolution of P(SR) across sessions and trials")
    sns.lineplot(data=df.reset_index(), x=df.reset_index()['session']+1, y='P(SR)', hue='trial', ci=None).set_title("Evolution of P(SR) across sessions and trials")
    plt.savefig(os.path.join(figure_folder, 'p_sr.png'))
    ax3.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    plt.close()

    # # plot the escape time per trial with vlines indicating new sessions
    # plt.figure()
    # for i in range(n_sessions*n_trials):
    #     if (i % n_trials) == 0:
    #         plt.axvline(x=i, ymin=0, ymax=1, linewidth=1, color='r', alpha=.3)
    # sns.lineplot(data=df, x='total trial', y='P(SR)')
    # plt.savefig(os.path.join(figure_folder, 'p_sr_pertrial.png'))
    # plt.close()


    # Plot the average escape time per platform
    plt.figure()
    sns.barplot(data=df.loc[(list(range(n_agents)), list(range(n_sessions)), 0)], x='platform location', y='escape time', ax=ax4).set_title("Escape time per platform")
    ax4.set(ylabel="Escape time (steps number)")

    sns.barplot(data=df.loc[(list(range(n_agents)), list(range(n_sessions)), 0)], x='platform location', y='escape time').set_title("Escape time per platform")
    #plt.set(ylabel="Escape time (steps number)")
    plt.savefig(os.path.join(figure_folder, 'et_per_platform.png'))

    plt.close()



def plot_pearce(exp, maze_size, n_trials, n_sessions, n_agents, mf_allo, sr_lr, q_lr, inv_temp, gamma, eta, mpe, A_alpha, A_beta, landmark_dist, HPCmode, time_limit, edge_states):

    # Path defining
    if exp == "first_exp_pearce":
        results_folder_normal = os.path.join("1pearce_mazesize"+str(maze_size)+"_ntrials"+str(n_trials)+"_nsessions"+str(n_sessions)+"_nagents"+str(n_agents)+"_mfallo"+str(mf_allo)+"_srlr"+str(sr_lr)+"_qlr"+str(q_lr)+"_explo"+str(inv_temp)+"_gamma"+str(gamma)+"_eta"+str(eta)+"_mpe"+str(mpe)+"_Aalpha"+str(A_alpha)+"_Abeta"+str(A_beta)+"_landmarkdist"+str(landmark_dist)+"_HPCmode"+str(HPCmode)+"_timelimit"+str(time_limit)+"_noHPC"+str(False)+"_noDLS"+str(False))
        results_folder_lesion = os.path.join("1pearce_mazesize"+str(maze_size)+"_ntrials"+str(n_trials)+"_nsessions"+str(n_sessions)+"_nagents"+str(n_agents)+"_mfallo"+str(mf_allo)+"_srlr"+str(sr_lr)+"_qlr"+str(q_lr)+"_explo"+str(inv_temp)+"_gamma"+str(gamma)+"_eta"+str(eta)+"_mpe"+str(mpe)+"_Aalpha"+str(A_alpha)+"_Abeta"+str(A_beta)+"_landmarkdist"+str(landmark_dist)+"_HPCmode"+str(HPCmode)+"_timelimit"+str(time_limit)+"_noHPC"+str(True)+"_noDLS"+str(False))

    # deprecated
    # if exp == "second_exp_pearce":
    #     results_folder_normal = os.path.join("2pearce_mazesize"+str(maze_size)+"_ntrials"+str(n_trials)+"_nsessions"+str(n_sessions)+"_nagents"+str(n_agents)+"_mfallo"+str(mf_allo)+"_srlr"+str(sr_lr)+"_qlr"+str(q_lr)+"_explo"+str(inv_temp)+"_gamma"+str(gamma)+"_eta"+str(eta)+"_mpe"+str(mpe)+"_Aalpha"+str(A_alpha)+"_Abeta"+str(A_beta)+"_dland-"+str(dland)+"_landmarkdist"+str(landmark_dist)+"_HPCmode"+str(HPCmode)+"_timelimit"+str(time_limit)+"_lesionHPC"+str(False))
    #     results_folder_lesion = os.path.join("2pearce_mazesize"+str(maze_size)+"_ntrials"+str(n_trials)+"_nsessions"+str(n_sessions)+"_nagents"+str(n_agents)+"_mfallo"+str(mf_allo)+"_srlr"+str(sr_lr)+"_qlr"+str(q_lr)+"_explo"+str(inv_temp)+"_gamma"+str(gamma)+"_eta"+str(eta)+"_mpe"+str(mpe)+"_Aalpha"+str(A_alpha)+"_Abeta"+str(A_beta)+"_dland-"+str(dland)+"_landmarkdist"+str(landmark_dist)+"_HPCmode"+str(HPCmode)+"_timelimit"+str(time_limit)+"_lesionHPC"+str(True))

    if os.path.isfile("results/"+results_folder_normal) and os.path.isfile("results/"+results_folder_lesion):
        saved_results_folder_normal = "results/"+results_folder_normal
        saved_results_folder_lesion = "results/"+results_folder_lesion
    else:
        saved_results_folder_normal = "saved_res/"+results_folder_normal
        saved_results_folder_lesion = "saved_res/"+results_folder_lesion

    # data retrieving for normal rats
    all_data = []
    for ag in range(n_agents):
        df = pd.read_csv(os.path.join(saved_results_folder_normal, 'agent{}.csv'.format(ag)))
        summary = df.pivot_table(index=['agent', 'session', 'trial'], aggfunc='mean')
        all_data.append(summary)

    df = pd.concat(all_data)
    df['platform location'] = df['platform'].astype('category')
    df.to_csv(os.path.join(saved_results_folder_normal, 'summary.csv'))

    # data retrieving for HPC lesioned rats
    all_data = []
    for ag in range(n_agents):
        df_lesion = pd.read_csv(os.path.join(saved_results_folder_lesion, 'agent{}.csv'.format(ag)))
        summary = df_lesion.pivot_table(index=['agent', 'session', 'trial'], aggfunc='mean')
        all_data.append(summary)

    df_lesion = pd.concat(all_data)
    df_lesion['platform location'] = df_lesion['platform'].astype('category')
    df_lesion.to_csv(os.path.join(saved_results_folder_lesion, 'summary.csv'))

    # plotting
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    fig.suptitle("Escape time as a function of sessions and trials", fontsize=20, y = 1.1)

    # first two results from Pearce and Geerts
    axs[0].imshow(mpimg.imread("results_pearce.jpg"),aspect="auto")
    axs[1].imshow(mpimg.imread("geerts_exp_1.jpg"),aspect="auto")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[0].set_title("Pearce results")
    axs[1].set_title("Geerts results")
    axs[0].set_frame_on(False)
    axs[1].set_frame_on(False)

    # Our results
    pal = sns.color_palette()

    df_lesion = df_lesion.reset_index()
    df_lesion_trial1 = df_lesion[df_lesion['trial']==0]
    sns.lineplot(data=df_lesion_trial1, x=df_lesion_trial1['session']+1, y='escape time', label="HPC lesion - trial 1", linewidth = 5, color=pal[1], ci=None)

    df_lesion = df_lesion.reset_index()
    df_lesion_trial4 = df_lesion[df_lesion['trial']==3]
    sns.lineplot(data=df_lesion_trial4, x=df_lesion_trial4['session']+1, y='escape time',  label="HPC lesion - trial 4", linewidth = 5, color=pal[1],  style=False, dashes=[(2,1)], ci=None)

    df = df.reset_index()
    df_trial1 = df[df['trial']==0]
    sns.lineplot(data=df_trial1, x=df_trial1['session']+1, y='escape time', label="Control - trial 1", linewidth = 5, color=pal[0], ci=None)

    df = df.reset_index()
    df_trial4 = df[df['trial']==3]
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

    plt.show()
    plt.close()
