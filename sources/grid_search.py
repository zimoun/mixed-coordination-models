from rodrigo_protocol import perform_group_rodrigo, create_path_rodrigo, get_values_rodrigo, get_mean_occupation_octant, create_df
from pearce_protocol import perform_group_pearce, create_path_main_pearce
from environments.HexWaterMaze import EnvironmentParams
from utils import isinoctant, get_MSLE, get_coords
from agents.agent import AgentsParams

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from IPython.display import display, HTML
from statsmodels.formula.api import ols
from scipy.stats import loguniform
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib
import os.path
import random
import time
import os
import re

from patsy.contrasts import Helmert
from scipy.stats import ttest_ind
from scipy import stats


def random_grid_search(directory, expe, n_datapoints, mf_allo, dolle, HPCmode, srlr=None, range_srlr=None, qlr=None, range_qlr=None, eta=None, range_eta=None, gamma=None, range_gamma=None, inv_temp=None, range_inv_temp=None):
    """
        Perform a random grid-search across several dimensions (parameters).
        The goal of a random grid search is to find the optimal combination of parameters.
        Here optimality is defined as producing data generating the lowest mean square
        error (MSE) when compared to Rodrigo's and Pearce's experimental data.
        In this function, only the simulated performance of agents on either Pearce's or Rodrigo's
        protocol is measured, it is not sufficient to infer the best set of parameters,
        the data has to be clusterized in another function for the MSE to be computed.

        :param directory: Where the simulation data will be stored
        :type directory: str
        :param expe: Specify whether Pearce's or Rodrigo experimental protocol must be used
        :type expe: str (values -> rodrigo or main_pearce)
        :param n_datapoints: Number of datapoints (parameter's combinations) to generate and test
        :type n_datapoints: int
        :param mf_allo: Whether the simulated model will be using an allocentric or egocentric MF frame of reference
        :type mf_allo: boolean
        :param dolle: Whether the model will be implementing Dolle's arbitrator (associative learning) or Geerts (incertainty based)
        :type dolle: boolean
        :param HPCmode: Whether the model hipocampus will be implemented using a model-based (MB) algorithm or the Successor-Representation (SR) algorithm
        :type HPCmode: str (values -> SR or MB)
        :param srlr: Block the SR learning rate to a given value if specified (warning : either srlr or range_srlr must be specified, not both)
        :type srlr: float
        :param range_srlr: The range within which to explore the SR learning rate influence on the model performance (only relevant for Geerts model)
        :type srlr: float tuple
        :param qlr: Block the MF learning rate to a given value if specified (warning : either qlr or range_qlr must be specified, not both)
        :type qlr: float
        :param range_qlr: The range within which to explore the MF learning rate influence on the model performance
        :type qlr: float tuple
        :param eta: Block the arbitrator learning rate to a given value if specified (warning : either eta or range_eta must be specified, not both)
        :type eta: float
        :param range_eta: The range within which to explore the arbitrator learning rate influence on the model performance
        :type eta: float tuple
        :param gamma: Block the reward discount factor to a given value if specified (warning : either gamma or range_gamma must be specified, not both)
        :type gamma: float
        :param range_gamma: The range within which to explore the reward discount factor influence on the model performance
        :type gamma: float tuple
        :param range_inv_temp: Block the softmax function inverse temperature to a given value if specified (warning : either inv_temp or range_inv_temp must be specified, not both)
        :type range_inv_temp: int
        :param range_inv_temp: The range within which to explore the softmax function inverse temperature influence on the model performance
        :type range_inv_temp: int tuple

        :returns: Nothing, results are stored in different files in directory
    """

    # returns a random value for parameter name_param if range_param is not None, returns the value param otherwise.
    # random values are selected using a loguniform distribution to ensure that lower intervals are sampled as much as others
    def get_real(param_value, range_param, name_param):
        if param_value is None and range_param is None:
            raise Exception("Either "+name_param+" or range_"+name_param+" must be defined")
        if param_value is not None and range_param is not None:
            raise Exception("Either "+name_param+" or range_"+name_param+" must be defined, not both")

        if range_param is not None:
            res = np.around(loguniform.rvs(range_param[0], range_param[1], size=1), 3)[0]
        # if the user rather want the param to be blocked on a given value
        else:
            res = param_value
        return res

    # dataframes that will associate each combination of parameters with the model performances (will be stored as CSV)
    res_df_tmp_pearce = pd.DataFrame(columns=['srlr', 'qlr', 'gamma', 'eta', 'inv_temp', "cont1", "cont4", "hpc1", "hpc4"])
    res_df_tmp_pearce_noPFC = pd.DataFrame(columns=['srlr', 'qlr', 'gamma', 'eta', 'inv_temp', "cont1", "cont4", "hpc1", "hpc4"])
    res_df_tmp_rodrigo = pd.DataFrame(columns=['srlr', 'qlr', 'gamma', 'eta', 'inv_temp', "0dist", "45dist", "90dist", "135dist", "180dist", "0prox", "45prox", "90prox", "135prox", "180prox"])

    # if the files already exist (part of the whole grid-search is already completed)
    try:
        os.mkdir("../results/"+directory)
        print("Directory " , "../results/"+directory ,  " Created ")
    except FileExistsError:
        print("Directory " , "../results/"+directory ,  " already exists")

    try:
        if expe == "main_pearce":
            res_df_pearce = pd.read_csv("../results/"+directory+"/mean_square_pearce.csv")
            res_df_pearce_noPFC = pd.read_csv("../results/"+directory+"/mean_square_pearce_noPFC.csv")
        if expe == "rodrigo":
            res_df_rodrigo = pd.read_csv("../results/"+directory+"/mean_square_rodrigo.csv")
    except:
        res_df_pearce = res_df_tmp_pearce
        res_df_pearce_noPFC = res_df_tmp_pearce_noPFC
        res_df_rodrigo = res_df_tmp_rodrigo
        if expe == "main_pearce":
            res_df_pearce.to_csv("../results/"+directory+"/mean_square_pearce.csv",index=False)
            res_df_pearce_noPFC.to_csv("../results/"+directory+"/mean_square_pearce_noPFC.csv",index=False)
        if expe == "rodrigo":
            res_df_rodrigo.to_csv("../results/"+directory+"/mean_square_rodrigo.csv",index=False)

    try:
        # if a dataframe of combinations of random parameters has already been generated
        params_df = pd.read_csv("../results/"+directory+"/params_df.csv")
    except:
        params_df = pd.DataFrame(columns = ['srlr', 'qlr', 'gamma', 'eta', 'inv_temp'])
        # creates several combinations of random parameters. Each combination will be used
        # to simulate a single agent performance on Pearce's and/or on Rodrigo's task
        for i in range(n_datapoints):
            srlr_tmp = get_real(srlr, range_srlr, "srlr")
            qlr_tmp = get_real(qlr, range_qlr, "qlr")
            gamma_tmp = get_real(gamma, range_gamma, "gamma")
            eta_tmp = get_real(eta, range_eta, "eta")
            inv_temp_tmp = int(get_real(inv_temp, range_inv_temp, "inv_temp"))
            new_row = {'srlr':srlr_tmp, "qlr":qlr_tmp, 'gamma':gamma_tmp, 'eta':eta_tmp, 'inv_temp':inv_temp_tmp}
            params_df = params_df.append(new_row, ignore_index=True)
        params_df.to_csv("../results/"+directory+"/params_df.csv",index=False)

    # used to infer the grid-search duration
    deb = time.time()
    # iterates over each parameter combination
    cpt = 0

    print("Beginning simulations...", end="\r")

    if expe == "main_pearce":
        res_df = res_df_pearce
    elif expe == "rodrigo":
        res_df = res_df_rodrigo
    else:
        raise Exception("experiment must be either rodrigo or pearce")

    while len(res_df) < n_datapoints:

        i = len(res_df)

        srlr_real = params_df.iloc[i].srlr
        qlr_real = params_df.iloc[i].qlr
        gamma_real = params_df.iloc[i].gamma
        eta_real = params_df.iloc[i].eta
        inv_temp_real = int(params_df.iloc[i].inv_temp)

        try:
            if expe == "main_pearce":

                env_params = EnvironmentParams()
                env_params.maze_size = 10
                env_params.n_sessions = 11
                env_params.n_trials = 4
                env_params.n_agents = 1
                env_params.init_sr = "zero"
                env_params.landmark_dist = 4
                env_params.time_limit = 500
                env_params.starting_states = [243,230,270,257]

                ag_params = AgentsParams()
                ag_params.mf_allo = mf_allo
                ag_params.hpc_lr = srlr_real
                ag_params.q_lr = qlr_real
                ag_params.arbi_learning_rate = eta_real
                ag_params.inv_temp = inv_temp_real
                ag_params.inv_temp_gd = inv_temp_real
                ag_params.inv_temp_mf = inv_temp_real
                ag_params.arbi_inv_temp = inv_temp_real
                ag_params.gamma = gamma_real
                ag_params.eta = eta_real # reliability learning rate
                ag_params.alpha1 = 0.01
                ag_params.beta1 = 0.1
                ag_params.A_alpha = 3.2 # Steepness of transition curve MF to SR
                ag_params.A_beta = 1.1 # Steepness of transition curve SR to MF
                ag_params.HPCmode = HPCmode
                ag_params.lesion_HPC = False
                ag_params.lesion_DLS = False
                ag_params.dolle = dolle
                ag_params.lesion_PFC= False

                # run the simulation (control)
                ag_params.lesion_HPC = False
                ag_params.lesion_DLS = False
                perform_group_pearce(env_params, ag_params, show_plots=False, save_plots=False, save_agents=False, directory = directory+"/pearce_control", verbose = False)
                results_folder_normal = create_path_main_pearce(env_params, ag_params, directory=directory+"/pearce_control")

                # run the simulation (lesioned)
                ag_params.lesion_HPC = True
                perform_group_pearce(env_params, ag_params, show_plots=False, save_plots=False, save_agents=False, directory = directory+"/pearce_lesion", verbose = False)
                results_folder_lesion = create_path_main_pearce(env_params, ag_params, directory=directory+"/pearce_lesion")

                # retrieve the simulation results as a simplified dataframe
                df_normal = create_df(results_folder_normal, 1)
                df_lesion = create_df(results_folder_lesion, 1)

                # if the model is an implementation of Dolle's algorithm
                # an additional simulation is performed, with a deactivated arbitrator on
                # HPC-lesioned condition (to systematically reject the HPC Q-values)
                if dolle:
                    ag_params.lesion_PFC=True
                    ag_params.lesion_HPC=True
                    perform_group_pearce(show_plots=False, save_plots=False, save_agents=False, directory = directory+"/pearce_noPFC", verbose = False)
                    results_folder_lesion_noHPC = create_path_main_pearce(env_params, ag_params, directory=directory+"/pearce_noPFC")
                    df_lesion_noHPC = create_df(results_folder_lesion_noHPC, 1)

                df = df_normal.reset_index()
                # get the control agents mean escape time for each session for trial 0 and trial 3
                control_1 = df[df["trial"]==0].groupby("session")["escape time"].mean().to_numpy()
                control_4 = df[df["trial"]==3].groupby("session")["escape time"].mean().to_numpy()

                df = df_lesion.reset_index()
                hpc_1 = df[df["trial"]==0].groupby("session")["escape time"].mean().to_numpy()
                hpc_4 = df[df["trial"]==3].groupby("session")["escape time"].mean().to_numpy()

                if dolle:
                    df = df_lesion_noHPC.reset_index()
                    hpc_1_noHPC = df[df["trial"]==0].groupby("session")["escape time"].mean().to_numpy()
                    hpc_4_noHPC = df[df["trial"]==3].groupby("session")["escape time"].mean().to_numpy()

                del df, df_normal, df_lesion

                # associates simulation mean performances to parameters combination
                res_df.loc[i] = [srlr_real, qlr_real, gamma_real, eta_real, inv_temp_real, control_1, control_4, hpc_1, hpc_4]

                if dolle:
                    del df_lesion_noHPC
                    res_df_pearce_noPFC.loc[i] = [srlr_real, qlr_real, gamma_real, eta_real, inv_temp_real, control_1, control_4, hpc_1_noHPC, hpc_4_noHPC]

                # program crashes otherwise
                del control_1, control_4
                del hpc_1, hpc_4

                if dolle:
                    del hpc_1_noHPC, hpc_4_noHPC
                    res_df_pearce_noPFC.to_csv("../results/"+directory+"/mean_square_pearce_noPFC.csv",index=False)

                # results saved at each iteration, program can be crashed and rerun at mid-simulation
                # with no loss of information
                res_df.to_csv("../results/"+directory+"/mean_square_pearce.csv",index=False)


            if expe == "rodrigo":
                env_params = EnvironmentParams()
                env_params.maze_size = 10
                env_params.n_sessions = 11
                env_params.n_trials = 4
                env_params.n_agents = 1
                env_params.init_sr = "zero"
                env_params.landmark_dist = 0
                env_params.time_limit = 500
                env_params.starting_states = [243,230,270,257]

                ag_params = AgentsParams()
                ag_params.mf_allo = mf_allo
                ag_params.hpc_lr = srlr_real
                ag_params.q_lr = qlr_real
                ag_params.arbi_learning_rate = eta_real
                ag_params.inv_temp = inv_temp_real
                ag_params.inv_temp_gd = inv_temp_real
                ag_params.inv_temp_mf = inv_temp_real
                ag_params.arbi_inv_temp = inv_temp_real
                ag_params.gamma = gamma_real
                ag_params.eta = eta_real # reliability learning rate
                ag_params.alpha1 = 0.01
                ag_params.beta1 = 0.1
                ag_params.A_alpha = 3.2 # Steepness of transition curve MF to SR
                ag_params.A_beta = 1.1 # Steepness of transition curve SR to MF
                ag_params.HPCmode = HPCmode
                ag_params.lesion_HPC = False
                ag_params.lesion_DLS = False
                ag_params.dolle = dolle
                ag_params.lesion_PFC= False

                # run the simulation
                perform_group_rodrigo(env_params, ag_params, directory=directory+"/rodrigo", show_plots=False, save_plots=False, save_agents=False, verbose=False)
                results_folder = create_path_rodrigo(env_params, ag_params, directory=directory+"/rodrigo")
                # retrieve the mean occupation time of different octants in the raw results logs
                (dist0, dist45, dist90, dist135, dist180, prox0, prox45, prox90, prox135, prox180, ydist0, ydist45, ydist90, ydist135, ydist180, yprox0, yprox45, yprox90, yprox135, yprox180) = get_values_rodrigo(results_folder, 1)
                # associates simulation mean performances to parameters combination
                res_df.loc[i] = [srlr_real, qlr_real, gamma_real, eta_real, inv_temp_real, dist0, dist45, dist90, dist135, dist180, prox0, prox45, prox90, prox135, prox180]
                # results saved at each iteration, program can be crashed and rerun at mid-simulation
                # with no loss of information
                res_df.to_csv("../results/"+directory+"/mean_square_rodrigo.csv",index=False)

        # or the program if unable to stop
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            # the program might crash mainly because of an overflow error when
            # computing the softmax function at decision time
            # the user must verify that not a too high number of errors have occured,
            # otherwise the number of datapoints generated might be too low to produce
            # a clear performance gradient, or the distribution of the datapoints might have been corrupted
            # association (between parametes value and overflow tendancies)
            if expe == "main_pearce":
                res_df.loc[i] = ["error", "error", "error", "error", "error", "error", "error", "error", "error"]
            if expe == "rodrigo":
                res_df.loc[i] = ["error", "error", "error", "error", "error", "error", "error", "error", "error", "error", "error", "error", "error", "error", "error"]
            print("An error occured: ", e)

        end = time.time()
        cpt += 1
        remaining_time = ((end-deb)/cpt)*(n_datapoints-i)

        print("Data point: ", i, "/",n_datapoints, ", estimated remaining time: ",time.strftime('%d:%H:%M:%S', time.gmtime(remaining_time)), end="\r")


def compute_clusters_perfs(directory, experimental_data_pearce=None, experimental_data_rodrigo=None, relative=True, verbose=False, noPFC=False):
    """
        Clusterizes the agents into local groups to get the group mean performances.
        This allows to dramatically reduce noise present in the raw data, where each
        datapoint is associated to only one agent performance.
        Clusterization of different set of parameters allows to test both a higher
        number and a larger range of parameters, compared to more classical grid-search methods.
        The clustering is done in a 4-dimensional euclidian space, where each parameter is a different dimension
        Clusters are formed by including the 100 closest datapoints from the origin, using 4d euclidian distance.
        To reduce the bias originating from the distribution of the 100 selected datapoints in space,
        the mean performance is associated to the centroid of the cluster, not with the origin parameters.
        Pearce's and Rodrigo's data can be processed at the same time or separately in different function calls,
        in the former case, Rodrigo's and Pearce's data are merged and an additional MSE value is
        processed for each set of parameters (product Rodrigo*Pearce). Each set of
        parameter is used twice, once to measure performances on Pearce's protocol and another on Rodrigo's protocol.

        :param directory: Where the processed data will be stored
        :type directory: str
        :param size_cluster: (removed) Number of datapoints to be included in each cluster (more -> less noise but less spatial precision)
        :type size_cluster: int
        :param experimental_data_pearce: The original data obtained in Pearce's main experiment, to compare to our simulation data, no processing on Pearce's data if set to None
        :type experimental_data_pearce: dict of {str:list}
        :param experimental_data_rodrigo: The original data obtained in Rodrigo's experiment, to compare to our simulation data, no processing on Rodrigo's data if set to None
        :type experimental_data_rodrigo: dict of {str:list}
        :param relative: Whether the MSE is computed using absolute or relative values for Rodrigo's experiment
        :type relative: boolean
        :param verbose: Print the beginning and ending of both Pearce's and Rodrigo's Dataframe to ensure they are aligned before fusion (development purpose)
        :type verbose: boolean

        :returns: Nothing, results are stored in different files in directory
    """
    # RAW DATA RETRIEVING

    size_cluster = 100 # was a parameter in a previous version

    if experimental_data_pearce is None and experimental_data_rodrigo is None:
        raise Exception("At least one set of experimental data must be provided")

    if experimental_data_pearce is not None:
        if noPFC:
            df_pearce = pd.read_csv("../results/"+directory+"/mean_square_pearce_noPFC.csv")
        else:
            df_pearce = pd.read_csv("../results/"+directory+"/mean_square_pearce.csv")
    if experimental_data_rodrigo is not None:
        df_rodrigo = pd.read_csv("../results/"+directory+"/mean_square_rodrigo.csv")

    # sets of parameters from Pearce's and Rodrigo's must be identical and aligned
    if verbose:
        print("Check Pearce's and Rodrigo's data alignment\n")
        print("Pearce data: ")
        print(df_pearce[["qlr", "eta", "gamma"]].iloc[np.r_[0:3, -3:0]], "\n")
        print("Rodrigo data: ")
        print(df_rodrigo[["qlr", "eta", "gamma"]].iloc[np.r_[0:3, -3:0]])

    if experimental_data_pearce is None:
        df_main = df_rodrigo
        # remove any lines containing an error (like hpc4, all columns are set to "error")
        df_main = df_main[df_main["45dist"] != "error"]
    elif experimental_data_rodrigo is None:
        df_main = df_pearce
        df_main = df_main[df_main["hpc4"] != "error"]
    else:
        df_main = pd.concat([df_pearce, df_rodrigo], axis=1)
        # as df1 and df2 contains the same set of parameters
        df_main = df_main.loc[:,~df_main.columns.duplicated()]
        df_main = df_main[np.logical_and(df_main["hpc4"] != "error", df_main["45dist"] != "error")]

    df_main = df_main.reset_index()
    print("Using "+ str(len(df_main)) + " remaining datapoints")

    if relative:
        label = "relative"
    else:
        label = "absolute"

    if experimental_data_pearce is not None:
        # saving the agent Mean Square Error and different conditions MSE
        df_main["se_cluster_"+str(size_cluster)+"_pearce_mean"] = None
        df_main["se_cluster_"+str(size_cluster)+"_pearce_control1"] = None
        df_main["se_cluster_"+str(size_cluster)+"_pearce_control4"] = None
        df_main["se_cluster_"+str(size_cluster)+"_pearce_hpc1"] = None
        df_main["se_cluster_"+str(size_cluster)+"_pearce_hpc4"] = None
    if experimental_data_rodrigo is not None:
        # saving the agent Mean Square Error and different conditions MSE
        df_main["se_cluster_"+str(size_cluster)+"_rodrigo_mean_"+label] = None
        df_main["se_cluster_"+str(size_cluster)+"_rodrigo_"+label] = None
    if experimental_data_pearce is not None and experimental_data_rodrigo is not None:
        # product between Pearce and Rodrigo SE
        df_main["se_cluster_"+str(size_cluster)+"_product_mean_"+label] = None

    # ACTUAL CLUSTERING AND MSLE PROCESSING

    df = df_main
    # list of 5d euclidian distances between rows (each parameters set is a coordinate)
    eu_dists = get_euclidian_distances(df)

    # for each parameter set (coordinate / datapoint)
    for i in df.index:
        print(i, end="\r")
        # creates a local cluster of the 100 closest points near datapoint i
        closests_ind = get_closest_datapoints(eu_dists, i)[0:size_cluster]

        if experimental_data_pearce is not None:

            # get mean performances of real rats for trial and session conditions
            real_data = experimental_data_pearce
            real_data = [real_data["cont1"], real_data["cont4"], real_data["hip1"], real_data["hip4"]]

            # get mean performances of a cluster of 100 simulated agents for trial and session conditions
            cluster_mean = lambda df, name_col, indices : df.iloc[indices][name_col].apply(lambda s : get_array_from_str(s)).mean()

            simu_c1 = cluster_mean(df, "cont1", closests_ind)
            simu_c4 = cluster_mean(df, "cont4", closests_ind)
            simu_h1 = cluster_mean(df, "hpc1", closests_ind)
            simu_h4 = cluster_mean(df, "hpc4", closests_ind)
            simu_data = [simu_c1, simu_c4, simu_h1, simu_h4]

            se_pearce = get_MSLE(real_data, simu_data, relative=True)

            df["se_cluster_"+str(size_cluster)+"_pearce_mean"].iloc[i] = se_pearce
            df["se_cluster_"+str(size_cluster)+"_pearce_control1"].iloc[i] = simu_c1
            df["se_cluster_"+str(size_cluster)+"_pearce_control4"].iloc[i] = simu_c4
            df["se_cluster_"+str(size_cluster)+"_pearce_hpc1"].iloc[i] = simu_h1
            df["se_cluster_"+str(size_cluster)+"_pearce_hpc4"].iloc[i] = simu_h4

        if experimental_data_rodrigo is not None:

            # get mean occupation of octants of real rats for both beacon conditions
            real_data = experimental_data_rodrigo["dist"] + experimental_data_rodrigo["prox"]

            # get mean occupation of octants of a cluster of 100 simulated rats for both beacon conditions
            cluster_mean = lambda df, name_col, indices : df.iloc[indices][name_col].apply(lambda s : float(s)).mean()
            vars = ["0dist", "45dist", "90dist", "135dist", "180dist", "0prox", "45prox", "90prox", "135prox", "180prox"]
            simu_data = [cluster_mean(df, var_name, closests_ind) for var_name in vars]

            se_rodrigo = get_MSLE([real_data], [simu_data], relative)

            df["se_cluster_"+str(size_cluster)+"_rodrigo_mean_"+label].iloc[i] = se_rodrigo
            df["se_cluster_"+str(size_cluster)+"_rodrigo_"+label].iloc[i] = simu_data

        if experimental_data_pearce is not None and experimental_data_rodrigo is not None:
            # product between Pearce's and Rodrigo's SE
            df["se_cluster_"+str(size_cluster)+"_product_mean_"+label].iloc[i] = se_pearce + se_rodrigo

    # if some data has already been processed, charges it to expand it
    if os.path.isfile("../results/"+directory+"/mean_square_processed.csv"):
        tmp = pd.read_csv("../results/"+directory+"/mean_square_processed.csv")
        df = pd.concat([df, tmp], axis=1)
        df = df.loc[:,~df.columns.duplicated()]

    df.to_csv("../results/"+directory+"/mean_square_processed.csv", index=False)


def get_euclidian_distances(res_df):
    """
        Computes a matrix of the 5-dimensional euclidian distances between each set of parameters used for the grid-search

        :param res_df: The dataframe where each row represent a set of parameter, and a coordinate in space
        :type res_df: pandas.DataFrame

        :returns: A 2D numpy array stating for each set of parameter, its distance to other datapoints.
    """
    scaler = MinMaxScaler()
    # all dimensions are normalized to a -1,1 scale, to allow to compare distances between dimensions
    v = lambda k : scaler.fit_transform(np.log(res_df[k].astype(float)).values.reshape(-1,1)).flatten()
    matrix = np.transpose(np.array([v("srlr"), v("qlr"),v("gamma"), v("inv_temp"),v("eta")]))
    return euclidean_distances(matrix, matrix)


def perform_statical_analyses_pearce(directory):
    """
        Performs a series of two-way ANOVAs to test whether the escape time variable significantly
        decreases across both sessions and trials, for different clusters of agents.
        Datapoints (clusters centroid) that do not validate the test are tagged as such.
        They will be later removed from the list of potential candidates for best set of parameter.

        :param directory: The directory in which all files related to the grid-search can be found
        :type directory: str

        :returns: Nothing, the results are stored in a CSV file in directory
    """
    res_df = pd.read_csv("../results/"+directory+"/mean_square_processed.csv")
    # list of 5D euclidian distances between rows (each parameters set is a coordinate)
    eu_dists = get_euclidian_distances(res_df)
    # 100 closests peers for each candidate cluster
    closests = [get_closest_datapoints(eu_dists, i) for i in range(len(res_df))]

    # will contain the data of all agents
    df_analysis = pd.DataFrame()
    df_analysis_les = pd.DataFrame()

    print("Retrieving all agents data")
    for agent_ind in range(len(res_df)):
        print("agent: "+ str(agent_ind), end="\r")
        try:
            # if model is Geerts
            one_agent_df = pd.read_csv("../results/"+directory+"/pearce_control/pearce_104111False"+str(res_df.iloc[agent_ind].srlr)+str(res_df.iloc[agent_ind].qlr)+str(int(res_df.iloc[agent_ind].inv_temp))+str(res_df.iloc[agent_ind].gamma)+str(res_df.iloc[agent_ind].eta)+"0.010.13.21.14SR500FalseFalseFalse/agent0.csv")
            one_agent_df_les = pd.read_csv("../results/"+directory+"/pearce_lesion/pearce_104111False"+str(res_df.iloc[agent_ind].srlr)+str(res_df.iloc[agent_ind].qlr)+str(int(res_df.iloc[agent_ind].inv_temp))+str(res_df.iloc[agent_ind].gamma)+str(res_df.iloc[agent_ind].eta)+"0.010.13.21.14SR500TrueFalseFalse/agent0.csv")
        except:
            # if model is Dolle
            one_agent_df = pd.read_csv("../results/"+directory+"/pearce_control/pearce_104111True"+str(res_df.iloc[agent_ind].srlr)+str(res_df.iloc[agent_ind].qlr)+str(int(res_df.iloc[agent_ind].inv_temp))+str(int(res_df.iloc[agent_ind].inv_temp))+str(int(res_df.iloc[agent_ind].inv_temp))+str(res_df.iloc[agent_ind].gamma)+str(res_df.iloc[agent_ind].eta)+"0.010.13.21.14MB500FalseFalseTrue/agent0.csv")
            one_agent_df_les = pd.read_csv("../results/"+directory+"/pearce_lesion/pearce_104111True"+str(res_df.iloc[agent_ind].srlr)+str(res_df.iloc[agent_ind].qlr)+str(int(res_df.iloc[agent_ind].inv_temp))+str(int(res_df.iloc[agent_ind].inv_temp))+str(int(res_df.iloc[agent_ind].inv_temp))+str(res_df.iloc[agent_ind].gamma)+str(res_df.iloc[agent_ind].eta)+"0.010.13.21.14MB500TrueFalseTrue/agent0.csv")
        one_agent_df["agent"] = agent_ind
        one_agent_df_les["agent"] = agent_ind
        one_agent_df = one_agent_df.pivot_table(index=['agent', 'trial'], aggfunc='mean')
        one_agent_df_les = one_agent_df_les.pivot_table(index=['agent', 'trial'], aggfunc='mean')
        df_analysis=df_analysis.append(one_agent_df)
        df_analysis_les=df_analysis_les.append(one_agent_df_les)


    # ols function doesn't like spaces
    # df_analysis["escape_time"] = df_analysis["escape time"]
    # df_analysis_les["escape_time"] = df_analysis_les["escape time"]

    tests_results = []
    p = 0.05


    # perform the statistical test for each cluster of agents
    print("Performing ANOVA")
    for cluster_ind in range(len(res_df)):
        df_tmp = df_analysis.loc[closests[cluster_ind][0:100]]
        df_tmp_les = df_analysis_les.loc[closests[cluster_ind][0:100]]

        # 200 rows, 100*trial 1, 100*trial 4
        df_anova_trial = df_tmp.reset_index()
        df_anova_trial = df_anova_trial[np.logical_or(df_anova_trial["trial"]==0, df_anova_trial["trial"]==3)]
        df_anova_trial = df_anova_trial.pivot_table(index=['agent', 'trial'], aggfunc='mean')
        df_anova_trial = df_anova_trial[["escape time"]]

        # 200 rows, 100*trial 1, 100*trial 4
        df_anova_trial1 = df_tmp.reset_index()
        df_anova_trial1 = df_anova_trial1[df_anova_trial1["trial"]==0]
        df_anova_trial1 = df_anova_trial1.pivot_table(index=['agent', 'trial'], aggfunc='mean')
        df_anova_trial1 = df_anova_trial1[["escape time"]]

        df_anova_trial4 = df_tmp.reset_index()
        df_anova_trial4 = df_anova_trial4[df_anova_trial4["trial"]==3]
        df_anova_trial4 = df_anova_trial4.pivot_table(index=['agent', 'trial'], aggfunc='mean')
        df_anova_trial4 = df_anova_trial4[["escape time"]]

        # 200 rows, 100*trial 1, 100*trial 4
        df_anova_trial_lesion1 = df_tmp_les.reset_index()
        df_anova_trial_lesion1 = df_anova_trial_lesion1[df_anova_trial_lesion1["trial"]==0]
        df_anova_trial_lesion1 = df_anova_trial_lesion1.pivot_table(index=['agent', 'trial'], aggfunc='mean')
        df_anova_trial_lesion1 = df_anova_trial_lesion1[["escape time"]]

        df_anova_trial_lesion4 = df_tmp_les.reset_index()
        df_anova_trial_lesion4 = df_anova_trial_lesion4[df_anova_trial_lesion4["trial"]==3]
        df_anova_trial_lesion4 = df_anova_trial_lesion4.pivot_table(index=['agent', 'trial'], aggfunc='mean')
        df_anova_trial_lesion4 = df_anova_trial_lesion4[["escape time"]]

        df_anova_trial1["group"] = "normal"
        df_anova_trial_lesion1["group"] = "lesioned"
        df_anova_trial4["group"] = "normal"
        df_anova_trial_lesion4["group"] = "lesioned"

        df_both_1 = pd.concat([df_anova_trial1, df_anova_trial_lesion1])
        df_both_4 = pd.concat([df_anova_trial4, df_anova_trial_lesion4])
        print("agent: "+ str(cluster_ind), end="\r")
        # two way anova on trial and session (IV) and escape time (DV)
        df_anova_trial["escape_time"] = df_anova_trial["escape time"]
        model = ols('escape_time ~ C(trial)', data=df_anova_trial.reset_index()).fit()
        tmp = sm.stats.anova_lm(model, typ=2)

        # perform ANOVA (IV -> group, DV -> escape time)
        df_both_4["escape_time"] = df_both_4["escape time"]
        model2 = ols('escape_time ~ C(group) + C(group)', data=df_both_4.reset_index()).fit()
        tmp2 = sm.stats.anova_lm(model2, typ=2)

        df_both_1["escape_time"] = df_both_1["escape time"]
        model3 = ols('escape_time ~ C(group) + C(group)', data=df_both_1.reset_index()).fit()
        tmp3 = sm.stats.anova_lm(model3, typ=2)
        # model2 = ols('escape_time ~ C(trial)', data=df_tmp_les.reset_index()).fit()
        # tmp2 = sm.stats.anova_lm(model2, typ=2)
        tests_results.append(tmp["PR(>F)"]["C(trial)"]<p and tmp2["PR(>F)"]["C(group)"]<p and tmp3["PR(>F)"]["C(group)"]<p)

    res_df["anova_pearce"] = tests_results
    res_df.to_csv("../results/"+directory+"/mean_square_processed.csv", index=False)


def perform_statical_analyses_rodrigo(directory):
    """
        Performs a series of statistical test (Helmert contrast, t-tests, two-way ANOVAs)
        to test whether there is a significant generalisation gradient, that escape time is significantly increasing with angle.
        Datapoints (clusters centroid) that do not validate all the tests are tagged as such.
        They will be later removed from the list of potential candidate for best set of parameter.

        :param directory: The directory in which all files related to the grid-search can be found
        :type directory: str

        :returns: Nothing, the results are stored in a CSV file in directory
    """
    res_df = pd.read_csv("../results/"+directory+"/mean_square_processed.csv")
    # list of 5D euclidian distances between rows (each parameters set is a coordinate)
    eu_dists = get_euclidian_distances(res_df)
    # 100 closests peers for each candidate cluster
    closests = [get_closest_datapoints(eu_dists, cluster_ind) for cluster_ind in range(len(res_df))]
    coords = get_coords()

    # dataframe in which all agents data will be stored
    df_analysis = pd.DataFrame()
    agents_df_lst = []
    print("Retrieving all agents data")
    # Retrieving all agents data, from the entire grid-search related simulations, stores it in a single DataFrame
    for agent_ind in range(len(res_df)):
        print("agent: "+ str(agent_ind), end="\r")
        try:
            # if model is Geerts
            one_agent_df = pd.read_csv("../results/"+directory+"/rodrigo/rodrigo_1False"+str(res_df.iloc[agent_ind].srlr)+str(res_df.iloc[agent_ind].qlr)+str(int(res_df.iloc[agent_ind].inv_temp))+str(res_df.iloc[agent_ind].gamma)+str(res_df.iloc[agent_ind].eta)+"0.010.13.21.10SR500FalseFalseFalse/agent0.csv")
        except:
            # if model is Dolle
            one_agent_df = pd.read_csv("../results/"+directory+"/rodrigo/rodrigo_1True"+str(res_df.iloc[agent_ind].srlr)+str(res_df.iloc[agent_ind].qlr)+str(int(res_df.iloc[agent_ind].inv_temp))+str(int(res_df.iloc[agent_ind].inv_temp))+str(int(res_df.iloc[agent_ind].inv_temp))+str(res_df.iloc[agent_ind].gamma)+str(res_df.iloc[agent_ind].eta)+"0.010.13.21.10MB500FalseFalseTrue/agent0.csv")
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
    helmert_rodrigo_135vs_results_p = []
    helmert_rodrigo_0vs_results_d = []
    helmert_rodrigo_45vs_results_d = []
    helmert_rodrigo_90vs_results_d = []
    helmert_rodrigo_135vs_results_d = []
    anova_rodrigo_results = []
    ttest_rodrigo_results = []

    p = 0.05

    print("Performing statistical analyses")
    for cluster_ind in range(len(res_df)):

        print("agent: "+ str(cluster_ind), end="\r")

        cluster_df = octant_occup_df.loc[closests[cluster_ind][0:100]]

        cluster_df = cluster_df.reset_index()

        # HELMERT TESTS
        helmert = lambda lst : ols("isinoctant_proximal ~ C(angle, Helmert)", data=cluster_df[cluster_df["angle"].isin(lst)]).fit()
        helmert_rodrigo_0vs_results_p.append(helmert([0,45,90,135,180]).f_pvalue)
        helmert_rodrigo_45vs_results_p.append(helmert([45,90,135,180]).f_pvalue)
        helmert_rodrigo_90vs_results_p.append(helmert([90,135,180]).f_pvalue)
        helmert_rodrigo_135vs_results_p.append(helmert([135,180]).f_pvalue)

        # HELMERT TESTS
        helmert = lambda lst : ols("isinoctant_distal ~ C(angle, Helmert)", data=cluster_df[cluster_df["angle"].isin(lst)]).fit()
        helmert_rodrigo_0vs_results_d.append(helmert([0,45,90,135,180]).f_pvalue)
        helmert_rodrigo_45vs_results_d.append(helmert([45,90,135,180]).f_pvalue)
        helmert_rodrigo_90vs_results_d.append(helmert([90,135,180]).f_pvalue)
        helmert_rodrigo_135vs_results_d.append(helmert([135,180]).f_pvalue)

        # ANOVA
        model = ols('isinoctant_proximal ~ C(angle) + C(angle) + C(angle):C(angle)', data=cluster_df).fit()
        tmp = sm.stats.anova_lm(model, typ=2)
        model2 = ols('isinoctant_distal ~ C(angle) + C(angle) + C(angle):C(angle)', data=cluster_df).fit()
        tmp2 = sm.stats.anova_lm(model2, typ=2)
        anova_rodrigo_results.append(tmp["PR(>F)"]["C(angle)"]<p and tmp2["PR(>F)"]["C(angle)"]<p)

        # TTESTS
        ttest = lambda ang : stats.ttest_1samp(cluster_df[cluster_df["angle"] == ang].groupby("agent").mean()["isinoctant_proximal"],0.125).pvalue
        ttest2 = lambda ang : stats.ttest_1samp(cluster_df[cluster_df["angle"] == ang].groupby("agent").mean()["isinoctant_distal"],0.125).pvalue
        ttest_rodrigo_results.append(ttest(0) < p and ttest(45) < p and ttest(90) < p and ttest(135) < p and ttest(180) < p and ttest2(0) < p and ttest2(45) < p and ttest2(90) > p and ttest2(135) > p and ttest2(180) > p)


    res_df["helmert_rodrigo_0vs_p"] = helmert_rodrigo_0vs_results_p
    res_df["helmert_rodrigo_45vs_p"] = helmert_rodrigo_45vs_results_p
    res_df["helmert_rodrigo_90vs_p"] = helmert_rodrigo_90vs_results_p
    res_df["helmert_rodrigo_135vs_p"] = helmert_rodrigo_135vs_results_p
    res_df["helmert_rodrigo_0vs_d"] = helmert_rodrigo_0vs_results_d
    res_df["helmert_rodrigo_45vs_d"] = helmert_rodrigo_45vs_results_d
    res_df["helmert_rodrigo_90vs_d"] = helmert_rodrigo_90vs_results_d
    res_df["helmert_rodrigo_135vs_d"] = helmert_rodrigo_135vs_results_d
    res_df["anova_rodrigo"] = anova_rodrigo_results
    res_df["ttest_rodrigo"] = ttest_rodrigo_results

    res_df.to_csv("../results/"+directory+"/mean_square_processed.csv",index=False)


# ____________ PLOTTING _____________

def plot_single_perfs(df, bins_nbr, dim1_name, dim2_name, expe, relative=True):
    """
        Display a two dimensional heatmap of the best performances obtained for each 2D bin

        :param df: A dataframe containing all agents data simulated during the grid-search
        :type df: pandas DataFrame
        :param bins_nbr: The number of bins to plot per axis
        :type bins_nbr: int
        :param dim1_name: The name of the dimension to plot on one axis
        :type dim1_name: str
        :param dim2_name: The name of the dimension to plot on one axis
        :type dim2_name: str
        :param expe: Whether the dislayed MSE are computed using pearce or rodrigo data, or the product of both
        :type expe: str
        :param relative: Whether the MSE is computed using absolute or relative experimental data reference
        :type relative: boolean

        :returns: (respectively) A DataFrame of clusters that validated all statistical tests,
        a DataFrame representing the values of the heatmap to display,
        a list of 2 elements tuples indicating the lower and upper bound of each bin of the first dimension,
        a list of 2 elements tuples indicating the lower and upper bound of each bin of the second dimension,
    """

    # will contain only datapoints which associated clusters have validated all the statistical tests
    validated_clusters_df = df

    if expe == "pearce" or expe == "product":
        try:
            validated_clusters_df = validated_clusters_df[validated_clusters_df["anova_pearce"] == True]
        except:
            print("Warning: no statistical tests were performed on simulations data for Pearce's experiment")

    if expe == "rodrigo" or expe == "product":
        try:
            validated_clusters_df = validated_clusters_df[validated_clusters_df["ttest_rodrigo"] == True]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["anova_rodrigo"] == True]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["helmert_rodrigo_0vs_p"] < 0.005]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["helmert_rodrigo_45vs_p"] < 0.005]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["helmert_rodrigo_90vs_p"] < 0.005]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["helmert_rodrigo_135vs_p"] < 0.005]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["helmert_rodrigo_0vs_d"] < 0.05]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["helmert_rodrigo_45vs_d"] > 0.05]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["helmert_rodrigo_90vs_d"] > 0.05]
            validated_clusters_df = validated_clusters_df[validated_clusters_df["helmert_rodrigo_135vs_d"] > 0.05]
        except:
            print("Warning: no statistical tests were performed on simulations data for Rodrigo's experiment")

    # create equally spaced bins in the log space, returns bins upper and lower bounds
    def create_bins(df, column_name, nbr_bin):
        mini = df[column_name].min()
        maxi = df[column_name].max()
        bins = np.logspace(np.log10(mini), np.log10(maxi), nbr_bin+1)
        return [*zip(bins, bins[1:])]

    # return a list of tuple, each representing the lower and upper limit of a one dimensional bin
    bins_dim1 = create_bins(df, dim1_name, bins_nbr)
    bins_dim2 = create_bins(df, dim2_name, bins_nbr)

    # two dimensional dataframe representing each bin value which is initialized to 0
    data = pd.DataFrame([[0.]*bins_nbr]*bins_nbr, index=range(bins_nbr), columns=range(bins_nbr))

    # exhaustive list of euclidian distances between datapoints
    eu_dists = get_euclidian_distances(df)
    closests = [get_closest_datapoints(eu_dists, i) for i in range(len(df))]

    def link_datapoint_to_cluster_centroid(df, name_var, closests):
        df[str(name_var)+"_centroid"] = df.index.map(lambda row: df[name_var].loc[closests[int(row)][0:100]].mean())

    link_datapoint_to_cluster_centroid(df, "qlr", closests)
    link_datapoint_to_cluster_centroid(df, "gamma", closests)
    link_datapoint_to_cluster_centroid(df, "srlr", closests)
    link_datapoint_to_cluster_centroid(df, "inv_temp", closests)
    link_datapoint_to_cluster_centroid(df, "eta", closests)

    for dim1 in range(bins_nbr):
        for dim2 in range(bins_nbr):
            # BELOW : COMPUTING BEST CLUSTER IN THE 2D BIN

            # return the indices of the rows contained in both var1_bin limits and var2_bin limits
            def get_indexes_in_bin(df, var1_name, var1_bin, var2_name, var2_bin):
                # retain only rows contained in the first dimension bin
                df_sub = df[np.logical_and(df[var1_name]>=var1_bin[0], df[var1_name]<var1_bin[1])]
                # refine again df_sub with only rows contained in the second dimension bin
                df_subsub = df_sub[np.logical_and(df_sub[var2_name]>=var2_bin[0], df_sub[var2_name]<var2_bin[1])]
                return np.array(df_subsub.index)

            indexes = get_indexes_in_bin(df, dim1_name, bins_dim1[dim1], dim2_name, bins_dim2[dim2])
            indexes = [ind for ind in indexes if ind in validated_clusters_df.index]
            if expe == "pearce":
                data[dim1][dim2] = df["se_cluster_100_pearce_mean"][indexes].min()
            elif expe == "rodrigo":
                if relative:
                    data[dim1][dim2] = df["se_cluster_100_rodrigo_mean_relative"][indexes].min()
                else:
                    data[dim1][dim2] = df["se_cluster_100_rodrigo_mean_absolute"][indexes].min()
            elif expe == "product":
                if relative:
                    data[dim1][dim2] = df["se_cluster_100_product_mean_relative"][indexes].min()
                else:
                    data[dim1][dim2] = df["se_cluster_100_product_mean_absolute"][indexes].min()
            else:
                raise Exception("The experiment should either be pearce, rodrigo or product")

    return validated_clusters_df, data, bins_dim2, bins_dim1


def plot_two_perfs(directory, expe="pearce", size_plot=10, relative=True, mode="geerts"):
    """
        Display two heatmaps of the best performances obtained for a given experiment,
        this allows the visualisation of four parameter dimensions in total

        :param directory: The path where the grid-search logs files are stored
        :type directory: str
        :param expe: Specify whether the performances (min MSE) to be plotted must be computed using pearce data, dolle's or the product of both
        :type expe: str
        :param size_plot: Number of bins for each axis
        :type size_plot: int
        :param relative: Whether the MSE is computed using absolute or relative experimental data reference
        :type relative: boolean
        :param mode: Whether the simulated data in directory has been generated with dolle or geerts model
        :type mode: str

        :returns: A pandas DataFrame containing all the valid datapoints (100 nearest-neighbors cluster
        validated all statistical tests) generated during the grid-search
    """
    try:
        res_df = pd.read_csv("../saved_results/"+directory+"/mean_square_processed.csv")
    except:
        res_df = pd.read_csv("../results/"+directory+"/mean_square_processed.csv")
    # plot only two dimensions, srlr and eta are parameters specific to respectively the geerts and dolle models
    if mode == "geerts":
        _,data1,bin12,bin11 = plot_single_perfs(res_df, size_plot, "inv_temp", "srlr", expe, relative)
    elif mode == "dolle":
        _,data1,bin12,bin11 = plot_single_perfs(res_df, size_plot, "inv_temp", "eta", expe, relative)
    else:
        raise Exception("Mode should be either geerts or dolle")

    # plot the other two dimensions, which exists for both dolle and geerts model
    df2, data2, bin22, bin21 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", expe, relative)

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

    # to retrieve matplotlib objects so each heatmap can be integrated in a subplot
    def plot_heatmap(ax, data, bin2, bin1):
        plot = ax.pcolor(data)
        ax.set_yticks(range(len(data.index))[1::2])
        ax.set_xticks(range(len(data.columns))[1::2])
        ax.set_yticklabels([round(w[0], 2) for w in bin2][1::2])
        ax.set_xticklabels([round(w[0], 2) for w in bin1][1::2])
        return plot

    plot1 = plot_heatmap(ax1, data1, bin12, bin11)
    plot2 = plot_heatmap(ax2, data2, bin22, bin21)

    ax1.set_xlabel('Inverse temperature')
    ax2.set_xlabel('MF learning rate')
    if mode == "geerts":
        ax1.set_ylabel('SR learning rate')
    elif mode == "dolle":
        ax1.set_ylabel('Arbitrator learning rate')
    else:
        raise Exception("Mode should be either geerts or dolle")
    ax2.set_ylabel('Discount factor')

    if expe == "product":
        lab = "both"
    else:
        lab = expe

    fig.suptitle("Lowest square error per bin on "+lab+" experiment, all dimensions considered")

    plt.colorbar(plot1, label="square error", ax = ax1)
    plt.colorbar(plot2, label="square error", ax = ax2)

    plt.show()

    if relative:
        try:
            df_sorted = df2.sort_values(by=["se_cluster_100_"+expe+"_mean_relative"], ascending=True)
        except:
            df_sorted = df2.sort_values(by=["se_cluster_100_"+expe+"_mean"], ascending=True)
        df_sorted["se_cluster_pearce"] = df_sorted["se_cluster_100_pearce_mean"]
        df_sorted["se_cluster_rodrigo_relative"] = df_sorted["se_cluster_100_rodrigo_mean_relative"]
        df_sorted["se_cluster_product_relative"] = df_sorted["se_cluster_100_product_mean_relative"]
        # display the three first datapoint with lowest product MSE
        display(HTML(df_sorted[["srlr_centroid", "qlr_centroid", "gamma_centroid", "inv_temp_centroid", "eta_centroid", "se_cluster_pearce", "se_cluster_rodrigo_relative", "se_cluster_product_relative"]].head(3).to_html()))
    else:
        try:
            df_sorted = df2.sort_values(by=["se_cluster_100_"+expe+"_mean_absolute"], ascending=True)
        except:
            df_sorted = df2.sort_values(by=["se_cluster_100_"+expe+"_mean"], ascending=True)
        df_sorted["se_cluster_pearce"] = df_sorted["se_cluster_100_pearce_mean"]
        df_sorted["se_cluster_rodrigo_absolute"] = df_sorted["se_cluster_100_rodrigo_mean_absolute"]
        df_sorted["se_cluster_product_absolute"] = df_sorted["se_cluster_100_product_mean_absolute"]
        # display the three first datapoint with lowest product MSE
        display(HTML(df_sorted[["srlr_centroid", "qlr_centroid", "gamma_centroid", "inv_temp_centroid", "eta_centroid", "se_cluster_pearce", "se_cluster_rodrigo_absolute", "se_cluster_product_absolute"]].head(3).to_html()))

    return df2

def plot_pearce_perfs(directory, size_plot=10, relative=True, mode="geerts"):
    """
        Display the best performances (min MSE) of the model on pearce data

        :param directory: The path where the grid-search logs files are stored
        :type directory: str
        :param size_plot: Number of bins for each axis
        :type size_plot: int
        :param relative: Whether the MSE is computed using absolute or relative experimental data reference
        :type relative: boolean
        :param mode: Whether the simulated data in directory has been generated with dolle or geerts model
        :type mode: str

        :returns: A pandas DataFrame containing all the datapoints generated during the grid-search (parameters value, MSE, ...)
    """
    try:
        res_df = pd.read_csv("../saved_results/"+directory+"/mean_square_processed.csv")
    except:
        res_df = pd.read_csv("../results/"+directory+"/mean_square_processed.csv")

    if mode == "geerts":
        df_pearce,data1,bin12,bin11 = plot_single_perfs(res_df, size_plot, "inv_temp", "srlr", "pearce", relative)
        _,data2,bin22,bin21 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", "pearce", relative)

    elif mode == "dolle":
        df_pearce,data1,bin12,bin11 = plot_single_perfs(res_df, size_plot, "inv_temp", "eta", "pearce", relative)
        _,data2,bin22,bin21 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", "pearce", relative)

    else:
        raise Exception("Mode should be either geerts or dolle")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))

    # to retrieve matplotlib objects so each heatmap can be integrated in a subplot
    def plot_heatmap(ax, data, bin2, bin1):
        plot = ax.pcolor(data)
        ax.set_yticks(range(len(data.index))[1::2])
        ax.set_xticks(range(len(data.columns))[1::2])
        ax.set_yticklabels([round(w[0], 2) for w in bin2][1::2])
        ax.set_xticklabels([round(w[0], 2) for w in bin1][1::2])
        return plot

    plot1 = plot_heatmap(ax1, data1, bin12, bin11)
    plot2 = plot_heatmap(ax2, data2, bin22, bin21)

    ax1.set_xlabel('Inverse temperature')
    ax2.set_xlabel('MF learning rate')
    ax2.set_xlabel('Inverse temperature')
    ax2.set_xlabel('MF learning rate')
    ax1.set_ylabel('SR learning rate')
    ax2.set_ylabel('Discount factor')

    fig.suptitle("Lowest square error per bin, all dimensions considered")
    plt.colorbar(plot1, ax = ax1)
    plt.colorbar(plot2, label="square error", ax = ax2)

    ax1.set_title("Pearce")

    plt.show()

    print("Best three sets of parameters on Pearce's data")
    df_pearce = res_df[res_df["index"].isin(df_pearce["index"])]
    if relative:
        df_sorted = df_pearce.sort_values(by=["se_cluster_100_pearce_mean"], ascending=True)
        df_sorted["se_cluster_pearce"] = df_sorted["se_cluster_100_pearce_mean"]
        # display the three first datapoint with lowest product MSE
        display(HTML(df_sorted[["srlr_centroid", "qlr_centroid", "gamma_centroid", "inv_temp_centroid", "eta_centroid", "se_cluster_pearce"]].head(3).to_html()))
    else:
        df_sorted = df_pearce.sort_values(by=["se_cluster_100_pearce_mean"], ascending=True)
        df_sorted["se_cluster_pearce"] = df_sorted["se_cluster_100_pearce_mean"]
        # display the three first datapoint with lowest product MSE
        display(HTML(df_sorted[["srlr_centroid", "qlr_centroid", "gamma_centroid", "inv_temp_centroid", "eta_centroid", "se_cluster_pearce"]].head(3).to_html()))

    print("Index of the best 3 sets of parameters:", list(df_sorted.index[0:3]))
    
    return res_df


def plot_all_perfs(directory, size_plot=10, relative=True, mode="geerts"):
    """
        Display the best performances (min MSE) of the model on both pearce data,
        rodrigo data, and their product

        :param directory: The path where the grid-search logs files are stored
        :type directory: str
        :param size_plot: Number of bins for each axis
        :type size_plot: int
        :param relative: Whether the MSE is computed using absolute or relative experimental data reference
        :type relative: boolean
        :param mode: Whether the simulated data in directory has been generated with dolle or geerts model
        :type mode: str

        :returns: A pandas DataFrame containing all the datapoints generated during the grid-search (parameters value, MSE, ...)
    """
    try:
        res_df = pd.read_csv("../saved_results/"+directory+"/mean_square_processed.csv")
    except:
        res_df = pd.read_csv("../results/"+directory+"/mean_square_processed.csv")

    if mode == "geerts":
        df_pearce,data1,bin12,bin11 = plot_single_perfs(res_df, size_plot, "inv_temp", "srlr", "pearce", relative)
        _,data4,bin42,bin41 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", "pearce", relative)

        df_rodrigo,data2,bin22,bin21 = plot_single_perfs(res_df, size_plot, "inv_temp", "srlr", "rodrigo", relative)
        _,data5,bin52,bin51 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", "rodrigo", relative)

        #_,data3,bin32,bin31 = plot_single_perfs(res_df, size_plot, "inv_temp", "srlr", "product", relative)
        #_,data6,bin62,bin61 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", "product", relative)
    elif mode == "dolle":
        df_pearce,data1,bin12,bin11 = plot_single_perfs(res_df, size_plot, "inv_temp", "eta", "pearce", relative)
        _,data4,bin42,bin41 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", "pearce", relative)

        df_rodrigo,data2,bin22,bin21 = plot_single_perfs(res_df, size_plot, "inv_temp", "eta", "rodrigo", relative)
        _,data5,bin52,bin51 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", "rodrigo", relative)

        #_,data3,bin32,bin31 = plot_single_perfs(res_df, size_plot, "inv_temp", "eta", "product", relative)
        #_,data6,bin62,bin61 = plot_single_perfs(res_df, size_plot, "qlr", "gamma", "product", relative)
    else:
        raise Exception("Mode should be either geerts or dolle")

    fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2, figsize=(10, 8))

    # to retrieve matplotlib objects so each heatmap can be integrated in a subplot
    def plot_heatmap(ax, data, bin2, bin1):
        plot = ax.pcolor(data)
        ax.set_yticks(range(len(data.index))[1::2])
        ax.set_xticks(range(len(data.columns))[1::2])
        ax.set_yticklabels([round(w[0], 2) for w in bin2][1::2])
        ax.set_xticklabels([round(w[0], 2) for w in bin1][1::2])
        return plot

    plot1 = plot_heatmap(ax1, data1, bin12, bin11)
    plot2 = plot_heatmap(ax2, data2, bin22, bin21)
    #plot3 = plot_heatmap(ax3, data3, bin32, bin31)
    plot4 = plot_heatmap(ax4, data4, bin42, bin41)
    plot5 = plot_heatmap(ax5, data5, bin52, bin51)
    #plot6 = plot_heatmap(ax6, data6, bin62, bin61)

    ax1.set_xlabel('Inverse temperature')
    ax4.set_xlabel('MF learning rate')
    ax2.set_xlabel('Inverse temperature')
    ax5.set_xlabel('MF learning rate')
    ax1.set_ylabel('SR learning rate')
    ax4.set_ylabel('Discount factor')

    fig.suptitle("Lowest square error per bin, all dimensions considered")
    plt.colorbar(plot1, ax = ax1)
    plt.colorbar(plot2, label="square error", ax = ax2)
    #plt.colorbar(plot3, label="square error", ax = ax3)
    plt.colorbar(plot4, ax = ax4)
    plt.colorbar(plot5, label="square error", ax = ax5)
    #plt.colorbar(plot6, label="square error", ax = ax6)

    ax1.set_title("Pearce")
    ax2.set_title("Rodrigo")
    #ax3.set_title("Pearce + Rodrigo")

    plt.show()

    print("Best three sets of parameters on Pearce's data")
    df_pearce = res_df[res_df["index"].isin(df_pearce["index"])]
    if relative:
        df_sorted = df_pearce.sort_values(by=["se_cluster_100_pearce_mean"], ascending=True)
        df_sorted["se_cluster_pearce"] = df_sorted["se_cluster_100_pearce_mean"]
        df_sorted["se_cluster_rodrigo_relative"] = df_sorted["se_cluster_100_rodrigo_mean_relative"]
        #df_sorted["se_cluster_product_relative"] = df_sorted["se_cluster_100_product_mean_relative"]
        # display the three first datapoint with lowest product MSE
        display(HTML(df_sorted[["srlr_centroid", "qlr_centroid", "gamma_centroid", "inv_temp_centroid", "eta_centroid", "se_cluster_pearce", "se_cluster_rodrigo_relative"]].head(3).to_html()))
    else:
        df_sorted = df_pearce.sort_values(by=["se_cluster_100_pearce_mean"], ascending=True)
        df_sorted["se_cluster_pearce"] = df_sorted["se_cluster_100_pearce_mean"]
        df_sorted["se_cluster_rodrigo_absolute"] = df_sorted["se_cluster_100_rodrigo_mean_absolute"]
        #df_sorted["se_cluster_product_absolute"] = df_sorted["se_cluster_100_product_mean_absolute"]
        # display the three first datapoint with lowest product MSE
        display(HTML(df_sorted[["srlr_centroid", "qlr_centroid", "gamma_centroid", "inv_temp_centroid", "eta_centroid", "se_cluster_pearce", "se_cluster_rodrigo_absolute"]].head(3).to_html()))

    print("Best three sets of parameters on Rodrigo's data")
    df_rodrigo = res_df[res_df["index"].isin(df_rodrigo["index"])]
    if relative:
        df_sorted = df_rodrigo.sort_values(by=["se_cluster_100_rodrigo_mean_relative"], ascending=True)
        df_sorted["se_cluster_pearce"] = df_sorted["se_cluster_100_pearce_mean"]
        df_sorted["se_cluster_rodrigo_relative"] = df_sorted["se_cluster_100_rodrigo_mean_relative"]
        #df_sorted["se_cluster_product_relative"] = df_sorted["se_cluster_100_product_mean_relative"]
        # display the three first datapoint with lowest product MSE
        display(HTML(df_sorted[["srlr_centroid", "qlr_centroid", "gamma_centroid", "inv_temp_centroid", "eta_centroid", "se_cluster_pearce", "se_cluster_rodrigo_relative"]].head(3).to_html()))
    else:
        df_sorted = df_rodrigo.sort_values(by=["se_cluster_100_rodrigo_mean_absolute"], ascending=True)
        df_sorted["se_cluster_pearce"] = df_sorted["se_cluster_100_pearce_mean"]
        df_sorted["se_cluster_rodrigo_absolute"] = df_sorted["se_cluster_100_rodrigo_mean_absolute"]
        #df_sorted["se_cluster_product_absolute"] = df_sorted["se_cluster_100_product_mean_absolute"]
        # display the three first datapoint with lowest product MSE
        display(HTML(df_sorted[["srlr_centroid", "qlr_centroid", "gamma_centroid", "inv_temp_centroid", "eta_centroid", "se_cluster_pearce", "se_cluster_rodrigo_absolute"]].head(3).to_html()))

    return res_df


def plot_local(expe, df, indi, relative=True):
    """
        Display the mean performance of a cluster of 100 agents on a specific task

        :param df: A DataFrame containing the data of all agents simulated during the grid-search, and which validated some statistical tests
        :type df: pandas DataFrame
        :param indi: Index of the cluster in df
        :type indi: int
        :param relative: Whether the MSE is computed using absolute or relative experimental data reference
        :type relative: boolean

        :returns: Nothing, but display a matplotlib object
    """
    if expe == "main_pearce":

        c1 = get_array_from_str(df.loc[indi]["se_cluster_100_pearce_control1"])
        c4 = get_array_from_str(df.loc[indi]["se_cluster_100_pearce_control4"])
        h1 = get_array_from_str(df.loc[indi]["se_cluster_100_pearce_hpc1"])
        h4 = get_array_from_str(df.loc[indi]["se_cluster_100_pearce_hpc4"])

        fig, axs = plt.subplots(1, 2, figsize=(15,6))

        srlr = round(df.loc[indi]["srlr_centroid"], 3)
        qlr = round(df.loc[indi]["qlr_centroid"], 3)
        gamma = round(df.loc[indi]["gamma_centroid"], 3)
        eta = round(df.loc[indi]["eta_centroid"], 3)
        inv_temp = round(df.loc[indi]["inv_temp_centroid"], 3)
        fig.suptitle('Awaited results for the following parameters      SRLR:'+str(srlr)+"      QLR:"+str(qlr)+"     GAMMA:"+str(gamma)+"      INV_TEMP:"+str(inv_temp)+"      ETA:"+str(eta), fontweight="bold")

        axs[0].set(title="Original results")
        axs[0].imshow(mpimg.imread("../images/results_pearce.jpg"))
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_frame_on(False)
        axs[0].plot(aspect="auto")

        axs[1].plot(h1, marker="o", color='black', label="HPC lesion - trial 1")
        axs[1].plot(c1, marker="o", markerfacecolor='none', color='black', label="Control - trial 1")
        axs[1].plot(h4, marker="o", linestyle='--', color='black', label="HPC lesion - trial 4")
        axs[1].plot(c4, marker="o", linestyle='--', markerfacecolor='none', color='black', label="Control - trial 4")

        axs[1].set(title="Local aggregation results")
        axs[1].set(ylabel="Escape latency (s)")
        axs[1].set(xlabel="Session")
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(handles=[handles[0],handles[1],handles[2],handles[3]], labels=[labels[0],labels[1],labels[2],labels[3]])
        axs[1].plot(aspect="auto")
        axs[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.xticks(np.arange(11), np.arange(1, 11+1))

        plt.show()
        plt.close()

    if expe == "rodrigo":

        if relative:
            distal = get_array_from_str(df.loc[indi]["se_cluster_100_rodrigo_relative"])[:5]
            proximal = get_array_from_str(df.loc[indi]["se_cluster_100_rodrigo_relative"])[5:]
        else:
            distal = get_array_from_str(df.loc[indi]["se_cluster_100_rodrigo_absolute"])[:5]
            proximal = get_array_from_str(df.loc[indi]["se_cluster_100_rodrigo_absolute"])[5:]

        fig, axs = plt.subplots(2, 2, figsize=(15,8))

        axs[0,0].set_title("Original visual results")
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

        srlr = round(df.loc[indi]["srlr_centroid"], 3)
        qlr = round(df.loc[indi]["qlr_centroid"], 3)
        gamma = round(df.loc[indi]["gamma_centroid"], 3)
        inv_temp = round(df.loc[indi]["inv_temp_centroid"], 3)
        eta = round(df.loc[indi]["eta_centroid"], 3)
        fig.suptitle('Awaited results for the following parameters      SRLR:'+str(srlr)+"      QLR:"+str(qlr)+"     GAMMA:"+str(gamma)+"      INV_TEMP:"+str(inv_temp)+"      ETA:"+str(eta), fontweight="bold")

        axs[0,1].bar(["0°", "45°", "90°", "135°", "180°"], np.array(proximal)-0.125, color='gray', edgecolor="black", ) # set 0 to chance level
        axs[0,1].set_title("Local aggregation results")
        axs[0,1].set_ylabel("Proportion of steps searching in the B octant")
        axs[0,1].set_xlabel("Tests")

        axs[1,1].bar(["0°", "45°", "90°", "135°", "180°"], np.array(distal)-0.125, color='gray', edgecolor="black") # set 0 to chance level
        axs[1,1].set_ylabel("Proportion of steps searching in the F octant")
        axs[1,1].set_xlabel("Tests")

        plt.show()
        plt.close()


# returns a list of all datapoints simulated during the grid-search,
# sorted from closest to furthest euclidian distance to datapoint index
# points to points distances are stored in the 2D array eu_dists
def get_closest_datapoints(eu_dists, index):
    return sorted(range(len(eu_dists[index])), key=lambda k: eu_dists[index][k])


# transforms strings from a DataFrame, representing an array of floats into an actual array of int
def get_array_from_str(s):
    splitted = s.split(" ")
    int_list = []
    for elem in splitted:
        try:
            actual_int = re.sub("[^0-9.]", "", elem)
            int_list.append(float(actual_int))
        except:
            pass
    return np.array(int_list)
