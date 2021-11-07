#!/usr/bin/python

import warnings
warnings.filterwarnings('ignore')

import sys
from agents.agent import AgentsParams
from rodrigo_protocol import perform_group_rodrigo
from environments.HexWaterMaze import EnvironmentParams

# This is an executable script

def perform_full_rodrigo():
    """Allows the user to launch the complete experiment of Rodrigo 2006 using the command line
        First mandatory parameter: the name of the directory where to store the results
        Second optional parameter: which set of parameters to use, either 'best_geerts', 'best_dolle' or 'custom'
        'custom' is a random set of parameters that is intended to be modified by the user to explore new models behaviors on Rodrigo's task
        'best_geerts' is the set of parameters that was found to produce the most biologically plausible behavior using geerts' coordination model
        'best_dolle' is the set of parameters that was found to produce the most biologically plausible behavior using dolle's' coordination model
        the best set of parameters for each models were found using a random grid-search (see grid_search module)
    """

    if len(sys.argv) < 2 or (len(sys.argv) == 2 and (sys.argv[1] == "best_geerts" or sys.argv[1] == "best_dolle" or sys.argv[1] == "custom")):
        raise Exception("Directory name is mandatory")

    if len(sys.argv) < 3 or sys.argv[2] == "best_geerts": # best parameters for geerts model (found using a grid-search)

        env_params = EnvironmentParams()
        env_params.maze_size = 10
        env_params.n_sessions = 11
        env_params.n_trials = 4
        env_params.n_agents = 3
        env_params.init_sr = "zero"
        env_params.landmark_dist = 4
        env_params.time_limit = 500
        env_params.starting_states = [243,230,270,257]

        ag_params = AgentsParams()
        ag_params.mf_allo = False
        ag_params.hpc_lr = 0.069
        ag_params.q_lr = 0.146
        ag_params.inv_temp = 13
        ag_params.gamma = 0.853
        ag_params.eta = 0.03 # reliability learning rate
        ag_params.alpha1 = 0.01
        ag_params.beta1 = 0.1
        ag_params.A_alpha = 3.2 # Steepness of transition curve MF to SR
        ag_params.A_beta = 1.1 # Steepness of transition curve SR to MF
        ag_params.HPCmode = "SR"
        ag_params.lesion_HPC = False
        ag_params.lesion_DLS = False
        ag_params.dolle = False

        print()
        print("Performing "+str(env_params.n_agents)+" simulations with Geerts coordination model and best set of parameters")

    elif sys.argv[2] == "best_dolle": # best parameters for dolle model (found using a grid-search)

        env_params = EnvironmentParams()
        env_params.maze_size = 10
        env_params.n_sessions = 11
        env_params.n_trials = 4
        env_params.n_agents = 1
        env_params.landmark_dist = 4
        env_params.time_limit = 500
        env_params.starting_states = [243,230,270,257]

        ag_params = AgentsParams()
        ag_params.mf_allo = True
        ag_params.hpc_lr = 0.07
        ag_params.q_lr = 0.20
        ag_params.inv_temp_gd = 46
        ag_params.inv_temp_mf = 46
        ag_params.arbi_inv_temp = 46
        ag_params.gamma = 0.86
        ag_params.arbi_learning_rate = 0.031 # reliability learning rate
        ag_params.HPCmode = "MB"
        ag_params.lesion_HPC = False
        ag_params.lesion_DLS = False
        ag_params.dolle = True
        ag_params.lesion_PFC = True

        print()
        print("Performing "+str(env_params.n_agents)+" simulations with Dolle coordination model and best set of parameters")

    elif sys.argv[2] == "custom": # intended to me modified by the user

        env_params = EnvironmentParams()
        env_params.maze_size = 10
        env_params.n_sessions = 11
        env_params.n_trials = 4
        env_params.n_agents = 1
        #env_params.init_sr = "zero"
        env_params.landmark_dist = 4
        env_params.time_limit = 500
        env_params.starting_states = [243,230,270,257]

        ag_params = AgentsParams()
        ag_params.mf_allo = True
        ag_params.hpc_lr = 0.07
        ag_params.q_lr = 0.028
        ag_params.inv_temp = 27
        ag_params.inv_temp_gd = 27
        ag_params.inv_temp_mf = 27
        ag_params.arbi_inv_temp = 27
        ag_params.gamma = 0.745
        ag_params.eta = 0.053 # reliability learning rate
        #ag_params.alpha1 = 0.01
        #ag_params.beta1 = 0.1
        #ag_params.A_alpha = 3.2 # Steepness of transition curve MF to SR
        #ag_params.A_beta = 1.1 # Steepness of transition curve SR to MF
        ag_params.HPCmode = "MB" # 'MB' or 'SR'
        ag_params.lesion_HPC = True
        ag_params.lesion_DLS = False
        ag_params.dolle = True
        ag_params.lesion_PFC = True

        print()
        print("Performing "+str(env_params.n_agents)+" simulations with custom parameters")

    else:
        print("arg[1] should either be 'best_geerts', 'best_dolle', 'custom' or None")
        sys.exit(1)

    print()

    ag_params.lesion_HPC = False
    print("Computing control group simulations")
    perform_group_rodrigo(env_params, ag_params, directory=sys.argv[1], show_plots=False, save_plots=True)

    print("Saving data at "+str(sys.argv[1])+" directory               ")
    print("Done")
    print()
    sys.exit()

perform_full_rodrigo()
