import os
import pickle
import numpy as np
import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
from environments.HexWaterMaze import HexWaterMaze
from matplotlib.patches import RegularPolygon, Rectangle


def plot_pearce(figsize=(1,1)):
    """
    Plot a representation of the Pearce water-maze. Eligible platforms in red, associated landmarks in blue.
    Possible starting states in green.
    """
    plt.figure(figsize=figsize)
    env = HexWaterMaze(10, 1, edge_states = [])
    plot_grid(env, platforms=[48, 52, 118, 122, 126, 94, 98, 44], landmarks=[108, 112, 116, 56, 60, 40, 100, 104], release_states=[243,230,270,257])


def plot_grid(self, c_mappable=None, ax=None, show_state_idx=True, alpha=1., c_map=None, platforms=None, landmarks=None, release_states=None):
    """
    Plot a representation of any Morris water-maze.
    :param platforms: Eligible platforms (will be colored in red)
    :type platforms: int list
    :param landmarks: Proximal landmarks associated to platforms (will be colored in blue)
    :type landmarks: int list
    :param release_states: Eligible starting states (will be colored in green)
    :type release_states: int list
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')

    if c_mappable is not None:
        if c_map is None:
            cmap = plt.get_cmap('Greys_r')
        else:
            cmap = plt.get_cmap(c_map)
        c_mappable = c_mappable / c_mappable.max()

    for i, (x, y) in enumerate(self.grid.cart_coords.values()):

        if c_mappable is not None:
            colour = cmap(c_mappable[i])
        else:
            colour = 'gray'
        if platforms is not None:
            if i in platforms:
                colour = "red"
        if landmarks is not None:
            if i in landmarks:
                colour = "blue"
        if release_states is not None:
            if i in release_states:
                colour = "green"

        hex_patch = RegularPolygon((x, y), numVertices=6, radius=2. / 3., facecolor=colour,
                                   orientation=np.radians(30), alpha=alpha, edgecolor='k')
        ax.add_patch(hex_patch)
        if show_state_idx:
            ax.text(x, y, str(i), ha='center', va='center', size=10)

    lower_bound = min(min(self.grid.cart_coords.values()))
    upper_bound = max(max(self.grid.cart_coords.values()))
    plt.xlim([lower_bound - 2, upper_bound + 2])
    plt.ylim([lower_bound - 2, upper_bound + 2])
    return ax


def get_best_action(agent, agent_pos):
    """
    Return the preferred action of an agent strategy at a given state, for four strategies:
    egocentric MF, allocentric MF, goal-directed and coordination model
    Used iteratively to create the heading-vectors quivers (see plot_main_pearce_quivs() in main_pearce module)
    :param agent: the agent object
    :type agent: Agent
    :param agent_pos: the state of interest
    :type agent_pos: int

    :returns: the preferred action (index of Qmax) of each strategy + the preferred choice of the dolle coordination model (optional)
    :return type: tuple of four int, tuple of five int if agent.dolle is True
    """
    # for display purposes of the egocentric heading-vectors, the agent is oriented toward the landmark
    if not agent.mf_allo:
        possible_orientations = np.round(np.degrees(agent.env.action_directions)) # retrieve the 6 possible orientations of the agent
        angles = []
        for i, o in enumerate(possible_orientations): # get angle of the landmark considering each possible orientation of the agent
            angle = angle_to_landmark(agent.env.get_state_location(agent_pos), agent.env.proximal_landmark_location, np.radians(o))
            angles.append(angle)
        # orientation of the agent is set to the one in which its angle to the landmark is the smallest
        agent.env.agent_orientation = possible_orientations[np.argmin(np.abs(angles))]

    if str(type(agent)) != "<class 'agents.dolle_agent.DolleAgent'>" : # if coordination model is Dolle
        agent.env.current_state = agent_pos
        Q_combined, Q_mf, Q_allo, Q_sr = agent.compute_Q(agent_pos)
        #return np.argmax(Q_mf), np.argmax(Q_allo), np.argmax(Q_sr), np.argmax(Q_combined)
        return Q_mf, Q_allo, Q_sr, Q_combined
    else: # if coordination model is Geerts
        agent.env.current_state = agent_pos
        Q_combined, Q_mf, Q_allo, Q_sr, Q_arbi, decision_arbi = agent.compute_Q(agent_pos)
        #return np.argmax(Q_mf), np.argmax(Q_allo), np.argmax(Q_sr), np.argmax(Q_combined), decision_arbi
        return Q_mf, Q_allo, Q_sr, Q_combined, decision_arbi



def get_mean_preferred_dirs(agents_lst, platform_idx=None, nb_trials=None):
    """
    Compute and returns the mean preferred direction of movement for a group of agents, for all 270 states of the water-maze.
    For four different strategies: Egocentric MF, allocentric MF, goal-directed and coordination model.
    The user may specify a new platform location, and additional training to give to the agents before recording their preferred behaviors.

    :param agents_lst: the list of agents of interest
    :type agents_lst: Agent list
    :param platform_idx: the new platform location (no changes if set to None)
    :type platform_idx: int
    :param nb_trials: the number of additional training episodes to give to each agents with the new platform configuration
    :type nb_trials: int

    :returns: the preferred mean direction of movement of each strategy, for each 270 state + the mean preferred choice of the dolle coordination model (optional)
    :return type: tuple of four list of 270 int, tuple of five list of 270 int if agent.dolle is True
    """
    # contain preferred actions index from 0 to 6
    preferred_actions_mf = []
    preferred_actions_allo = []
    preferred_actions_sr = []
    preferred_actions_combined = []
    preferred_actions_arbi = []
    p_sr = []
    prevaMF = []
    prevaSR = []
    varmf = []
    varsr = []

    # create an action index to vectors dictionary
    actions_to_vectors = {}
    for action in range(0,6):
        next_state = list(agents_lst[0].env.transition_probabilities[0,action]).index(1)
        actions_to_vectors[action] = agents_lst[0].env.grid.cart_coords[next_state]

    # modify platform location if needed, and train the agent for nb_trials episodes with the new platform location
    if platform_idx is not None:
        for agent in agents_lst:
            agent.env.set_platform_state(platform_idx)
            #agent.env.delete_distal_landmark()
            agent.env.set_proximal_landmark()
            for i in range(nb_trials):
                agent.env.one_episode(agent, 500)

    # for each state of the water-maze
    for pos in range(0,271):
        max_mfs = []
        max_allos = []
        max_srs = []
        max_combineds = []
        mean_arbi = []
        mean_p_sr = []
        mean_prevaMF = []
        mean_prevaSR = []
        mean_varmf = []
        mean_varsr = []
        # retrieve each agent strategies preferred action
        for agent in agents_lst:
            if str(type(agent)) != "<class 'agents.dolle_agent.DolleAgent'>" :
                max_mf, max_allo, max_sr, max_combined = get_best_action(agent, agent_pos = pos)
                mean_p_sr.append(agent.p_sr)
                mean_varmf.append(np.array(max_mf).var())
                mean_varsr.append(np.array(max_sr).var())
                if np.array(max_allo).argmax() == np.array(max_combined).argmax():
                    mean_prevaMF.append(1)
                else:
                    mean_prevaMF.append(0)

                if np.array(max_sr).argmax() == np.array(max_combined).argmax():
                    mean_prevaSR.append(1)
                else:
                    mean_prevaSR.append(0)
            else:
                max_mf, max_allo, max_sr, max_combined, decision_arbi = get_best_action(agent, agent_pos = pos)
                mean_arbi.append(decision_arbi)

            max_mfs.append(np.array(max_mf).argmax())
            max_allos.append(np.array(max_allo).argmax())
            max_srs.append(np.array(max_sr).argmax())
            max_combineds.append(np.array(max_combined).argmax())

        def most_common(lst): # return the most common element of lst
            return max(set(lst), key=lst.count)

        # get the mean preferred action of all agents for the state of interest
        preferred_actions_mf.append(most_common(max_mfs))
        preferred_actions_allo.append(most_common(max_allos))
        preferred_actions_sr.append(most_common(max_srs))
        preferred_actions_combined.append(most_common(max_combineds))
        if str(type(agent)) == "<class 'agents.dolle_agent.DolleAgent'>" :
            preferred_actions_arbi.append(np.array(mean_arbi).mean())
        else:
            p_sr.append(np.array(mean_p_sr).mean())
            varmf.append(np.array(mean_varmf).mean())
            varsr.append(np.array(mean_varsr).mean())
            prevaMF.append(np.array(mean_prevaMF).mean())
            prevaSR.append(np.array(mean_prevaSR).mean())

    # put the straight egocentric direction to the north for plotting purposes
    if not agents_lst[0].mf_allo:
        preferred_actions_mf = [x+1 if x<5 else 0 for x in preferred_actions_mf]

    # transform actions indices to vectors
    preferred_vectors_mf = [actions_to_vectors[action] for action in preferred_actions_mf]
    preferred_vectors_allo = [actions_to_vectors[action] for action in preferred_actions_allo]
    preferred_vectors_sr = [actions_to_vectors[action] for action in preferred_actions_sr]
    preferred_vectors_combined = [actions_to_vectors[action] for action in preferred_actions_combined]

    # returns lists of 270 vectors (preferred orientation) for each strategy
    if str(type(agent)) != "<class 'agents.dolle_agent.DolleAgent'>" :
        return preferred_vectors_mf, preferred_vectors_allo, preferred_vectors_sr, preferred_vectors_combined, p_sr, prevaMF, prevaSR, varmf, varsr
    else:
        return preferred_vectors_mf, preferred_vectors_allo, preferred_vectors_sr, preferred_vectors_combined, preferred_actions_arbi


def plot_mean_arrows(agents_lst, prefvectors_mf, prefvectors_allo, prefvectors_sr, prefvectors_combined, nb_trials=None, decisions_arbi=None):
    """
    Plot four quivers of the heading vectors of the navigation strategies of an agent or a group of agents.
    Each quiver display 270 arrows showing the mean preferred direction of a given strategy at each state of the maze
    One quiver for egocentric MF strategy
    One quiver for allocentric MF strategy
    One quiver for goal-directed strategy
    One quiver for the coordination model strategy
    All agents in agents_lst must have been instanciated with identical parameters.

    :param agents_lst: the list of agents of interest
    :type agents_lst: Agent list
    :param prefvectors_mf: the mean preferred directions of movement of the egocentric MF strategy for each 270 states of the maze.
    :type prefvectors_mf: list of int
    :param prefvectors_allo: the mean preferred directions of movement of the allocentric MF strategy for each 270 states of the maze.
    :type prefvectors_allo: list of int
    :param prefvectors_sr: the mean preferred directions of movement of the goal-directed strategy for each 270 states of the maze.
    :type prefvectors_sr: list of int
    :param prefvectors_combined: the mean preferred directions of movement of the coordination model strategy for each 270 states of the maze.
    :type prefvectors_combined: list of int
    :param nb_trials: the number of additional training episodes to give to each agents with the new platform configuration
    :type nb_trials: int
    :param decisions_arbi: the mean preferred choice (dual between HPC or DLS) of the coordination model for each 270 states of the maze.
    :type decisions_arbi: list of int

    :returns: the axis to plot and the figure
    """

    # transforms the preferred vectors of an agent for each state into readable information for plt.quiver
    def create_vectors(agent, prefvectors):
        X = []
        Y = []
        U = []
        V = []
        for i in range(len(agent.env.grid.cart_coords.values())):
            X.append(list(agent.env.grid.cart_coords.values())[i][0])
            Y.append(list(agent.env.grid.cart_coords.values())[i][1])
            U.append(prefvectors[i][0])
            V.append(prefvectors[i][1])
        X = np.array(X)
        Y = np.array(Y)
        U = np.array(U)
        V = np.array(V)
        return X,Y,U,V

    agent = agents_lst[0] # to access general agents parameters (all agents are supposed to share the same parameters)

    platform_idx = agent.env.platform_state
    platform_coord = agent.env.grid.cart_coords[platform_idx]
    landmark_coord = agent.env.proximal_landmark_location

    if str(type(agents_lst[0])) != "<class 'agents.dolle_agent.DolleAgent'>" :
        p_sr = 0
        hpc_reliability = 0
        dls_reliability = 0
        for agent in agents_lst:
            p_sr += agent.p_sr
            hpc_reliability += agent.HPC.reliability
            dls_reliability += agent.DLS.reliability

        p_sr = p_sr/len(agents_lst)
        hpc_reliability = hpc_reliability/len(agents_lst)
        dls_reliability = dls_reliability/len(agents_lst)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

    # QUIVER 1 : EGOCENTRIC MF
    X,Y,U,V = create_vectors(agent, prefvectors_mf)
    if agents_lst[0].mf_allo or agents_lst[0].lesion_striatum:
        ax1.text(-3.3, 0, 'There is no egocentric data', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 20})
        ax1.quiver(X, Y, U, V,units='xy' ,scale=10)
    else:
        ax1.quiver(X, Y, U, V,units='xy' ,scale=1)
    ax1.grid()
    ax1.quiver(-5.5,8.5,0,2,units='xy' ,scale=1, label="platform direction",  color='r')
    ax1.quiver(-7,8.5,0,2,units='xy' ,scale=1, label="landmark direction",  color='b')
    ax1.quiver(-8.5,8.5,0,2,units='xy' ,scale=1, label="agent direction",  color='g')
    ax1.title.set_text('Mean preferred action for each state (egocentric mf)')
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # QUIVER 2 : ALLOCENTRIC MF
    X,Y,U,V = create_vectors(agent, prefvectors_allo)
    if agents_lst[0].lesion_striatum:
        ax2.text(-3.3, 0, 'There is no model-free data', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 20})
        ax2.quiver(X, Y, U, V,units='xy' ,scale=10)
    else:
        ax2.quiver(X, Y, U, V, units='xy' ,scale=1)
    ax2.grid()
    ax2.plot(platform_coord[0], platform_coord[1],'-or', label="platform (s"+str(platform_idx)+")")
    if landmark_coord is not None:
        ax2.plot(landmark_coord[0], landmark_coord[1],'-ob', label="landmark")
    ax2.title.set_text('Mean preferred action for each state (allocentric mf)')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # QUIVER 3 : HPC MODEL
    X,Y,U,V = create_vectors(agent, prefvectors_sr)
    if agents_lst[0].lesion_hippocampus:
        ax3.text(-3.3, 0, 'There is no goal-directed data', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 20})
        ax3.quiver(X, Y, U, V,units='xy' ,scale=10)
    else:
        ax3.quiver(X, Y, U, V,units='xy' ,scale=1)
    ax3.grid()
    ax3.plot(platform_coord[0], platform_coord[1],'-or', label="platform (s"+str(platform_idx)+")")
    if landmark_coord is not None:
        ax3.plot(landmark_coord[0], landmark_coord[1],'-ob', label="landmark")
    ax3.title.set_text('Mean preferred action for each state (goal directed)')
    ax3.legend()
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    # QUIVER 4 : COORDINATION MODEL
    X,Y,U,V = create_vectors(agent, prefvectors_combined)
    if str(type(agents_lst[0])) != "<class 'agents.dolle_agent.DolleAgent'>" :
        q = ax4.quiver(X, Y, U, V,  units='xy' ,scale=1)
    else:
        q = ax4.quiver(X, Y, U, V, decisions_arbi, units='xy' ,scale=1, cmap=plt.cm.viridis)
        cbar = plt.colorbar(q, cmap=plt.cm.viridis)
        cbar.set_label('P(Goal-Directed)', rotation=270, labelpad=20)
    ax4.grid()
    ax4.plot(platform_coord[0], platform_coord[1],'-or', label="platform (s"+str(platform_idx)+")")
    if landmark_coord is not None:
        ax4.plot(landmark_coord[0], landmark_coord[1],'-ob', label="landmark")
    if str(type(agents_lst[0])) == "<class 'agents.dolle_agent.DolleAgent'>" :
        ax4.title.set_text('Mean preferred action for each state (allo_mf and GD combined)')
    else:
        ax4.title.set_text('Mean preferred action for each state (allo_mf and GD combined), P(GD)= '+ str(round(p_sr,2)))
    ax4.legend()
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    if len(agents_lst) > 1:
        ax1.title.set_text('Mean preferred action for each state (egocentric mf) - '+str(len(agents_lst))+' agents')
        ax2.title.set_text('Mean preferred action for each state (allocentric mf) - '+str(len(agents_lst))+' agents')
        ax3.title.set_text('Mean preferred action for each state (goal directed) - '+str(len(agents_lst))+' agents')
        if str(type(agents_lst[0])) == "<class 'agents.dolle_agent.DolleAgent'>" :
            ax4.title.set_text('Mean preferred action for each state (allo_mf and GD combined)')
        else:
            ax4.title.set_text('Mean preferred action for each state (allo_mf and GD combined), mean P(GD)= '+ str(round(p_sr,2))+', \nmean hpc_reliability= '+ str(round(hpc_reliability,2))+', mean dls_reliability= '+ str(round(dls_reliability,2)))
    else:
        ax1.title.set_text('Preferred action for each state (egocentric mf)')
        ax2.title.set_text('Preferred action for each state (allocentric mf)')
        ax3.title.set_text('Preferred action for each state (goal directed)')
        if str(type(agents_lst[0])) == "<class 'agents.dolle_agent.DolleAgent'>" :
            ax4.title.set_text('Preferred action for each state (allo_mf and GD combined)')
        else:
            ax4.title.set_text('Preferred action for each state (allo_mf and GD combined), P(GD)= '+ str(round(p_sr,2))+', \nhpc_reliability= '+ str(round(hpc_reliability,2))+', dls_reliability= '+ str(round(dls_reliability,2)))

    return ax1, ax2, ax3, ax4, fig

# charge multiple agents objects saved as pickle at a given path
def charge_agents(path):
    file_to_read = open(path, "rb")
    agents = pickle.load(file_to_read)
    file_to_read.close()
    return agents

# return a dictionary associating the 270 states of the water-maze to cartesian coordinates
def get_coords():
    env = HexWaterMaze(10, 4, [243,230,270,257])
    coords = env.grid.cart_coords
    return coords

# n_agents, mf_allo, sr_lr, q_lr, gamma, eta, alpha1, beta1, A_alpha, A_beta, landmark_dist, HPCmode, time_limit, edge_states, lesion_HPC, lesion_DLS, dolle, inv_temp=None, inv_temp_gd=None, inv_temp_mf=None, arbi_inv_temp = None
def create_path(env_params, ag_params):
    """
    Associate multiple parameters used to run a simulation of a Morris water-maze derived task (Pearce 1998 or Rodrigo 2006),
    to form a complex path where the data resulting from the simulation is stored.
    Identical parameters to perform_rodrigo()
    :return type: str
    """
    if ag_params.dolle:
        if ag_params.inv_temp_gd is None:
            raise Exception("inv_temp_gd is undefined")
        if ag_params.inv_temp_mf is None:
            raise Exception("inv_temp_mf is undefined")
        if ag_params.arbi_inv_temp is None:
            raise Exception("arbi_inv_temp is undefined")
        path = os.path.join(str(env_params.n_agents)+str(ag_params.mf_allo)+str(ag_params.hpc_lr)+str(ag_params.q_lr)+str(ag_params.inv_temp_gd)+str(ag_params.inv_temp_mf)+str(ag_params.arbi_inv_temp)+str(ag_params.gamma)+str(ag_params.arbi_learning_rate)+str(ag_params.alpha1)+str(ag_params.beta1)+str(ag_params.A_alpha)+str(ag_params.A_beta)+str(env_params.landmark_dist)+str(ag_params.HPCmode)+str(env_params.time_limit)+str(ag_params.lesion_HPC)+str(ag_params.lesion_DLS)+str(ag_params.dolle))

    else:
        if ag_params.inv_temp is None:
            raise Exception("inv_temp is undefined")
        path = os.path.join(str(env_params.n_agents)+str(ag_params.mf_allo)+str(ag_params.hpc_lr)+str(ag_params.q_lr)+str(ag_params.inv_temp)+str(ag_params.gamma)+str(ag_params.eta)+str(ag_params.alpha1)+str(ag_params.beta1)+str(ag_params.A_alpha)+str(ag_params.A_beta)+str(env_params.landmark_dist)+str(ag_params.HPCmode)+str(env_params.time_limit)+str(ag_params.lesion_HPC)+str(ag_params.lesion_DLS)+str(ag_params.dolle))

    return path


def create_df(path, n_agents, grouped=False):
    """
    Retrieve a DataFrame containing a group of agents and envionments variables at each timestep of their respective simulations
    The rows can be grouped and averaged by agents, sessions and trials (optional, see param grouped)
    """
    if os.path.exists(path):
        saved_results_folder = path
    elif os.path.exists("../results/"+path):
        saved_results_folder = "../results/"+path
    else:
        saved_results_folder = "../saved_results/"+path

    all_data = []
    for ag in range(n_agents):
        if grouped:
            df = pd.read_csv(os.path.join(saved_results_folder, 'agent{}.csv'.format(ag)))
            summary = df.pivot_table(index=['agent', 'session', 'trial'], aggfunc='mean') # grouping
        else:
            df = pd.read_csv(os.path.join(saved_results_folder, 'agent{}.csv'.format(ag)))
            summary = df
        all_data.append(summary)

    df = pd.concat(all_data)
    df['platform location'] = df['platform'].astype('category')
    return df


def get_angle(coord1, origin, coord2):
    """
    Compute and return the angle between two points coord1 and coord2, respecting to the origin
    All parameters are cartesian coordinates, tuple of floats
    """
    a = np.array(coord1)
    b = np.array(origin)
    c = np.array(coord2)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def isinoctant(coord, platform):
    """
    Check whether a coordinate is in the octant of a platform,
    in order words, that the angle between coord, plaform and the origin is inferior to 22.5 degrees
    """
    return get_angle(coord, (0,0),platform) <= 22.5


def get_MSLE(real, expected, relative):
    """
        Computes the difference between experimental and simulated data using the mean-square-log-error method.
        The MSLE was preferred to the Mean-Square-Error method because errors values computed at different order
        of magnitude can be compared together.

        :param real: A list of the experimental data for each conditions
        :type real: List of lists (or arrays)
        :param expected: A list of the simulated data, mean of 100 agents, for each condition
        :type expected: List of lists (or arrays)
        :param relative: Whether the simulated performances are compared to
        experimental data using absolute values, or if simulated data can be fitted to real data, using an adjustment ratio
        :type relative: boolean

        :returns: The best additive MSLE between conditions, for all ratios considered
    """
    if relative:
        ratios = np.arange(0.1, 10, 0.1)
    else:
        ratios = [1.]

    real_norm = []
    expected_norm = []
    try: # pearce data
        # real_norm = minmax_scale(np.array(real).flatten(), feature_range=(50,100)).reshape((4,11))
        # expected_norm = minmax_scale(np.array(expected).flatten(), feature_range=(50,100)).reshape((4,11))

        real_norm = real
        expected_norm = expected
        return min([sum([mean_squared_error([real_norm[cond]], [np.array(expected_norm[cond])/ratio]) for cond in range(len(real_norm))]) for ratio in ratios])

    except: # rodrigo data
        if relative:
            real_norm = minmax_scale(np.array(real).flatten(), feature_range=(50,100)).reshape((2,5))
            expected_norm = minmax_scale(np.array(expected).flatten(), feature_range=(50,100)).reshape((2,5))
        else:
            real_norm = np.array(real).flatten().reshape(2,5)
            expected_norm = np.array(expected).flatten().reshape(2,5)
        return min([sum([mean_squared_error([real_norm[cond]], [np.array(expected_norm[cond])/ratio]) for cond in range(len(real_norm))]) for ratio in ratios])*4.4


# print every environment's and agent's parameters into a file
def save_params_in_txt(results_folder, env_params, ag_params):

    f = open(results_folder+"/parameters.txt",'w')
    [print(key,':',value, file=f) for key, value in vars(env_params).items()]
    [print(key,':',value, file=f) for key, value in vars(ag_params).items()]
    f.close()

# ____________________________________
# Original code from Geerts 2020 below
# ____________________________________

def random_argmax(x):
    """Argmax operation, but if there are multiple maxima, return one randomly.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    b = np.flatnonzero(arg_maxes)
    choice = np.random.choice(b)
    return choice


def all_argmax(x):
    """Argmax operation, but if there are multiple maxima, return all.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    indices = np.flatnonzero(arg_maxes)
    return indices


def softmax(x, beta=2):
    """Compute the softmax function.
    :param x: Data
    :param beta: Inverse temperature parameter.
    :return:
    """
    x = np.array(x)
    x = np.array([Decimal(i) for i in x])
    res = np.exp(beta * x) / sum(np.exp(beta * x))
    return np.float64(res)


def rotation_matrix_2d(angle):
    """
    :param angle: In radians
    :return:
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def to_agent_frame(object_location, agent_location, agent_direction):
    """Shift reference frame to agent's current location and direction.

    :param object_location:
    :param agent_location:
    :param agent_direction:
    :return:
    """
    translate = np.array(object_location) - np.array(agent_location)
    rotation_mat = rotation_matrix_2d(agent_direction).T
    result = rotation_mat.dot(translate)
    return np.asarray(result).squeeze()


def get_relative_angle(angle_1, angle_2):
    """Return the smallest difference in angles between two angles (in degrees).
    """
    a = angle_1 - angle_2
    a = (a + 180) % 360 - 180
    return a


def angle_to_landmark(agent_location, landmark_centre, agent_orientation):
    """Get the relative direction to the landmark from the viewpoint of the

    :return:
    """
    if landmark_centre is None:
        return 0
    relative_cue_pos = to_agent_frame(landmark_centre, agent_location, np.radians(agent_orientation))
    angle = np.arctan2(relative_cue_pos[1], relative_cue_pos[0])
    return np.degrees(angle)


def generate_random_policy(env):
    """Generate a random policy assigning equal probability to all possible actions in each state.

    :param env: Environment object to be evaluated.
    :return: Nested list[state][action] giving the probabilities of choosing [action] when in [state].
    """
    random_policy = []
    for state in env.state_indices:
        possible_actions = env.get_possible_actions(state)
        if not possible_actions:
            random_policy.append([[]])
            continue
        rand_pol = [1 / len(possible_actions)] * len(possible_actions)
        random_policy.append(rand_pol)
    return random_policy


def value_iteration(env, theta=1e-4, gamma=.9):
    """Implement the  value iteration algorithm from dynamic programming to compute the optimal policy and corresponding
    state-value function for a given environment.

    :param env: Environment object to be evaluated.
    :param theta: Cutoff criterion for convergence of the optimal policy.
    :param gamma: Exponential discount factor for future rewards.
    :return: List containing the optimal policy, and array containing values for each state.
    """
    optimal_values = find_optimal_values(env, gamma=gamma, theta=theta)
    optimal_policy = optimal_policy_from_value(env, optimal_values, gamma)
    return optimal_policy, optimal_values
