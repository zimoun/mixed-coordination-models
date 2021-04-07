import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from environments.HexWaterMaze import *
import utils
from copy import deepcopy
import pickle
import pandas as pd

def plot_img(path, figsize=(10,10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mpimg.imread(path))
    plt.axis('off')
    plt.show()

def plot_2img(path1, path2, figsize=(10,10)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].imshow(mpimg.imread(path1))
    axs[1].imshow(mpimg.imread(path2))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_frame_on(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.axis('off')

    plt.show()
    plt.close()

# maze_size:int, platforms: int list
def plot_geerts(maze_size, platforms=None, landmarks=None, figsize=(4,4)):
    plt.figure(figsize=figsize)
    hwm = HexWaterMaze(maze_size, 1, edge_states = [])
    plot_grid(hwm, platforms=platforms, landmarks=landmarks)

def plot_grid(self, c_mappable=None, ax=None, show_state_idx=True, alpha=1., c_map=None, platforms=None, landmarks=None):
    """

    :param show_state_idx:
    :param (np.array) c_mappable:
    :return:
    """

    # TODO: move to plotting module, make class for plotting hex grids.
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



def get_maze(size_maze, landmark_dist, edge_states):
    if size_maze == 6:
        possible_platform_states = np.array([48, 51, 54, 57, 60, 39, 42, 45])
        g = HexWaterMaze(6, landmark_dist, edge_states)
        return possible_platform_states, g
    if size_maze == 10:
        # [192, 185, 181, 174, 216, 210, 203, 197]
        possible_platform_states = np.array([48, 52, 118, 122, 126, 94, 98, 44])
        g = HexWaterMaze(10, landmark_dist, edge_states)
        return possible_platform_states, g
    if size_maze == 12:
        possible_platform_states = np.array([300, 292, 284, 277, 329, 321, 313, 306])
        g = HexWaterMaze(12, landmark_dist, edge_states)
        return possible_platform_states, g


def determine_platform_seq(g, platform_states, n_sessions):
    # MODIF

    plat_seq = None
    try:
        indices = np.arange(len(platform_states))
        usage = np.zeros(len(platform_states))

        plat_seq = [np.random.choice(platform_states)]
        for sess in range(1, n_sessions):

            distances = np.array([g.grid.distance(plat_seq[sess - 1], s) for s in platform_states])

            # MODIF

            candidates = indices[np.logical_and(usage < n_sessions/8+1, distances > g.grid.radius*0.8)]
            #candidates = indices[usage < n_sessions/8+1]

            platform_idx = np.random.choice(candidates)

            plat_seq.append(platform_states[platform_idx])
            usage[platform_idx] += 1.
    except Exception:
        return determine_platform_seq(g, platform_states, n_sessions)

    return plat_seq

# return the best action from a given state, for each strategy: mf_ego, mf_allo, sr and combined
def get_best_action(agent, agent_pos):

    # give the 6 different orientation of the agent
    possible_orientations = np.round(np.degrees(agent.env.action_directions))

    angles = []


    # get angle of the landmark considering each possible orientation of the agent
    for i, o in enumerate(possible_orientations):
        angle = utils.angle_to_landmark(agent.env.get_state_location(agent_pos), agent.env.landmark_location, np.radians(o))
        angles.append(angle)
    #print("angles: ", angles)
    # orientation of the agent is the one in which its angle to the landmark is the smallest
    if agent.mf_allo:
        Q_combined, Q_mf, Q_allo, Q_sr = agent.compute_Q(agent_pos, agent.p_sr)
    else:
        orientation = possible_orientations[np.argmin(np.abs(angles))]

        # select action
        # dim 6, same as Q_mf but for combined and allo
        Q_combined, Q_mf, Q_allo, Q_sr = agent.compute_Q(agent_pos, agent.p_sr, orientation)
    # act
    #next_state, reward = agent.env.act(Q_allo)
    return np.argmax(Q_mf), np.argmax(Q_allo), np.argmax(Q_sr), np.argmax(Q_combined)

#
# # return the best action from a given state, for each strategy: mf_ego, mf_allo, sr and combined
# def get_best_action_allo(agent, agent_pos):
#
#
#     # select action
#     # dim 6, same as Q_mf but for combined and allo
#     Q_combined, Q_mf, Q_allo, Q_sr = agent.compute_Q_allo(agent_pos, agent.p_sr)
#
#     # act
#     #next_state, reward = agent.env.act(Q_allo)
#     return np.argmax(Q_mf), np.argmax(Q_allo), np.argmax(Q_sr), np.argmax(Q_combined)

# take and agents list and a platform location and return a list of the preferred
# action of each agent for each location
def get_mean_preferred_dirs(agents_lst, platform_idx=None, nb_trials=None):

    def most_common(lst):
        return max(set(lst), key=lst.count)

    # All res wil contain preferred directions from 0 to 6
    res_mf = []
    res_allo = []
    res_sr = []
    res_combined = []

    actions_to_cardinals = {}
    for action in range(0,6):
        next_state = list(agents_lst[0].env.transition_probabilities[0,action]).index(1)
        actions_to_cardinals[action] = agents_lst[0].env.grid.cart_coords[next_state]

    if platform_idx is not None:

        for agent in agents_lst:
            agent.env.set_platform_state(platform_idx)
            for i in range(nb_trials):
                agent.one_episode(1000, random_policy=False)
            # else:
            #     agent.one_episode(1000, random_policy=False)
            #     agent.one_episode(1000, random_policy=False)
            #     agent.one_episode(1000, random_policy=False)
            #     agent.one_episode(1000, random_policy=False)

    for pos in range(0,271):
        max_mfs = []
        max_allos = []
        max_srs = []
        max_combineds = []

        for agent in agents_lst:
            # if agent.mf_allo:
            #     max_mf, max_allo, max_sr, max_combined = get_best_action_allo(agent, agent_pos = pos)
            # else:
            max_mf, max_allo, max_sr, max_combined = get_best_action(agent, agent_pos = pos)
            max_mfs.append(max_mf)
            max_allos.append(max_allo)
            max_srs.append(max_sr)
            max_combineds.append(max_combined)

        res_mf.append(most_common(max_mfs))
        res_allo.append(most_common(max_allos))
        res_sr.append(most_common(max_srs))
        res_combined.append(most_common(max_combineds))

    # put the straight ego direction to the north for plotting purposes
    if not agents_lst[0].mf_allo:
        res_mf = [x+1 if x<5 else 0 for x in res_mf]

    # all res2 contains cardinals coords
    res2_mf = [actions_to_cardinals[action] for action in res_mf]
    res2_allo = [actions_to_cardinals[action] for action in res_allo]
    res2_sr = [actions_to_cardinals[action] for action in res_sr]
    res2_combined = [actions_to_cardinals[action] for action in res_combined]

    return res2_mf, res2_allo, res2_sr, res2_combined


def plot_mean_arrows(agents_lst, res2_mf, res2_allo, res2_sr, res2_combined, nb_trials=None):

    def create_vectors(agent, Q_data):

        X = []
        Y= []
        U=[]
        V=[]

        for i in range(len(agent.env.grid.cart_coords.values())):
            X.append(list(agent.env.grid.cart_coords.values())[i][0])
            Y.append(list(agent.env.grid.cart_coords.values())[i][1])
            U.append(Q_data[i][0])
            V.append(Q_data[i][1])

        X = np.array(X)
        Y = np.array(Y)
        U = np.array(U)
        V = np.array(V)

        return X,Y,U,V

    agent = agents_lst[0]

    platform_idx = agent.env.platform_state
    platform_coord = agent.env.grid.cart_coords[platform_idx]
    landmark_coord = agent.env.landmark_location

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

    X,Y,U,V = create_vectors(agent, res2_mf)
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

    X,Y,U,V = create_vectors(agent, res2_allo)
    if agents_lst[0].lesion_striatum:
        ax2.text(-3.3, 0, 'There is no model-free data', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 20})
        ax2.quiver(X, Y, U, V,units='xy' ,scale=10)
    else:
        ax2.quiver(X, Y, U, V,units='xy' ,scale=1)
    ax2.grid()
    ax2.plot(platform_coord[0], platform_coord[1],'-or', label="platform (s"+str(platform_idx)+")")
    ax2.plot(landmark_coord[0], landmark_coord[1],'-ob', label="landmark")
    ax2.title.set_text('Mean preferred action for each state (allocentric mf)')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    X,Y,U,V = create_vectors(agent, res2_sr)
    if agents_lst[0].lesion_hippocampus:
        ax3.text(-3.3, 0, 'There is no goal-directed data', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 20})
        ax3.quiver(X, Y, U, V,units='xy' ,scale=10)
    else:
        ax3.quiver(X, Y, U, V,units='xy' ,scale=1)
    ax3.grid()
    ax3.plot(platform_coord[0], platform_coord[1],'-or', label="platform (s"+str(platform_idx)+")")
    ax3.plot(landmark_coord[0], landmark_coord[1],'-ob', label="landmark")
    ax3.title.set_text('Mean preferred action for each state (successor representation)')
    ax3.legend()
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    X,Y,U,V = create_vectors(agent, res2_combined)
    ax4.quiver(X, Y, U, V,units='xy' ,scale=1)
    ax4.grid()
    ax4.plot(platform_coord[0], platform_coord[1],'-or', label="platform (s"+str(platform_idx)+")")
    ax4.plot(landmark_coord[0], landmark_coord[1],'-ob', label="landmark")
    ax4.title.set_text('Mean preferred action for each state (allo_mf and SR combined), P(SR)= '+ str(round(p_sr,2)))
    ax4.legend()
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    if len(agents_lst) > 1:
        ax1.title.set_text('Mean preferred action for each state (egocentric mf) - '+str(len(agents_lst))+' agents')
        ax2.title.set_text('Mean preferred action for each state (allocentric mf) - '+str(len(agents_lst))+' agents')
        ax3.title.set_text('Mean preferred action for each state (successor representation) - '+str(len(agents_lst))+' agents')
        ax4.title.set_text('Mean preferred action for each state (allo_mf and SR combined), mean P(SR)= '+ str(round(p_sr,2))+', \nmean hpc_reliability= '+ str(round(hpc_reliability,2))+', mean dls_reliability= '+ str(round(dls_reliability,2)))
    else:
        ax1.title.set_text('Preferred action for each state (egocentric mf)')
        ax2.title.set_text('Preferred action for each state (allocentric mf)')
        ax3.title.set_text('Preferred action for each state (successor representation)')
        ax4.title.set_text('Preferred action for each state (allo_mf and SR combined), P(SR)= '+ str(round(p_sr,2))+', \nhpc_reliability= '+ str(round(hpc_reliability,2))+', dls_reliability= '+ str(round(dls_reliability,2)))

    return ax1, ax2, ax3, ax4, fig

def clone_agents(agents, nbr_copy):
    agents_dup = []
    for agent in agents:
        for n in range(nbr_copy):
            agents_dup.append(deepcopy(agent))
    return agents_dup



def exp3():

    exp = "first_exp_pearce"
    maze_size = 10
    n_sessions = 11
    n_trials = 4
    n_agents = 100
    mf_allo = False
    sr_lr = 0.07
    q_lr = 0.07
    inv_temp = 25
    gamma = 0.95
    eta = 0.03 # reliability learning rate
    mpe = 1 # maximum prediction error
    A_alpha = 1.8 # Steepness of transition curve MF to SR
    A_beta = 1.1 # Steepness of transition curve SR to MF
    landmark_dist = 4

    agents_group1 = perform_pearce(exp, maze_size, n_trials, n_sessions, n_agents, mf_allo, sr_lr, q_lr, inv_temp, gamma, eta, mpe, A_alpha, A_beta, landmark_dist, show_quiv=False, show_perfs=False)
    agents_group2 = perform_pearce(exp, maze_size, n_trials, n_sessions, n_agents, mf_allo, sr_lr, q_lr, inv_temp, gamma, eta, mpe, A_alpha, A_beta, landmark_dist, show_quiv=False, show_perfs=False)


    # agents_group1_dup = clone_agents(agents_group1, 20)
    # agents_group2_dup = clone_agents(agents_group2, 20)



    return agents_group1, agents_group2


def get_mean_escape_time(agents, landmark_dist, allo):

    df_agents = []
    id_agent = 0
    for agent in agents:
        # print(agent.env.platform_state)
        #print(agent.HPC.R_hat)
        #agent.HPC.R_hat = np.zeros(agent.env.nr_states)
        agent.env.landmark_dist = landmark_dist
        agent.env.set_platform_state(0)
        #print(agent.HPC.R_hat)
        # if allo:
        #     res = agent.one_episode_allo(random_policy=False)
        # else:
        res = agent.one_episode(random_policy=False)
        res['escape time'] = res.time.max()
        res['id_agent'] = id_agent
        df_agents.append(res)
        id_agent += 1

    df_agents = pd.concat(df_agents)
    result_df = df_agents.pivot_table(index=["id_agent"], aggfunc='mean')

    return result_df

def charge_agents(path):
    file_to_read = open(path, "rb")
    agents = pickle.load(file_to_read)
    file_to_read.close()
    return agents

def run_test(allo, landmark_dist):

    if allo:
        agents = charge_agents("agents_allo_group_normal.p")
    else:
        agents = charge_agents("agents_group_normal.p")

    df = get_mean_escape_time(agents, landmark_dist, allo)

    del agents

    return df


def exp3_pearce(allo=False, agents_per_group=100, alt_dist=-4):

    nb_iter=agents_per_group/100
    df_normal = []
    df_inverted = []

    for n in range(int(nb_iter)):
        df_normal.append(run_test(allo, 4))
        df_inverted.append(run_test(allo, alt_dist))

    df_normal = pd.concat(df_normal)
    df_inverted = pd.concat(df_inverted)


    x = ["normal", "inverted"]
    y = [df_normal["escape time"].mean(),df_inverted["escape time"].mean()]


    # dfplot.bar(yerr=df['std'].unstack(level=1) * 1.96, ax=ax, capsize=4)
    fig,(ax1)=plt.subplots(1,1)
    ax1.bar(x, height=y, yerr=[df_normal["escape time"].std(),df_inverted["escape time"].std()])
    ax1.title.set_text("Mean escape time for normal (2) and inverted ("+str(int(alt_dist/2))+") landmark distances (n_agent=)"+str(agents_per_group)+")")
    ax1.set_xlabel('landmark position')
    ax1.set_ylabel('mean escape time')
    #ax1.errorbar(x, y, yerr=[df_normal["escape time"].std(),df_inverted["escape time"].std()], label="standard deviation")
    #plt.legend()
    plt.show()

    if allo:
        agents = charge_agents("agents_allo_group_normal.p")
    else:
        agents = charge_agents("agents_group_normal.p")

    for ag in agents:
        #ag.HPC.R_hat = np.zeros(ag.env.nr_states)
        ag.env.landmark_dist = 4
    res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs(agents, 0)
    ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined)
    fig.suptitle("Strategies used with normal landmarks directions", fontsize=14)
    for ag in agents:
        #ag.HPC.R_hat = np.zeros(ag.env.nr_states)
        ag.env.landmark_dist = alt_dist

    res2_mf, res2_allo, res2_sr, res2_combined = get_mean_preferred_dirs(agents, 0)
    ax1, ax2, ax3, ax4, fig = plot_mean_arrows(agents, res2_mf, res2_allo, res2_sr, res2_combined)
    fig.suptitle("Strategies used with inverted landmarks directions", fontsize=14)

    plt.show()
    plt.close()
