import copy
import math
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from matplotlib.patches import RegularPolygon


class Environment(object):
    """Parent class for RL environments holding some general methods.
    Identical to Geerts original code
    """
    def __init__(self):
        self.nr_states = None
        self.nr_actions = None
        self.actions = None
        self.adjacency_graph = None
        self.goal_state = None
        self.reward_func = None
        self.graph = None
        self.n_features = None
        self.rf = None
        self.transition_probabilities = None
        self.terminal_state = None
        self.state_indices = None
        self.current_state = None
        self.landmark_location = None
        self.grid = None
        self.ego_angles = None
        self.allo_angles = None
        self.action_directions = None

    def act(self, action):
        pass

    def get_goal_state(self):
        pass

    def get_current_state(self):
        return self.current_state

    def reset(self, random_loc=True):
        pass

    def define_adjacency_graph(self):
        pass

    def _fill_adjacency_matrix(self):
        pass

    def get_adjacency_matrix(self):
        if self.adjacency_graph is None:
            self._fill_adjacency_matrix()
        return self.adjacency_graph

    def create_graph(self):
        """Create networkx graph from adjacency matrix.
        """
        self.graph = nx.from_numpy_array(self.get_adjacency_matrix())

    def show_graph(self, map_variable=None, layout=None, node_size=1500, **kwargs):
        """Plot graph showing possible state transitions.

        :param node_size:
        :param map_variable: Continuous variable that can be mapped on the node colours.
        :param layout:
        :param kwargs: Any other drawing parameters accepted. See nx.draw docs.
        :return:
        """
        if layout is None:
            layout = nx.spring_layout(self.graph)
        if map_variable is not None:
            categories = pd.Categorical(map_variable)
            node_color = categories
        else:
            node_color = 'b'
        nx.draw(self.graph, with_labels=True, pos=layout, node_color=node_color, node_size=node_size, **kwargs)

    def set_reward_location(self, state_idx, action_idx):
        self.goal_state = state_idx
        action_destination = self.transition_probabilities[state_idx, action_idx]
        self.reward_func = np.zeros([self.nr_states, self.nr_actions, self.nr_states])
        self.reward_func[state_idx, action_idx] = action_destination

    def is_terminal(self, state_idx):
        if not self.get_possible_actions(state_idx):
            return True
        else:
            return False

    def get_destination_state(self, current_state, current_action):
        transition_probabilities = self.transition_probabilities[current_state, current_action]
        return np.flatnonzero(transition_probabilities)

    def get_degree_mat(self):
        degree_mat = np.eye(self.nr_states)
        for state, degree in self.graph.degree:
            degree_mat[state, state] = degree
        return degree_mat

    def get_laplacian(self):
        return self.get_degree_mat() - self.adjacency_graph

    def get_normalised_laplacian(self):
        """Return the normalised laplacian.
        """
        D = self.get_degree_mat()
        L = self.get_laplacian()  # TODO: check diff with non normalised laplacian. check adverserial examples
        exp_D = utils.exponentiate(D, -.5)
        return exp_D.dot(L).dot(exp_D)

    def compute_laplacian(self, normalization_method=None):
        """Compute the Laplacian.

        :param normalization_method: Choose None for unnormalized, 'rw' for RW normalized or 'sym' for symmetric.
        :return:
        """
        if normalization_method not in [None, 'rw', 'sym']:
            raise ValueError('Not a valid normalisation method. See help(compute_laplacian) for more info.')

        D = self.get_degree_mat()
        L = D - self.adjacency_graph

        if normalization_method is None:
            return L
        elif normalization_method == 'sym':
            exp_D = utils.exponentiate(D, -.5)
            return exp_D.dot(L).dot(exp_D)
        elif normalization_method == 'rw':
            exp_D = utils.exponentiate(D, -1)
            return exp_D.dot(L)

    def get_possible_actions(self, state_idx):
        pass

    def get_adjacent_states(self, state_idx):
        pass

    def compute_feature_response(self):
        pass

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])
        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            actions = self.get_possible_actions(state)
            for a, action in enumerate(actions):
                transition_matrix[state] += self.transition_probabilities[state, a] * policy[state][a]

        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def get_next_state(self, state_idx, a):
        pass

    def get_state_location(self, idx):
        pass


class HexGrid(object):
    """Using only a radius parameter, allows to build a hexagonal, discrete-state grid, to model a circular maze.
    Gives a cartesian coordinate to each state of the hexagon. Allows to retrieve the direct neighbors of a state.
    Identical to Geerts original code
    """
    def __init__(self, radius, edge_states):
        self.deltas = [[1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0]]
        self.radius = radius
        self.cube_coords = {0: (0, 0, 0)}
        self.edge_states = edge_states
        state = 1
        for r in range(radius):
            a = 0
            b = -r
            c = +r
            for j in range(6):
                num_of_hexes_in_edge = r
                for i in range(num_of_hexes_in_edge):
                    a = a + self.deltas[j][0]
                    b = b + self.deltas[j][1]
                    c = c + self.deltas[j][2]
                    self.cube_coords[state] = (a, b, c)
                    # if r == radius - 1:
                    #     self.edge_states.append(state)
                    state += 1

        self.cart_coords = {state: self.to_cartesian(coord) for state, coord in self.cube_coords.items()}
        self.size = len(self.cube_coords)

    def get_adjacency(self):
        adjacency_matrix = np.zeros((len(self.cube_coords), len(self.cube_coords)))
        for state, coord in self.cube_coords.items():
            for d in self.deltas:
                a = coord[0] + d[0]
                b = coord[1] + d[1]
                c = coord[2] + d[2]
                neighbour = self.get_state_id((a, b, c))
                if neighbour is not None:
                    adjacency_matrix[state, neighbour] = 1
        return adjacency_matrix

    def get_sas_transition_mat(self):
        """Fill and return the state by action by state transition matrix.

        :return:
        """
        sas_matrix = np.zeros((len(self.cube_coords), len(self.deltas), len(self.cube_coords)))
        for state, coord in self.cube_coords.items():
            for i, d in enumerate(self.deltas):
                a = coord[0] + d[0]
                b = coord[1] + d[1]
                c = coord[2] + d[2]
                neighbour = self.get_state_id((a, b, c))
                if neighbour is not None:
                    sas_matrix[state, i, neighbour] = 1.
                else:  # if a wall state is the neighbour
                    sas_matrix[state, i, state] = 1.
        return sas_matrix

    def get_state_id(self, cube_coordinate):
        for state, loc in self.cube_coords.items():
            if loc == cube_coordinate:
                return state
        return None

    def is_state_location(self, coordinate):
        """Return true if cube coordinate exists.

        :param coordinate: Tuple cube coordinate
        :return:
        """
        for state, loc in self.cube_coords.items():
            if loc == coordinate:
                return True
        return False

    @staticmethod
    def to_cartesian(coordinate):
        xcoord = coordinate[0]
        ycoord = 2. * np.sin(np.radians(60)) * (coordinate[1] - coordinate[2]) / 3.
        return xcoord, ycoord

    def plot_grid(self, platforms=None):

        fig, ax = plt.subplots(1)

        ax.set_aspect('equal')
        for x, y in self.cart_coords.values():
            hex_patch = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                       orientation=np.radians(30), alpha=0.2, edgecolor='k')
            ax.add_patch(hex_patch)

        lower_bound = min(min(self.cart_coords.values()))
        upper_bound = max(max(self.cart_coords.values()))
        plt.xlim([lower_bound - 2, upper_bound + 2])
        plt.ylim([lower_bound - 2, upper_bound + 2])
        return fig, ax

    def distance(self, state_a, state_b):
        return euclidean(self.cart_coords[state_a], self.cart_coords[state_b])


class HexWaterMaze(Environment):
    """Model of a Morris water-maze pool which consist in a hexagonal grid with discrete states.
    Possess 3 dimensions which can be translated as 2D cartesian coordinates. Is used for the simulation of rats
    spatial navigation experiments such as Pearce 1998 and Rodrigo 2006.
    Several functions in the following lines of code were entirely copied or strongly inspired from the original code of Geerts 2020.
    """
    def __init__(self, radius, landmark_dist, edge_states):
        super().__init__()
        self.grid = HexGrid(radius, edge_states)
        # stored transitions, as using np.flatnonzero caused dramatical computational cost increase (see get_next_state())
        self.transitions = {}
        self.adjacency_graph = self.grid.get_adjacency()
        self.transition_probabilities = self.grid.get_sas_transition_mat()
        self.action_labels = ['N', 'NE', 'SE', 'S', 'SW', 'NW']
        self.actions = self.grid.deltas
        self.action_directions = []
        for a in self.actions:
            mv = self.grid.to_cartesian(a)
            self.action_directions.append(np.arctan2(mv[1], mv[0]))
        self.other_terminals = []
        self.nr_actions = len(self.actions)
        self.nr_states = self.grid.size
        self.state_indices = list(range(self.nr_states))
        states_close_to_centre = [i for i in self.state_indices if
                                  euclidean(self.grid.cart_coords[i], self.grid.cart_coords[0]) < radius / 3]
        self.platform_state = np.random.choice([i for i in self.state_indices if not i in states_close_to_centre])
        self.previous_platform_state = None
        self.reward_func = np.zeros((self.nr_states, self.nr_actions, self.nr_states))
        self.set_reward_func()
        # MODIF
        self.landmark_dist = landmark_dist
        self.set_landmark()
        self.set_distal_landmark()
        self.starting_state = 0
        self.allo_angles = np.array([30, 90, 150, 210, 270, 330])
        self.ego_angles = np.array([0, 60, 120, -180, -120, -60])  # q values correspond to these
        self.eligible_start_states = []
        self.agent_orientation = 90

    def set_platform_state(self, state_idx):
        self.previous_platform_state = self.platform_state
        self.platform_state = state_idx
        self.set_landmark()
        self.set_distal_landmark()
        self.set_reward_func()

    def add_terminal(self, state):
        self.other_terminals.append(state)

    def set_reward_func(self):
        for state in self.state_indices:
            for action in range(self.nr_actions):
                next_state, reward = self.get_next_state_and_reward(state, action)
                self.reward_func[state, action, next_state] = reward

    def set_landmark(self): # set landmark next to the platform

        platform_loc = self.grid.cube_coords[self.platform_state]
        # the hexagonal grid has 3 dimensions
        # here the second dimension is considered as the Y dimension of a cartesian coordinate system
        landmark_loc = (platform_loc[0], platform_loc[1]+self.landmark_dist, platform_loc[2])
        self.landmark_location = self.grid.to_cartesian(landmark_loc)

    def set_distal_landmark(self):

        platform_loc = self.grid.cart_coords[self.platform_state]
        landmark_loc = (platform_loc[0]*2, platform_loc[1]*2)
        self.distal_landmark_location = landmark_loc

    def get_next_state(self, current_state, action):
        # stored transitions, as using np.flatnonzero caused dramatical computational cost increase
        if (current_state, action) in self.transitions:
            next_state = self.transitions[(current_state, action)]
        else:
            self.transitions[(current_state,action)] = np.flatnonzero(self.transition_probabilities[current_state, action])[0]
            next_state = self.transitions[(current_state, action)]
        return next_state

    def get_reward(self, next_state):
        return next_state == self.platform_state

    def get_next_state_and_reward(self, current_state, action):
        # If current state is terminal absorbing state:
        if self.is_terminal(current_state):
            return current_state, 0

        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(next_state)
        return next_state, reward

    def act(self, action):
        next_state, reward = self.get_next_state_and_reward(self.current_state, action)
        orientation = self.get_orientation(self.current_state, next_state, self.agent_orientation)
        self.agent_orientation = orientation
        self.current_state = next_state

        return next_state, reward, orientation

    def get_orientation(self, state, next_state, current_orientation):
        if state == next_state:
            return current_orientation
        s1 = self.get_state_location(state)
        s2 = self.get_state_location(next_state)
        return np.degrees(np.arctan2(s2[1] - s1[1], s2[0] - s1[0]))

    # reset the environment as it should be at the beginning of an episode
    def reset(self):
        # select starting state in eligible_states list
        if len(self.eligible_start_states) == 0:
            self.eligible_start_states = copy.copy(self.grid.edge_states)
        eligible_start_states = np.array(self.eligible_start_states)
        self.starting_state = np.random.choice(eligible_start_states)
        self.eligible_start_states.remove(self.starting_state)
        self.current_state = self.starting_state

        # orient agent
        if self.landmark_location is not None:
            self.orient_agent_to_platform()

    def get_state_location(self, state, cube_system=False):
        if cube_system:
            return self.grid.cube_coords[state]
        else:
            return self.grid.cart_coords[state]

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])
        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            for action in range(self.nr_actions):
                transition_matrix[state] += self.transition_probabilities[state, action] * policy[state][action]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        """Compute the Successor Representation through inversion of the transition matrix.

        :param (list) policy: Nested list containing the action probabilities for each state.
        :param (float) gamma: Discount parameter
        :return:
        """
        transition_matrix = self.get_transition_matrix(policy)
        print(transition_matrix[240])
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def get_possible_actions(self, state_idx):
        if self.is_terminal(state_idx):
            return []
        else:
            return list(range(self.nr_actions))

    def is_terminal(self, state_idx):
        if state_idx == self.platform_state:
            return True
        elif state_idx in self.other_terminals:
            return True
        else:
            return False

    def plot_grid(self, c_mappable=None, ax=None, show_state_idx=True, alpha=1., c_map=None, platforms=None):
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

    def plot_occupancy_on_grid(self, trial_results, **kwargs):
        color_pal = sns.color_palette()
        state_occupancy = trial_results['state'].astype(int).value_counts()
        occupancy = np.zeros(self.nr_states)
        for s, count in state_occupancy.iteritems():
            occupancy[s] = count
        ax = self.plot_grid(occupancy, **kwargs)

        platform = trial_results['platform'].iloc[0]
        previous_platform = trial_results['previous platform'].iloc[0]

        hex_patch = RegularPolygon(self.grid.cart_coords[platform], numVertices=6, radius=2./2.5,
                                   facecolor=color_pal[8], orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)
        hex_patch = RegularPolygon(self.grid.cart_coords[previous_platform], numVertices=6, radius=2./2.5,
                                   facecolor=color_pal[9], orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)

        # add legend
        hex_patch = RegularPolygon((6, 10), numVertices=6, radius=2. / 3.,
                                   facecolor=color_pal[8],
                                   orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)
        hex_patch = RegularPolygon((6, 8.5), numVertices=6, radius=2. / 3.,
                                   facecolor=color_pal[9],
                                   orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)

        ax.text(x=7, y=10, s='Platform', va='center')
        ax.text(x=7, y=8.5, s='Previous platform', va='center')

        start = trial_results['state'].iloc[0]
        hex_patch = RegularPolygon(self.grid.cart_coords[start], numVertices=6, radius=2./2.5,
                                   facecolor=color_pal[2],
                                   orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)
        ax.text(self.grid.cart_coords[start][0], self.grid.cart_coords[start][1], 'S',
                ha='center', va='center', size=10)

        ax.axis('off')
        return ax

    def set_angle_beacon(self, angle):

        def rotate(origin, point, angle):
            angle = math.radians(angle)
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """
            ox, oy = origin
            px, py = point

            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy
        self.landmark_location = rotate((0,0), self.landmark_location, angle)

# _____________________________

    def get_goal_state(self):
        return self.platform_state

    def delete_plaform_and_landmarks(self):
        self.previous_platform_state = self.platform_state
        self.platform_state = None
        self.landmark_location = None
        self.distal_landmark_location = None
        self.set_reward_func()

    def delete_plaform(self):
        self.previous_platform_state = self.platform_state
        self.platform_state = None
        self.set_reward_func()

    def delete_landmarks(self):
        self.landmark_location = None
        self.distal_landmark_location = None

    def delete_proximal_landmark(self):
        self.landmark_location = None

    def delete_distal_landmark(self):
        self.distal_landmark_location = None

    def orient_agent_to_platform(self):
        possible_orientations = np.round(np.degrees(self.action_directions))
        angles = []
        for i, o in enumerate(possible_orientations):
            angle = utils.angle_to_landmark(self.get_current_state(), self.agent_orientation, self.landmark_location)
            angles.append(angle)
        orientation = possible_orientations[np.argmin(np.abs(angles))]
        self.agent_orientation = orientation
        return orientation

    def one_episode(self, agent, time_limit):
        """
        Run an episode of any Morris water-maze derived experiment

        :param agent: the agent to be subjected to the episode
        :type agent: Agent
        :param time_limit: max number of timestep to find the reward, the episode is forced to end if reached
        :type a: int

        :returns type: float
        """

        agent.setup()
        self.reset()
        orientation = self.agent_orientation
        t = 0
        s = self.get_current_state()

        agent.init_saving(t, s) # to initialize agent's and environment variables to store, erase previous episode data

        # run until the agent find the platform or reach the time limit
        while not self.is_terminal(s) and t < time_limit:

            allo_a, ego_a = agent.take_decision(s, orientation) # get the agent's allocentric and egocentric preferred action
            previous_state = s
            s, reward, orientation = self.act(allo_a) # actually perform the preferred action of the agent
            agent.update(previous_state, reward, s, allo_a, ego_a, orientation) # update the agent modules' weights
            t += 1
            agent.save(t, s) # save agent's and environment's variables

        # if a platform exists, but the agent did not find it and the time limit was reached,
        # simulate the agent being dropped on the platform by the hand of the experimenter
        if t == time_limit and not self.is_terminal(s) and self.get_goal_state() is not None:
            # the agent is dropped at distance 1 from the platform,
            # it works because all successive states are neighbors in self.env.state_indices
            if self.get_goal_state()+1 <= max(self.state_indices):
                self.current_state = self.get_goal_state()+1
                s = self.get_goal_state()+1
            else:
                self.current_state = self.get_goal_state()-1
                s = self.get_goal_state()-1

            # to simulate a transition to the platform
            next_state = self.get_goal_state()
            ns = 0
            act = 0
            for a in range(0,6): # iteratively search the correct action to transition from s to the platform
                ns = self.get_next_state_and_reward(s,a)[0]
                if ns == next_state:
                    act = a
                    break

            # update the agent's modules weights after the forced transitioning to the platform
            ego_a = agent.DLS.get_ego_action(act, orientation)
            agent.update(previous_state, reward, s, act, ego_a, orientation)

        return pd.DataFrame.from_dict(agent.results) # returns the agent's and environment's saved variables (to put in log files)
