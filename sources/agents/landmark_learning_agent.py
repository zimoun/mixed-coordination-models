import utils
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from agents.agent import Agent, AssociativeAgent, FirstOrderAgent


class LandmarkLearningAgent(Agent, AssociativeAgent, FirstOrderAgent):
    """Q-learning agent using landmark features
    """
    max_RPE = 1 # see update_reliability()

    def __init__(self, env, gamma, learning_rate, inv_temp, eta, allo):
        """
        :param env: the environment
        :type env: Environment
        :param gamma: discount factor of value propagation
        :type gamma: float
        :param learning_rate: learning rate of the model
        :type learning_rate: int
        :param inv_temp: exploration's inverse temperature (used with softmax)
        :type inv_temp: int
        :param eta: used to update the model's reliability
        :type eta: float
        :param allo: whether the model uses an allocentric or egocentric frame of reference
        :type allo: boolean
        """
        super().__init__(env=env, gamma=0.9, learning_rate=learning_rate, inv_temp=inv_temp)

        self.eta = eta
        self.allo = allo
        self.responses = {} # to temporarily store agent's and environment's variables during each episode (to build log files)
        self.reliability = 0. # used as a parameter by an uncertainty driven coordination model
        self.features = LandmarkCells() # neurons that encode the visual input
        self.weights = np.zeros((self.features.n_cells*2, self.env.nr_actions)) # weights associating each sensory neurons to action neurons
        self.last_observation = np.zeros(self.features.n_cells*2) # last visual perception

    def setup(self): # not used yet
        pass

    def init_saving(self, t, s): # not used yet
        pass

    def save(self, t, s): # not used yet
        pass

    def take_decision(self): # not used yet
        pass

    def update(self, reward, s, ego_a):
        """
        Update the model weights using TD-learning

        :param reward: reward obtained by transitioning to the current state s
        :type reward: float
        :param s: the current state
        :type s: int
        :param ego_a: the last performed action (in the egocentric frame)
        :type ego_a: int

        :returns: The RPE, which is used to later compute the model's reliability
        """
        visual_rep = self.get_feature_rep() # landmark neurons input
        Q = self.compute_Q(self.last_observation)
        RPE = self.compute_error(Q, ego_a, visual_rep, s, reward)
        self.update_weights(RPE, ego_a, self.last_observation)
        self.last_observation = visual_rep
        return RPE

    def update_reliability(self, RPE):
        """
        Update the model's reliability (used by superordinate, uncertainty driven coordination model)

        :param RPE: the TD error of the model
        :type RPE: float
        """
        self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)


    def get_feature_rep(self, state=None, orientation=None):
        """
        Compute and returns the landmark neurons' activity, for both the proximal and distal landmarks.
        80 neurons encodes the proximal beacon distance and orientation relative to the agent
        80 other neurons encodes the distal beacon distance and orientation relative to the agent
        The neurons activity might either encode visual features in the egocentric or allocentric frame of reference

        :param state: current state of the agent
        :type state: int
        :param orientation: current orientation of the agent
        :type orientation: int

        :returns: The landmark neurons activity
        :returns type: float array of dim 160
        """
        # the method can be used with either virtual or real (current) agent's state and orientation
        if state is None:
            state = self.env.get_current_state()
        if orientation is None:
            orientation = self.env.agent_orientation

        proximal_rep = self.get_single_feature_rep(state, orientation, self.env.landmark_location) # 80 neurons activity
        distal_rep = self.get_single_feature_rep(state, orientation, self.env.distal_landmark_location) # # 80 neurons activity
        visual_rep = np.concatenate((proximal_rep, distal_rep), axis=None)
        return visual_rep

    def get_single_feature_rep(self, state, orientation, landmark_location):
        """
        Compute and returns the landmark neurons' activity for a single landmark.
        The neurons activity might either encode visual features in the egocentric or allocentric frame of reference

        :param state: current state of the agent
        :type state: int
        :param orientation: current orientation of the agent
        :type orientation: int
        :param landmark_location: location (state) of the landmark of interest (either distal or proximal)
        :type landmark_location: int

        :returns: The landmark neurons activity
        :returns type: float array of dim 80
        """
        # if the model works in an allocentric frame of reference, then the visual
        # signal is not function of the agent's orientation
        if self.allo:
            orientation = 0
        # no signal if no landmark
        if landmark_location is None:
            return np.zeros(80)

        distance = self.get_distance_to_landmark(state, landmark_location)
        angle = self.angle_to_landmark(state, orientation, landmark_location)

        if (int(distance),int(angle)) in self.responses:
            response = self.responses[(int(distance),int(angle))]
        else:
            response = self.features.compute_response(distance, angle)

        # stores observed signal for each angle-distance couple, to dramatically increase computational performances
        self.responses[(int(distance),int(angle))] = response

        return response

    def angle_to_landmark(self, state, orientation, landmark_location):
        """
        Compute the angle between agent and landmark of interest

        :param state: current state of the agent
        :type state: int
        :param orientation: current orientation of the agent
        :type orientation: int
        :param landmark_location: location (state) of the landmark of interest (either distal or proximal)
        :type landmark_location: int

        :returns type: int
        """
        rel_pos = utils.to_agent_frame(landmark_location, self.env.get_state_location(state), np.radians(orientation))
        angle = np.arctan2(rel_pos[1], rel_pos[0])
        return np.degrees(angle)

    def get_distance_to_landmark(self, state, landmark_location):
        distance_to_landmark = np.linalg.norm(
            np.array(landmark_location) - np.array(self.env.get_state_location(state)))
        return distance_to_landmark

    def compute_error(self, Q, a, next_f, next_state, reward):
        """
        Compute the TD error for a given transition

        :param Q: the Q-values at pre-transition state
        :type Q: float array
        :param a: allocentric action chosen by the model at pre-transition state
        :type a: int
        :param next_f: visual landmark neurons input (160 dim array) at post-transition state
        :type next_f: float array
        :param next_state: post-transition state
        :type next_state: int
        :param reward: reward obtained transitioning to next_state
        :type reward: float

        :returns type: float
        """
        next_Q = self.weights.T @ next_f
        if self.env.is_terminal(next_state):
            RPE = reward - Q[a]
        else:
            RPE = reward + self.gamma * np.max(next_Q) - Q[a]
        return RPE

    def compute_Q(self, features):
        """
        Compute the Q-values for a given visual input, in an egocentric frame of reference

        :param features: a 160 dim vector representing the visual input (distal and proximal landmark)
        :type features: float array

        :returns type: float array
        """
        return self.weights.T @ features

    def compute_Q_allo(self, Q_ego):
        """
        Transform Q-values in an egocentric frame of reference to Q-values in an allocentric frame of reference

        :param Q-ego: a 6 dimension array
        :type features: float array

        :returns type: float array
        """
        # no need to transform Q if the model itself work in the allocentric domain, as Q-ego is already allocentric
        if self.allo:
            return Q_ego
        allocentric_idx = [self.get_allo_action(idx, self.env.agent_orientation) for idx in range(self.env.nr_actions)]
        Q_allo = np.empty(len(Q_ego))
        for i in range(len(Q_ego)):
            allo_idx = allocentric_idx[i]
            Q_allo[allo_idx] = Q_ego[i]
        return Q_allo

    # returns the allocentric version of an egocentric action
    def get_allo_action(self, ego_action_idx, orientation):
        allo_angle = (orientation + self.env.ego_angles[ego_action_idx]) % 360
        for i, theta in enumerate(self.env.allo_angles):
            if theta == round(allo_angle):
                return i
        raise ValueError('Angle not in list.')

    # returns the egocentric version of an allocentric action
    def get_ego_action(self, allo_a, orientation):
        # no need to transform allo_a if the model itself work in the allocentric domain, as allo_a is already egocentric
        if self.allo:
            return allo_a
        ego_angle = round(utils.get_relative_angle(np.degrees(self.env.action_directions[allo_a]), orientation))
        if ego_angle == 180:
            ego_angle = -180
        for i, theta in enumerate(self.env.ego_angles):
            if theta == round(ego_angle):
                return i
        raise ValueError('Angle {} not in list.'.format(ego_angle))


# implements all methods that allows to compute the visual features using landmarks' and agent's positions
class LandmarkCells(object):
    def __init__(self):
        # there are 80 visual neurons with their receptive_fields tuned to 8 angles and
        # 10 different distances between agent and landmark
        self.n_angles = 8
        self.angles = np.linspace(-np.pi, np.pi, self.n_angles)
        min_distance = 0.5
        max_distance = 26
        sample_numbers = 10
        self.preferred_distances = np.linspace(min_distance, max_distance, sample_numbers)
        self.field_length = 9.
        self.field_width = np.radians(30)

        self.receptive_fields = []
        self.rf_locations = []
        for r, th in product(self.preferred_distances, self.angles):
            f = multivariate_normal([r, th], [[self.field_length, 0], [0, self.field_width]])
            self.receptive_fields.append(f)
            self.rf_locations.append((r, th))

        self.n_cells = self.n_angles * len(self.preferred_distances)

    def compute_response(self, distance, angle):
        """
        Compute the activity of all visual neurons when the landmark of interest
        is at a given angle and distance from the agent

        :param distance: the distance between the agent and the beacon
        :type distance: float
        :param angle: the angle from the agent to the beacon
        :type angle: int

        :returns: A 80 dim vector representing the activity of the visual neurons at a given timestep
        :returns type: float array
        """
        angle = np.radians(angle)
        return np.array([f.pdf([distance, angle]) * np.sqrt((2*np.pi)**2 * np.linalg.det(f.cov)) for f in self.receptive_fields])

    def plot_receptive_field(self, idx):
        """
        Plot the receptive field of a given neuron. For development purposes

        :param idx: the neuron of interest index
        :type idx: int
        """
        ax = plt.subplot(projection="polar")

        n = 360

        min_distance = 0.5
        max_distance = 26
        sample_numbers = 10
        rad = np.linspace(min_distance, max_distance, sample_numbers)
        a = np.linspace(-np.pi, np.pi, n)
        r, th = np.meshgrid(rad, a)

        pos = np.empty(r.shape + (2,))
        pos[:, :, 0] = r
        pos[:, :, 1] = th

        z = self.receptive_fields[idx].pdf(pos)
        plt.xlim([-np.pi, np.pi])

        plt.pcolormesh(th, r, z)
        ax.set_theta_zero_location('N')
        ax.set_thetagrids(np.linspace(-180, 180, 6, endpoint=False))

        plt.plot(a, r, ls='none', color='k')
        plt.grid(True)

        plt.colorbar()
        return ax
