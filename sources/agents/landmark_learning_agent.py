from agents.agent import Agent, AssociativeAgent, FirstOrderAgent
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal
import utils


class LandmarkLearningAgent(Agent, AssociativeAgent, FirstOrderAgent):

    """Q learning agent using landmark features.
    """
    max_RPE = 1

    def __init__(self, environment, gamma,learning_rate, inv_temp, eta, allo, beta=10):
        """
        :param environment:
        :param learning_rate:
        :param gamma:
        """
        super().__init__(environment, 0.9, learning_rate, inv_temp, eta)

        self.allo  = allo
        self.beta = beta
        self.responses={}
        self.reliability = 0
        self.features = LandmarkCells()
        self.weights = np.zeros((self.features.n_cells*2, self.env.nr_actions))
        self.last_observation = np.zeros(self.features.n_cells*2)

    def setup(self):
        pass

    def init_saving(self, t, s):
        pass

    def save(self, t, s):
        pass

    def take_decision(self):
        pass

    def update(self, reward, s, ego_a):
        visual_rep = self.get_feature_rep()
        RPE, Q = self.compute_error(self.last_observation, ego_a, visual_rep, s, reward)
        self.update_weights(RPE, ego_a, self.last_observation)
        self.last_observation = visual_rep
        return RPE

    def update_reliability(self, RPE):
        self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)

    def get_feature_rep(self, state=None, orientation=None):
        if state is None:
            state = self.env.get_current_state()

        if orientation is None:
            orientation = self.env.agent_orientation

        proximal_rep = self.get_single_feature_rep(state, orientation, self.env.landmark_location)
        distal_rep = self.get_single_feature_rep(state, orientation, self.env.distal_landmark_location)
        visual_rep = np.concatenate((proximal_rep, distal_rep), axis=None)
        return visual_rep

    def get_single_feature_rep(self, state, orientation, landmark_location):
        if self.allo:
            orientation = 0
        if landmark_location is None:
            return np.zeros(80)

        distance = self.get_distance_to_landmark(state, landmark_location)
        angle = self.angle_to_landmark(state, orientation, landmark_location)

        if (int(distance),int(angle)) in self.responses:
            response = self.responses[(int(distance),int(angle))]
        else:
            response = self.features.compute_response(distance, angle)

        self.responses[(int(distance),int(angle))] = response

        return response

    def angle_to_landmark(self, state, orientation, landmark_location):
        rel_pos = utils.to_agent_frame(landmark_location, self.env.get_state_location(state), np.radians(orientation))
        angle = np.arctan2(rel_pos[1], rel_pos[0])
        return np.degrees(angle)

    def get_distance_to_landmark(self, state, landmark_location):
        distance_to_landmark = np.linalg.norm(
            np.array(landmark_location) - np.array(self.env.get_state_location(state)))
        return distance_to_landmark

    def compute_error(self, f, a, next_f, next_state, reward):
        Q = self.compute_Q(f)
        next_Q = self.weights.T @ next_f
        if self.env.is_terminal(next_state):
            RPE = reward - Q[a]
        else:
            RPE = reward + self.gamma * np.max(next_Q) - Q[a]
        return RPE, next_Q

    def compute_Q(self, features):
        return self.weights.T @ features

    def compute_Q_allo(self, Q_ego):
        if self.allo:
            return Q_ego
        allocentric_idx = [self.get_allo_action(idx, self.env.agent_orientation) for idx in range(self.env.nr_actions)]
        Q_allo = np.empty(len(Q_ego))
        for i in range(len(Q_ego)):
            allo_idx = allocentric_idx[i]
            Q_allo[allo_idx] = Q_ego[i]
        return Q_allo

    def get_allo_action(self, ego_action_idx, orientation):
        allo_angle = (orientation + self.env.ego_angles[ego_action_idx]) % 360
        for i, theta in enumerate(self.env.allo_angles):
            if theta == round(allo_angle):
                return i
        raise ValueError('Angle not in list.')

    def get_ego_action(self, allo_a, orientation):
        if self.allo:
            return allo_a

        ego_angle = round(utils.get_relative_angle(np.degrees(self.env.action_directions[allo_a]), orientation))
        if ego_angle == 180:
            ego_angle = -180
        for i, theta in enumerate(self.env.ego_angles):
            if theta == round(ego_angle):
                return i
        raise ValueError('Angle {} not in list.'.format(ego_angle))




# _____________________________


class LandmarkCells(object):
    def __init__(self):
        self.n_angles = 8
        self.angles = np.linspace(-np.pi, np.pi, self.n_angles)
        self.preferred_distances = np.linspace(1, 18, 10)
        self.preferred_distances = np.linspace(0.5, 26, 10)
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
        angle = np.radians(angle)
        return np.array([f.pdf([distance, angle]) * np.sqrt((2*np.pi)**2 * np.linalg.det(f.cov)) for f in self.receptive_fields])

    def plot_receptive_field(self, idx):
        ax = plt.subplot(projection="polar")

        n = 360
        m = 100

        rad = np.linspace(0, 10, m)
        a = np.linspace(-np.pi, np.pi, n)
        r, th = np.meshgrid(rad, a)

        pos = np.empty(r.shape + (2,))
        pos[:, :, 0] = r
        pos[:, :, 1] = th

        z = self.receptive_fields[idx].pdf(pos)
        # plt.ylim([0, 2*np.pi])
        plt.xlim([-np.pi, np.pi])

        plt.pcolormesh(th, r, z)
        ax.set_theta_zero_location('N')
        ax.set_thetagrids(np.linspace(-180, 180, 6, endpoint=False))

        plt.plot(a, r, ls='none', color='k')
        plt.grid(True)

        plt.colorbar()
        return ax
