from agents.agent import Agent, AssociativeAgent
from agents.sr_agent import SRTD
from agents.mb_agent import RTDP
from agents.landmark_learning_agent import LandmarkLearningAgent
import numpy as np
import utils
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize

class DolleAgent(Agent, AssociativeAgent):
    image = {}
    def __init__(self, env, gamma, q_lr, hpc_lr, arbi_inv_temp, inv_temp_mf, inv_temp_gd, eta, init_sr='zero', HPCmode="SR", mf_allo=False, lesion_dls=False,
                lesion_hpc=False, inact_hpc=0., inact_dls=0., lesion_PFC=False):

        if init_sr != 'zero' and init_sr != 'rw' and init_sr != 'identity':
            raise Exception("init_sr should be set to either 'zero', 'rw' or 'identity'")

        super().__init__(env=env, gamma=gamma, learning_rate=eta, inv_temp=arbi_inv_temp, eta=eta)

        # self.alpha1 = 0.01
        # self.beta1 = 0.1
        # self.A_alpha = 3.2 # Steepness of transition curve MF to SR
        # self.A_beta = 1.1

        self.inv_temp_gd = inv_temp_gd
        self.inv_temp_mf = inv_temp_mf
        self.arbi_inv_temp = arbi_inv_temp

        self.HPCmode = HPCmode
        self.mf_allo = mf_allo

        self.lesion_striatum = lesion_dls
        self.lesion_hippocampus = lesion_hpc
        self.lesion_PFC = lesion_PFC

        if self.HPCmode == "SR":
            self.HPC = SRTD(self.env, init_sr=init_sr, gamma=gamma, learning_rate=hpc_lr, inv_temp=inv_temp_gd, eta=eta)
        elif self.HPCmode == "MB":
            self.HPC = RTDP(self.env, gamma=gamma, learning_rate=hpc_lr, inv_temp=inv_temp_gd, eta=eta)
        else:
            raise Exception("HPCmode should either be MB or SR")

        self.DLS = LandmarkLearningAgent(self.env, gamma=gamma, learning_rate=q_lr, inv_temp=inv_temp_mf, eta=eta, allo=mf_allo)
        self.weights = np.zeros((80+80+80, 2))
        self.last_observation = np.zeros(80+80+80)
        self.last_decision_arbi = 0
        # comes from the original implementation, to introduce partial lesions
        # if inact_hpc:
        #     self.max_psr = 1. - inact_hpc
        #     self.p_sr = self.max_psr
        #     self.inact_dls = 0.
        # elif inact_dls:
        #     self.max_psr = 1
        #     self.inact_dls = inact_dls
        #     self.p_sr = .8
        # else:
        #     self.max_psr = 1
        #     self.inact_dls = 0.
        #     self.p_sr = .9


    def setup(self):
        pass

    def init_saving(self, t, s):

        self.results = {'time': [t],
                  'state': [s],
                  'arbitrator_choice': [1],
                  'previous_platform': [self.env.previous_platform_state],
                  'platform': [self.env.get_goal_state()],
                  }

    def save(self, t, s):
        self.results['time'].append(t)
        self.results['state'].append(s)
        self.results['arbitrator_choice'].append(self.last_decision_arbi)
        self.results['previous_platform'].append(self.env.previous_platform_state)
        self.results['platform'].append(self.env.get_goal_state())

    def take_decision(self, s, orientation):

        # print("gamma: ", self.gamma)
        # print("invtemp: ", self.inv_temp)
        # print("lr: ", self.learning_rate)
        # print("eta: ", self.eta)
        # print("DLSgamma: ", self.DLS.gamma)
        # print("DLSinvtemp: ", self.DLS.inv_temp)
        # print("DLSlr: ", self.DLS.learning_rate)
        # print("DLSeta: ", self.DLS.eta)
        # print("HPCgamma: ", self.HPC.gamma)
        # print("HPCinvtemp: ", self.HPC.inv_temp)
        # print("HPClr: ", self.HPC.learning_rate)
        # print("HPCeta: ", self.HPC.eta)
        # print()

        # computing of different Q-values tables
        Q, _, _, _, _, decision_arbi = self.compute_Q(s)

        # selection of preferred action
        if decision_arbi == 0 or self.lesion_PFC == True:
            allo_a = self.softmax_selection(state_index=s, Q=Q, nbr_actions=6, inv_temp=self.DLS.inv_temp)
        else:
            allo_a = self.softmax_selection(state_index=s, Q=Q, nbr_actions=6, inv_temp=self.HPC.inv_temp)


        self.last_decision_arbi = decision_arbi
        ego_a = self.DLS.get_ego_action(allo_a, orientation)
        return allo_a, ego_a


    def update(self, previous_state, reward, s, allo_a, ego_a, orientation):
        decision_arbi = self.last_decision_arbi
        RPE = self.DLS.update(reward, s, ego_a)
        if not self.lesion_hippocampus:
            SPE = self.HPC.update(reward, previous_state, s, allo_a)

        features_arb = self.get_feature_rep(s)

        RPEa, Qa = self.compute_error(self.last_observation, decision_arbi, features_arb, s, reward)
        self.update_weights(RPEa, decision_arbi, self.last_observation)
        self.last_observation = features_arb

        return RPEa


    def get_feature_rep(self, state=None):

        # if orientation is not None:
        #     visual_rep = self.DLS.get_feature_rep(state, orientation)
        # else:
        #     visual_rep = self.DLS.get_feature_rep(state)
        # print(visual_rep)
        visual_rep = self.DLS.get_feature_rep()
        # print(visual_rep)
        features_dist = self.get_feature_dist(state)
        features_rep = np.concatenate((visual_rep, features_dist), axis=None)
        return features_rep


    def get_feature_dist(self, state_idx):
        if self.lesion_hippocampus:
            features_dist = np.zeros(80)

        def compute_response2(distance, angle):
            return np.array([f.pdf([distance, angle])for f in receptive_fields])

        if state_idx in DolleAgent.image:
            blurred = DolleAgent.image[state_idx]
        else:
            empty = np.array([[0.]*60]*60)
            coord = self.env.get_state_location(state_idx)
            # print("X: ", abs(int(coord[1]*10)))
            # print("Y: ", abs(int(coord[0]*10)))
            empty[int(coord[1]*2.5)+30, int(coord[0]*2.5)+30] += 100
            blurred = gaussian_filter(empty, sigma=2)
            blurred = resize(blurred, (8, 10))
            DolleAgent.image[state_idx] = blurred

        flattened = blurred.flatten()
        return flattened

    # def softmax_selection(self, state_index, Q, nbr_actions, inv_temp):
    #     try:
    #         probabilities = utils.softmax(Q, inv_temp)
    #         action_idx = np.random.choice(list(range(nbr_actions)), p=probabilities)
    #     except Exception:
    #         raise Exception()
    #         action_idx = np.array(Q).argmax()
    #
    #     return action_idx

    # def update_weights(self, RPE, action, features):
    #     self.weights[:, action] = self.weights[:, action] + self.learning_rate * RPE * features

    def compute_Q(self, state_idx, orientation=None):

        if orientation is None:
            orientation = 0
        # compute Q_SR
        if self.lesion_hippocampus:
            Q_sr = np.array([0.,0.,0.,0.,0.,0.])
        else:
            Q_sr = self.HPC.compute_Q(state_idx)

        # compute Q_MF
        visual_rep = self.DLS.get_feature_rep(state_idx)
        Q_ego = self.DLS.compute_Q(visual_rep)
        Q_allo = self.DLS.compute_Q_allo(Q_ego)


        features_arb = self.get_feature_rep(state_idx)
        Q_arbi = self.weights.T @ features_arb
        decision_arbitrator = self.softmax_selection(state_index=state_idx, Q=Q_arbi, nbr_actions=2, inv_temp=self.inv_temp)

        if decision_arbitrator == 0 or self.lesion_PFC == True:
            Q = Q_allo
        else:
            Q = Q_sr

        return Q, Q_ego, Q_allo, Q_sr, Q_arbi, decision_arbitrator

    def compute_error(self, f, a, next_f, next_state, reward):
        #Q = self.compute_Q(f)
        Q = self.weights.T @ f
        next_Q = self.weights.T @ next_f
        if self.env.is_terminal(next_state):
            RPE = reward - Q[a]
        else:
            RPE = reward + self.gamma * np.max(next_Q) - Q[a]
        return RPE, next_Q


    # def update_p_sr(self):
    #     if self.lesion_hippocampus:
    #         self.p_sr = 0.
    #         return
    #     if self.lesion_striatum:
    #         self.p_sr = 1.
    #         return
    #
    #     alpha = self.get_alpha(self.DLS.reliability)
    #     beta = self.get_beta(self.HPC.reliability)
    #
    #     tau = self.max_psr / (alpha + beta)
    #     fixedpoint = (alpha + self.inact_dls * beta) * tau
    #
    #     dpdt = (fixedpoint - self.p_sr) / tau
    #
    #     new_p_sr = self.p_sr + dpdt
    #
    #     if new_p_sr > self.max_psr:
    #         new_p_sr = self.max_psr
    #     if new_p_sr < 0:
    #         new_p_sr = 0
    #
    #     if new_p_sr < 0 or new_p_sr > 1:
    #         raise ValueError('P(SR) is not a probability: {}'.format(new_p_sr))
    #
    #     self.p_sr = new_p_sr
    #
    #
    # def get_alpha(self, chi_mf):
    #     alpha1 = self.alpha1
    #     A = self.A_alpha
    #     B = np.log((alpha1 ** -1) * A - 1)
    #     return A / (1 + np.exp(B * chi_mf))
    #
    #
    # def get_beta(self, chi_mb):
    #     beta1 = self.beta1
    #     A = self.A_beta
    #     B = np.log((beta1 ** -1) * A - 1)
    #     return A / (1 + np.exp(B * chi_mb))
