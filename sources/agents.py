from itertools import product

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.stats import norm
import gc
from mdp import RTDP
import utils
import random

class CombinedAgent(object):
    def __init__(self, env, init_sr='zero', lesion_dls=False, lesion_hpc=False, gamma=.95, eta=.03,
                 inv_temp=10, learning_rate=.02, inact_hpc=0., inact_dls=0., A_alpha=1., A_beta=.5,
                 alpha1=.01, beta1=.1, max_psr=1, mf_allo=False, HPCmode="SR"):
        self.inv_temp = inv_temp
        self.eta = eta
        #print("eta: ", eta)
        self.learning_rate = learning_rate
        self.A_alpha = A_alpha
        self.A_beta = A_beta
        self.mf_allo = mf_allo
        # self.A_alpha = A_alpha
        # self.A_beta = A_beta

        self.alpha1 = alpha1
        self.beta1 = beta1

        self.HPCmode = HPCmode

        self.lesion_striatum = lesion_dls
        self.lesion_hippocampus = lesion_hpc
        #if self.lesion_hippocampus and self.lesion_striatum:
        #    raise ValueError('cannot lesion both')
        self.env = env

        if self.HPCmode == "SR":
            self.HPC = SRTD(self.env, init_sr=init_sr, gamma=gamma, eta=self.eta)
        elif self.HPCmode == "MB":
            self.HPC = RTDP(self.env,gamma=gamma, eta=self.eta)

        self.DLS = LandmarkLearningAgent(self.env, eta=self.eta)
        self.current_choice = None
        self.weights = np.zeros((self.DLS.features.n_cells, self.env.nr_actions))
        self.gamma = gamma

        if inact_hpc:
            self.max_psr = 1. - inact_hpc
            self.p_sr = self.max_psr
            self.inact_dls = 0.
        elif inact_dls:
            self.max_psr = 1
            self.inact_dls = inact_dls
            self.p_sr = .8
        else:
            self.max_psr = 1
            self.inact_dls = 0.
            self.p_sr = .9

        self.max_psr = max_psr

    def set_exploration(self, inv_temp):
        self.inv_temp = inv_temp

    def get_arrow_map(self):

        res = []
        possible_orientations = np.round(np.degrees(self.env.action_directions))
        print("possible_orien: ",possible_orientations)
        #for state in range(0, 271):



    def one_episode(self, time_limit, random_policy=False, setp_sr=None, random_start_loc=True, center=False, tt=False):

        #print("sr lr: ", self.HPC.learning_rate)
        #print("q lr: ", self.DLS.learning_rate)
        if self.lesion_striatum and self.lesion_hippocampus:
            random_policy = True

        self.env.reset(random_loc=random_start_loc)
        t = 0
        s = self.env.get_current_state()
        cumulative_reward = 0

        if self.mf_allo:
            f = self.DLS.get_feature_rep_allo(s)
        else:
            possible_orientations = np.round(np.degrees(self.env.action_directions))
            angles = []
            for i, o in enumerate(possible_orientations):
                angle = utils.angle_to_landmark(self.env.get_state_location(s), self.env.landmark_location, np.radians(o))
                angles.append(angle)
            orientation = possible_orientations[np.argmin(np.abs(angles))]
            # get MF system features
            f = self.DLS.get_feature_rep(s, orientation)

        Q_mf = self.weights.T @ f


        results = {'time': [t],
                                  # 'reward': [0],
                                  # 'SPE': [0],
                                  # 'RPE': [0],
                                  # 'HPC reliability': [self.HPC.reliability],
                                  # 'DLS reliability': [self.DLS.reliability],
                                  # 'alpha': [self.get_alpha(self.DLS.reliability)],
                                  # 'beta': [self.get_beta(self.DLS.reliability)],
                                  'state': [s],
                                  'P(SR)': [self.p_sr],
                                  # 'choice': [self.current_choice],
                                  # 'M_hat': [self.HPC.M_hat.flatten()],
                                  # 'R_hat': [self.HPC.R_hat.copy()],
                                  # 'Q_mf': [Q_mf],
                                  # 'Q_allo': [None],
                                  # 'Q': [None],
                                  # 'features': [None],
                                  # 'weights': [None],
                                  'platform': [self.env.get_goal_state()],
                                  # 'landmark': [self.env.landmark_location]
                                  }

        while not self.env.is_terminal(s) and t < time_limit:

            if setp_sr is None:
                self.update_p_sr()
            else:
                self.p_sr = setp_sr


            # select action
            if self.mf_allo:
                Q_combined, Q_ego, Q_allo, Q_sr = self.compute_Q(s, self.p_sr)
            else:
                Q_combined, Q_ego, Q_allo, Q_sr = self.compute_Q(s, self.p_sr, orientation)
            #print("bises:: ", str(Q_combined))
            if random_policy:
                allo_a = np.random.choice(list(range(self.env.nr_actions)))
            else:
                allo_a = np.argmax(Q_allo)
                #print("allo_a: ", np.argmax(Q_allo))
                allo_a = self.softmax_selection(s, Q_combined)

            if self.mf_allo:
                ego_a = allo_a
            else:
                ego_a = self.get_ego_action(allo_a, orientation)


            # act
            next_state, reward = self.env.act(allo_a)

            # get MF state representation
            if self.mf_allo:
                next_Q, next_f, next_state, orientation = self.updates(s, next_state, allo_a, reward, Q_mf, f, Q_sr, ego_a)
            else:
                next_Q, next_f, next_state, orientation = self.updates(s, next_state, allo_a, reward, Q_mf, f, Q_sr, ego_a, orientation)

            s = next_state
            f = next_f
            Q_mf = next_Q

            t += 1
            cumulative_reward += reward

            results['time'].append(t)
            # results['reward'].append(reward)
            # results['SPE'].append(SPE)
            # results['RPE'].append(RPE)
            # results['HPC reliability'].append(self.HPC.reliability)
            # results['DLS reliability'].append(self.DLS.reliability)
            # results['alpha'].append(self.get_alpha(self.DLS.reliability))
            # results['beta'].append(self.get_beta(self.DLS.reliability))
            results['state'].append(s)
            results['P(SR)'].append(self.p_sr)
            # results['choice'].append(self.current_choice)
            # results['M_hat'].append(self.HPC.M_hat.copy())
            # results['R_hat'].append(self.HPC.R_hat.copy())
            # results['Q_mf'].append(Q_mf)
            # results['Q_allo'].append(Q_allo)
            # results['Q'].append(Q_combined)
            # results['features'].append(f.copy())
            # results['weights'].append(self.weights.copy())
            results['platform'].append(self.env.get_goal_state())
            # results['landmark'].append(self.env.landmark_location)

        if t == time_limit and not self.env.is_terminal(s):
            self.env.current_state = self.env.get_goal_state()+1
            s = self.env.get_goal_state()+1
            next_state = self.env.get_goal_state()

            ns = 0
            act = 0
            for a in range(0,6):
                ns = self.env.get_next_state_and_reward(s,a)[0]
                if ns == next_state:
                    act = a
                    break

            if self.mf_allo:
                Q_combined, Q_ego, Q_allo, Q_sr = self.compute_Q(s, self.p_sr)
            else:
                Q_combined, Q_ego, Q_allo, Q_sr = self.compute_Q(s, self.p_sr, orientation)

            if self.mf_allo:
                self.updates(s, next_state, act, 1, Q_mf, f, Q_sr, ego_a)
            else:
                ego_a = self.get_ego_action(act, orientation)
                self.updates(s, next_state, act, 1, Q_mf, f, Q_sr, ego_a, orientation)

        return pd.DataFrame.from_dict(results)


    #
    # def one_episode_allo(self, time_limit, random_policy=False, setp_sr=None, random_start_loc=True, center=False, tt=False):
    #
    #     #print("sr lr: ", self.HPC.learning_rate)
    #     #print("q lr: ", self.DLS.learning_rate)
    #     if self.lesion_striatum and self.lesion_hippocampus:
    #         random_policy = True
    #     self.env.reset(random_loc=random_start_loc)
    #     t = 0
    #     s = self.env.get_current_state()
    #     cumulative_reward = 0
    #
    #
    #     # get MF system features
    #     f = self.DLS.get_feature_rep_allo(s)
    #     Q_mf = self.weights.T @ f
    #
    #
    #     results = {'time': [t],
    #                               # 'reward': [0],
    #                               # 'SPE': [0],
    #                               # 'RPE': [0],
    #                               # 'HPC reliability': [self.HPC.reliability],
    #                               # 'DLS reliability': [self.DLS.reliability],
    #                               # 'alpha': [self.get_alpha(self.DLS.reliability)],
    #                               # 'beta': [self.get_beta(self.DLS.reliability)],
    #                               'state': [s],
    #                               'P(SR)': [self.p_sr],
    #                               # 'choice': [self.current_choice],
    #                               # 'M_hat': [self.HPC.M_hat.flatten()],
    #                               # 'R_hat': [self.HPC.R_hat.copy()],
    #                               # 'Q_mf': [Q_mf],
    #                               # 'Q_allo': [None],
    #                               # 'Q': [None],
    #                               # 'features': [None],
    #                               # 'weights': [None],
    #                               'platform': [self.env.get_goal_state()],
    #                               # 'landmark': [self.env.landmark_location]
    #                               }
    #
    #     while not self.env.is_terminal(s) and t < time_limit:
    #
    #         if tt:
    #             print("tt: ", self.env.get_current_state())
    #         if setp_sr is None:
    #             self.update_p_sr()
    #         else:
    #             self.p_sr = setp_sr
    #
    #         # select action
    #         Q_combined, Q_ego, Q_allo, Q_sr = self.compute_Q_allo(s, self.p_sr)
    #         if random_policy:
    #             allo_a = np.random.choice(list(range(self.env.nr_actions)))
    #         else:
    #             allo_a = np.argmax(Q_allo)
    #             #print("allo_a: ", np.argmax(Q_allo))
    #             allo_a = self.softmax_selection(s, Q_combined)
    #         ego_a = allo_a
    #
    #
    #         # act
    #         next_state, reward = self.env.act(allo_a)
    #
    #         # get MF state representation
    #         #orientation = -90
    #         next_f = self.DLS.get_feature_rep_allo(next_state)
    #
    #         # SR updates
    #         if self.HPCmode == "SR":
    #             SPE = self.HPC.compute_error(next_state, s)
    #             delta_M = self.HPC.learning_rate * SPE
    #             self.HPC.M_hat[s, :] += delta_M
    #             self.HPC.update_R(next_state, reward)
    #
    #
    #         elif self.HPCmode == "MB":
    #
    #             self.HPC.update(s, allo_a, next_state)
    #             #self.HPC.R_hat[s, allo_a] = reward
    #             self.HPC.update_R(next_state, reward)
    #             Qmax = self.HPC.Q.max(axis=1)
    #             #self.HPC.Q[s,allo_a] =  self.HPC.R_hat[s,allo_a] + self.gamma*(np.dot(self.HPC.hatP[s,allo_a,:], Qmax))
    #             for ss in range(0,271):
    #                 for a in range(0,6):
    #                     self.HPC.Q[ss,a] =  self.HPC.R_hat[self.env.get_next_state_and_reward(ss,a)[0]] + self.gamma*(np.dot(self.HPC.hatP[ss,a,:], Qmax))
    #             SPE = self.HPC.compute_error(s,next_state, allo_a, reward, Q_sr[allo_a])
    #         # MF updates
    #         next_Q = self.weights.T @ next_f
    #         if self.env.is_terminal(next_state):
    #             RPE = reward - Q_mf[ego_a]
    #         else:
    #             RPE = reward + self.gamma * np.max(next_Q) - Q_mf[ego_a]
    #
    #         self.weights[:, ego_a] = self.weights[:, ego_a] + self.learning_rate * RPE * f
    #
    #         # Reliability updates
    #         if self.env.is_terminal(next_state):
    #             self.DLS.update_reliability(RPE)
    #             self.HPC.update_reliability(SPE, s)
    #
    #         s = next_state
    #         f = next_f
    #         Q_mf = next_Q
    #         t += 1
    #         cumulative_reward += reward
    #
    #         results['time'].append(t)
    #         # results['reward'].append(reward)
    #         # results['SPE'].append(SPE)
    #         # results['RPE'].append(RPE)
    #         # results['HPC reliability'].append(self.HPC.reliability)
    #         # results['DLS reliability'].append(self.DLS.reliability)
    #         # results['alpha'].append(self.get_alpha(self.DLS.reliability))
    #         # results['beta'].append(self.get_beta(self.DLS.reliability))
    #         results['state'].append(s)
    #         results['P(SR)'].append(self.p_sr)
    #         # results['choice'].append(self.current_choice)
    #         # results['M_hat'].append(self.HPC.M_hat.copy())
    #         # results['R_hat'].append(self.HPC.R_hat.copy())
    #         # results['Q_mf'].append(Q_mf)
    #         # results['Q_allo'].append(Q_allo)
    #         # results['Q'].append(Q_combined)
    #         # results['features'].append(f.copy())
    #         # results['weights'].append(self.weights.copy())
    #         results['platform'].append(self.env.get_goal_state())
    #         # results['landmark'].append(self.env.landmark_location)
    #         if self.HPCmode == "SR":
    #
    #             ns = 0
    #             act = 0
    #             for a in range(0,6):
    #                 ns = self.env.get_next_state_and_reward(self.env.get_goal_state()+1,a)[0]
    #                 if ns == self.env.get_goal_state():
    #                     act = a
    #                     break
    #
    #
    #             next_f = self.DLS.get_feature_rep_allo(next_state)
    #
    #             ego_a = allo_a
    #
    #             SPE = self.HPC.compute_error(self.env.get_goal_state(), self.env.get_goal_state()+1)
    #             delta_M = self.HPC.learning_rate * SPE
    #             self.HPC.M_hat[self.env.get_goal_state()+1, :] += delta_M
    #             self.HPC.update_R(self.env.get_goal_state(), reward)
    #
    #             # MF updates
    #             next_Q = self.weights.T @ next_f
    #             if self.env.is_terminal(self.env.get_goal_state()):
    #                 RPE = reward - Q_mf[ego_a]
    #             else:
    #                 RPE = reward + self.gamma * np.max(next_Q) - Q_mf[ego_a]
    #
    #             self.weights[:, ego_a] = self.weights[:, ego_a] + self.learning_rate * RPE * f
    #
    #             # Reliability updates
    #             if self.env.is_terminal(self.env.get_goal_state()):
    #                 self.DLS.update_reliability(RPE)
    #                 self.HPC.update_reliability(SPE, self.env.get_goal_state()+1)
    #
    #         if self.HPCmode == "MB":
    #
    #             ns = 0
    #             act = 0
    #             for a in range(0,6):
    #                 ns = self.env.get_next_state_and_reward(self.env.get_goal_state()+1,a)[0]
    #                 if ns == self.env.get_goal_state():
    #                     act = a
    #                     break
    #
    #
    #             next_f = self.DLS.get_feature_rep_allo(next_state)
    #             ego_a = self.allo_a
    #
    #             self.HPC.update(s, allo_a, next_state)
    #             #self.HPC.R_hat[s, allo_a] = reward
    #             self.HPC.update_R(next_state, reward)
    #             Qmax = self.HPC.Q.max(axis=1)
    #             #self.HPC.Q[s,allo_a] =  self.HPC.R_hat[s,allo_a] + self.gamma*(np.dot(self.HPC.hatP[s,allo_a,:], Qmax))
    #             for ss in range(0,271):
    #                 for a in range(0,6):
    #                     self.HPC.Q[ss,a] =  self.HPC.R_hat[self.env.get_next_state_and_reward(ss,a)[0]] + self.gamma*(np.dot(self.HPC.hatP[ss,a,:], Qmax))
    #             SPE = self.HPC.compute_error(s,next_state, allo_a, reward, Q_sr[allo_a])
    #
    #             # MF updates
    #             next_Q = self.weights.T @ next_f
    #             if self.env.is_terminal(self.env.get_goal_state()):
    #                 RPE = reward - Q_mf[ego_a]
    #             else:
    #                 RPE = reward + self.gamma * np.max(next_Q) - Q_mf[ego_a]
    #
    #             self.weights[:, ego_a] = self.weights[:, ego_a] + self.learning_rate * RPE * f
    #
    #             # Reliability updates
    #             if self.env.is_terminal(self.env.get_goal_state()):
    #                 self.DLS.update_reliability(RPE)
    #                 self.HPC.update_reliability(SPE, self.env.get_goal_state()+1)
    #
    #     return pd.DataFrame.from_dict(results)
    #


    def updates(self, s, next_state, allo_a, reward, Q_mf, f, Q_sr, ego_a, orientation=None):

        if self.mf_allo:
            next_f = self.DLS.get_feature_rep_allo(next_state)
        else:
            orientation = self.DLS.get_orientation(s, next_state, orientation)
            next_f = self.DLS.get_feature_rep(next_state, orientation)

        # SR updates
        if self.HPCmode == "SR":
            SPE = self.HPC.compute_error(next_state, s)
            delta_M = self.HPC.learning_rate * SPE
            self.HPC.M_hat[s, :] += delta_M
            self.HPC.update_R(next_state, reward)

        elif self.HPCmode == "MB":

            self.HPC.update(s, allo_a, next_state)
            #self.HPC.R_hat[s, allo_a] = reward
            self.HPC.update_R(next_state, reward)
            Qmax = self.HPC.Q.max(axis=1)
            #self.HPC.Q[s,allo_a] =  self.HPC.R_hat[s,allo_a] + self.gamma*(np.dot(self.HPC.hatP[s,allo_a,:], Qmax))
            for ss in range(0,271):
                for a in range(0,6):
                    self.HPC.Q[ss,a] =  self.HPC.R_hat[self.env.get_next_state_and_reward(ss,a)[0]] + self.gamma*(np.dot(self.HPC.hatP[ss,a,:], Qmax))
            SPE = self.HPC.compute_error(s,next_state, allo_a, reward, Q_sr[allo_a])

        # MF updates
        next_Q = self.weights.T @ next_f
        if self.env.is_terminal(next_state):
            RPE = reward - Q_mf[ego_a]
        else:
            RPE = reward + self.gamma * np.max(next_Q) - Q_mf[ego_a]

        self.weights[:, ego_a] = self.weights[:, ego_a] + self.learning_rate * RPE * f

        # Reliability updates
        if self.env.is_terminal(next_state):
            self.DLS.update_reliability(RPE)
            self.HPC.update_reliability(SPE, s)

        return next_Q, next_f, next_state, orientation

    def get_ego_action(self, allo_a, orientation):
        ego_angle = round(utils.get_relative_angle(np.degrees(self.env.action_directions[allo_a]), orientation))
        if ego_angle == 180:
            ego_angle = -180
        for i, theta in enumerate(self.env.ego_angles):
            if theta == round(ego_angle):
                return i
        raise ValueError('Angle {} not in list.'.format(ego_angle))

    def update_p_sr(self):
        if self.lesion_hippocampus:
            self.p_sr = 0.
            return
        if self.lesion_striatum:
            self.p_sr = 1.
            return

        alpha = self.get_alpha(self.DLS.reliability)
        beta = self.get_beta(self.HPC.reliability)

        tau = self.max_psr / (alpha + beta)
        fixedpoint = (alpha + self.inact_dls * beta) * tau

        dpdt = (fixedpoint - self.p_sr) / tau

        new_p_sr = self.p_sr + dpdt

        if new_p_sr > self.max_psr:
            new_p_sr = self.max_psr
        if new_p_sr < 0:
            new_p_sr = 0

        if new_p_sr < 0 or new_p_sr > 1:
            raise ValueError('P(SR) is not a probability: {}'.format(new_p_sr))

        self.p_sr = new_p_sr

    def get_alpha(self, chi_mf):
        alpha1 = self.alpha1
        A = self.A_alpha
        #print(A)
        B = np.log((alpha1 ** -1) * A - 1)
        return A / (1 + np.exp(B * chi_mf))

    def get_beta(self, chi_mb):
        beta1 = self.beta1
        A = self.A_beta
        #print(A)
        B = np.log((beta1 ** -1) * A - 1)
        return A / (1 + np.exp(B * chi_mb))

    def compute_Q(self, state_idx, p_sr, orientation=None):

        # compute Q_SR
        if self.HPCmode == "SR":
            V = self.HPC.M_hat @ self.HPC.R_hat
            next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
            Q_sr = [V[s] for s in next_state]
        elif self.HPCmode == "MB":
            #V = np.array([x.max() for x in self.HPC.Q])
            Q_sr = self.HPC.Q[state_idx,:]
            #print(Q_sr)
        #print("lllll: ", str(len(V)))


        # compute Q_MF
        if self.mf_allo:
            features = self.DLS.get_feature_rep_allo(state_idx)
            Q_ego = self.weights.T @ features

            Q_allo = Q_ego

        else:
            features = self.DLS.get_feature_rep(state_idx, orientation)
            Q_ego = self.weights.T @ features
            allocentric_idx = [self.DLS.get_allo_action(idx, orientation) for idx in range(self.env.nr_actions)]

            Q_allo = np.empty(len(Q_ego))
            for i in range(len(Q_ego)):
                allo_idx = allocentric_idx[i]
                Q_allo[allo_idx] = Q_ego[i]

        Q_mf = Q_allo

        Q = p_sr * np.array(Q_sr) + (1-p_sr) * np.array(Q_mf)
        return Q, Q_ego, Q_allo, Q_sr

    #
    # def compute_Q_allo(self, state_idx, p_sr):
    #
    #     # compute Q_SR
    #     #print("bonjour")
    #     # compute Q_SR
    #     if self.HPCmode == "SR":
    #         V = self.HPC.M_hat @ self.HPC.R_hat
    #         next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
    #         Q_sr = [V[s] for s in next_state]
    #     elif self.HPCmode == "MB":
    #         #V = np.array([x.max() for x in self.HPC.Q])
    #         Q_sr = self.HPC.Q[state_idx,:]
    #
    #     # compute Q_MF
    #     features = self.DLS.get_feature_rep_allo(state_idx)
    #     Q_ego = self.weights.T @ features
    #     #allocentric_idx = [self.DLS.get_allo_action(idx, orientation) for idx in range(self.env.nr_actions)]
    #
    #     # Q_allo = np.empty(len(Q_ego))
    #     # for i in range(len(Q_ego)):
    #     #     allo_idx = allocentric_idx[i]
    #     #     Q_allo[allo_idx] = Q_ego[i]
    #
    #     Q_allo = Q_ego
    #
    #     Q_mf = Q_allo
    #
    #     Q = p_sr * np.array(Q_sr) + (1-p_sr) * np.array(Q_mf)
    #
    #     return Q, Q_ego, Q_allo, Q_sr
    #



    def softmax_selection(self, state_index, Q):
        #print("lol"+str(Q))
        probabilities = utils.softmax(Q, self.inv_temp)
        action_idx = np.random.choice(list(range(self.env.nr_actions)), p=probabilities)
        return action_idx


class QLearningAgent(object):
    """Vanilla Q learning agent with tabular state representation.
    """
    max_RPE = 1

    def __init__(self, environment, learning_rate=.1, gamma=.9, epsilon=.1, eta=.03,
                 anneal_epsilon=False, beta=10):
        """

        :param environment:
        :param learning_rate:
        :param gamma:
        :param epsilon:
        """
        self.env = environment
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta
        self.eta = eta

        self.reliability = 0
        self.omega = 0.

        self.Q = np.zeros((self.env.nr_states, self.env.nr_actions))

        self.anneal_epsilon = anneal_epsilon
        if anneal_epsilon:
            self.epsilon = 1

    def one_episode(self, time_limit):
        self.env.reset()

        results = pd.DataFrame({'time': [],
                                'reward': [],
                                'RPE': [],
                                'reliability': [],
                                'omega': []})

        t = 0
        cumulative_reward = 0
        s = self.env.get_current_state()

        while not self.env.is_terminal(s) and t < time_limit:

            a = self.softmax_selection(s)

            next_state, reward = self.env.act(a)

            RPE = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[s][a]

            if self.env.is_terminal(next_state):
                self.omega += .4 * (np.abs(RPE) - self.omega)
                self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)

            self.Q[s][a] = self.Q[s][a] + self.learning_rate * RPE

            cumulative_reward += reward
            s = next_state
            t += 1

            if self.anneal_epsilon and self.epsilon >= .1:
                self.epsilon -= .9 / 50000

            results = results.append({'time': t,
                                      'reward': reward,
                                      'RPE': RPE,
                                      'reliability': self.reliability,
                                      'omega': self.omega}, ignore_index=True)
        return results

    def epsilon_greedy(self, state_idx):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.choice(list(range(self.env.nr_actions)))
        else:
            action_idx = utils.random_argmax(self.Q[state_idx])
        return action_idx

    def softmax_selection(self, state_idx):
        probabilities = utils.softmax(self.Q[state_idx], self.beta)
        action_idx = np.random.choice(list(range(self.env.nr_actions)), p=probabilities)
        return action_idx


class SRTD(object):
    def __init__(self, env, init_sr='identity', beta=20, eta=.03, gamma=.99):
        self.env = env
        self.learning_rate = .1
        self.epsilon = .1
        self.gamma = gamma
        self.beta = beta
        self.eta = eta

        self.reliability = .8
        self.omega = 1.  # np.ones(self.env.nr_states)

        # SR initialisation
        self.M_hat = self.init_M(init_sr)

        self.identity = np.eye(self.env.nr_states)
        self.R_hat = np.zeros(self.env.nr_states)

    def init_M(self, init_sr):
        M_hat = np.zeros((self.env.nr_states, self.env.nr_states))
        if init_sr == 'zero':
            return M_hat
        if init_sr == 'identity':
            M_hat = np.eye(self.env.nr_states)
        elif init_sr == 'rw':  # Random walk initalisation
            random_policy = utils.generate_random_policy(self.env)
            M_hat = self.env.get_successor_representation(random_policy, gamma=self.gamma)
        elif init_sr == 'opt':
            optimal_policy, _ = value_iteration(self.env)
            M_hat = self.env.get_successor_representation(optimal_policy, gamma=self.gamma)
        return M_hat

    def get_SR(self):
        return self.M_hat

    def one_episode(self, time_limit, random_policy=False):
        #print("sr lr: ", self.learning_rate)

        self.env.reset()
        t = 0
        s = self.env.get_current_state()
        cumulative_reward = 0

        results = pd.DataFrame({'time': [],
                                'reward': [],
                                'RPE': [],
                                'reliability': [],
                                'state': []})

        while not self.env.is_terminal(s) and t < time_limit:
            if random_policy:
                a = np.random.choice(list(range(self.env.nr_actions)))
            else:
                a = self.select_action(s)

            next_state, reward = self.env.act(a)

            SPE = self.compute_error(next_state, s)

            self.update_reliability(SPE, s)
            self.M_hat[s, :] += self.update_M(SPE)
            self.update_R(next_state, reward)

            s = next_state
            t += 1
            cumulative_reward += reward

            results = results.append({'time': t, 'reward': reward, 'SPE': SPE, 'reliability': self.reliability,
                                      'state': s}, ignore_index=True)

        return results

    def update_R(self, next_state, reward):
        RPE = reward - self.R_hat[next_state]
        self.R_hat[next_state] += 1. * RPE

    def update_M(self, SPE):

        delta_M = self.learning_rate * SPE
        return delta_M

    def update_reliability(self, SPE, s):
        self.reliability += self.eta * (1 - abs(SPE[s]) / 1 - self.reliability)

    def compute_error(self, next_state, s):
        if self.env.is_terminal(next_state):
            SPE = self.identity[s, :] + self.identity[next_state, :] - self.M_hat[s, :]
        else:
            SPE = self.identity[s, :] + self.gamma * self.M_hat[next_state, :] - self.M_hat[s, :]
        return SPE

    def select_action(self, state_idx, softmax=True):
        # TODO: get categorical dist over next state
        # okay because it's local
        # gradient-based (hill-climbing) gradient ascent
        # graph hill climbing
        # Maybe change for M(sa,sa). potentially over state action only in two step
        V = self.M_hat @ self.R_hat
        next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
        Q = [V[s] for s in next_state]
        probabilities = utils.softmax(Q, self.beta)
        return np.random.choice(list(range(self.env.nr_actions)), p=probabilities)


class LandmarkLearningAgent(object):


    """Q learning agent using landmark features.
    """
    max_RPE = 1

    def __init__(self, environment, learning_rate=.1, gamma=.9, eta=.03, beta=10):
        """

        :param environment:
        :param learning_rate:
        :param gamma:
        """
        self.env = environment
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta
        self.eta = eta

        self.responses={}


        self.reliability = 0

        self.features = LandmarkCells()
        self.weights = np.zeros((self.features.n_cells, self.env.nr_actions))

    def one_episode(self, time_limit):
        self.env.reset()
        #print("q lr: ", self.learning_rate)
        t = 0
        cumulative_reward = 0
        s = self.env.get_current_state()
        orientation = 30  # np.random.choice([30, 90, 150, 210, 270, 330])
        f = self.get_feature_rep(s, orientation)
        Q = self.weights.T @ f

        results = pd.DataFrame({'time': [],
                                'reward': [],
                                'RPE': [],
                                'reliability': [],
                                'state': []})

        while not self.env.is_terminal(s) and t < time_limit:
            a = self.softmax_selection(s, Q)
            allo_a = self.get_allo_action(a, orientation)
            next_state, reward = self.env.act(allo_a)

            orientation = self.get_orientation(s, next_state, orientation)

            next_f = self.get_feature_rep(next_state, orientation)

            RPE, next_Q = self.compute_error(f, a, next_f, next_state, reward)

            if self.env.is_terminal(next_state):
                self.update_reliability(RPE)

            self.update_weights(RPE, a, f)

            cumulative_reward += reward
            s = next_state
            f = next_f
            Q = next_Q
            t += 1

            results = results.append({'time': t, 'reward': reward, 'RPE': RPE, 'reliability': self.reliability,
                                      'state': s}, ignore_index=True)
        return results

    def update_reliability(self, RPE):
        self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)

    def update_weights(self, RPE, a, f):
        self.weights[:, a] = self.weights[:, a] + self.learning_rate * RPE * f

    def compute_error(self, f, a, next_f, next_state, reward):
        Q = self.compute_Q(f)
        next_Q = self.weights.T @ next_f
        if self.env.is_terminal(next_state):
            RPE = reward - Q[a]
        else:
            RPE = reward + self.gamma * np.max(next_Q) - Q[a]
        return RPE, next_Q

    def softmax_selection(self, state_index, Q):
        probabilities = utils.softmax(Q, self.beta)
        action_idx = np.random.choice(list(range(self.env.nr_actions)), p=probabilities)
        return action_idx

    def angle_to_landmark(self, state, orientation):
        rel_pos = utils.to_agent_frame(self.env.landmark_location, self.env.get_state_location(state), np.radians(orientation))
        angle = np.arctan2(rel_pos[1], rel_pos[0])
        return np.degrees(angle)

    def get_feature_rep(self, state, orientation):
        # MODIF
        # response = np.zeros(80)
        # response += 0.1
        distance = self.get_distance_to_landmark(state)
        angle = self.angle_to_landmark(state, orientation)
        #print("dist: ",distance)
        #print("angle: ",angle)
        if (int(distance),int(angle)) in self.responses:
            response = self.responses[(int(distance),int(angle))]
        else:

            response = self.features.compute_response(distance, angle)
            # print("orientation: ", orientation)
            # print("angle: ", angle)
            #response = np.random.rand(80)
            #response[0:10]+=0.5

            #response = response + 0.000000000000000000000001
        #print(response)

        self.responses[(int(distance),int(angle))] = response

        return response


    def get_feature_rep_allo(self, state):
        # MODIF
        # response = np.zeros(80)
        # response += 0.1
        distance = self.get_distance_to_landmark(state)
        angle = self.angle_to_landmark(state, 0)
        #print("dist: ",distance)
        #print("angle: ",angle)
        if (int(distance),int(angle)) in self.responses:
            response = self.responses[(int(distance),int(angle))]
        else:

            response = self.features.compute_response(distance, angle)
        self.responses[(int(distance),int(angle))] = response

        return response

    def get_distance_to_landmark(self, state):
        distance_to_landmark = np.linalg.norm(
            np.array(self.env.landmark_location) - np.array(self.env.get_state_location(state)))
        return distance_to_landmark

    def get_orientation(self, state, next_state, current_orientation):
        if state == next_state:
            return current_orientation
        s1 = self.env.get_state_location(state)
        s2 = self.env.get_state_location(next_state)
        return np.degrees(np.arctan2(s2[1] - s1[1], s2[0] - s1[0]))

    def get_allo_action(self, ego_action_idx, orientation):
        allo_angle = (orientation + self.env.ego_angles[ego_action_idx]) % 360
        for i, theta in enumerate(self.env.allo_angles):
            if theta == round(allo_angle):
                return i
        raise ValueError('Angle not in list.')

    def compute_Q(self, features):
        return self.weights.T @ features


class QLearningTwoStep(LandmarkLearningAgent):
    def __init__(self, env, eta=.03):
        super().__init__(environment=env, eta=eta)
        self.omega = 1

    def get_feature_rep(self, state_idx, orientation):
        return np.eye(self.env.nr_states)[state_idx]

    def update_omega(self, RPE):
        self.omega += self.eta * (np.abs(RPE) - self.omega)


class LandmarkCells(object):
    def __init__(self):
        self.n_angles = 8
        self.angles = np.linspace(-np.pi, np.pi, self.n_angles)
        self.preferred_distances = np.linspace(1, 18, 10)
        self.preferred_distances = np.linspace(0.5, 6, 10)
        self.field_length = 2.
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

    def compute_response2(self, distance, angle):
        for f in self.receptive_fields:
            #print(f.pdf([distance, angle]))
            f.pdf([distance, angle])
            np.sqrt((2*np.pi)**2 * np.linalg.det(f.cov))
            np.linalg.det(f.cov)
            np.sqrt(2*np.pi)

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


if __name__ == '__main__':
    from tqdm import tqdm
    from definitions import ROOT_FOLDER
    import os

    g = HexWaterMaze(6)
    g.set_platform_state(30)

    agent = SRTD(g, init_sr='rw')
    agent_results = []
    agent_ets = []
    for ep in tqdm(range(50)):
        res = agent.one_episode()
        res['trial'] = ep
        res['escape time'] = res.time.max()
        agent_results.append(res)
        agent_ets.append(res.time.max())

    df = pd.concat(agent_results)

    results_dir = os.path.join(ROOT_FOLDER, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df.to_csv(os.path.join(results_dir, 'SRwatermaze.csv'))

    SPEs = np.array(df['SPE'].tolist())
    np.save(os.path.join(results_dir, 'watermazeSPEs.npy'), SPEs)
