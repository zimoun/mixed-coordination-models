import numpy as np
import pandas as pd
from agents.agent import Agent
from agents.sr_agent import SRTD
from agents.mb_agent import RTDP
from agents.landmark_learning_agent import LandmarkLearningAgent


class CombinedAgent(Agent):

    def __init__(self, env, gamma, q_lr, hpc_lr, inv_temp, eta, A_alpha, A_beta, alpha1, beta1,
                init_sr='zero', HPCmode="SR", mf_allo=False, lesion_dls=False,
                lesion_hpc=False, inact_hpc=0., inact_dls=0.):

        if init_sr != 'zero' and init_sr != 'rw' and init_sr != 'identity':
            raise Exception("init_sr should be set to either 'zero', 'rw' or 'identity'")

        Agent().__init__(env=env, gamma=gamma, learning_rate=None, inv_temp=inv_temp)

        self.A_alpha = A_alpha
        self.A_beta = A_beta
        self.alpha1 = alpha1
        self.beta1 = beta1

        self.HPCmode = HPCmode
        self.mf_allo = mf_allo

        self.lesion_striatum = lesion_dls
        self.lesion_hippocampus = lesion_hpc

        if self.HPCmode == "SR":
            self.HPC = SRTD(self.env, init_sr=init_sr, gamma=gamma, learning_rate=hpc_lr, inv_temp=inv_temp, eta=eta)
        elif self.HPCmode == "MB":
            self.HPC = RTDP(self.env, gamma=gamma, learning_rate=hpc_lr, inv_temp=inv_temp, eta=eta)
        else:
            raise Exception("HPCmode should either be MB or SR")

        self.DLS = LandmarkLearningAgent(self.env, gamma=gamma, learning_rate=q_lr, inv_temp=inv_temp, eta=eta, allo=mf_allo)

        # comes from the original implementation, to introduce partial lesions
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

    def setup(self):
        self.update_p_sr()

    def init_saving(self, t, s):

        self.results = {'time': [t],
                  'state': [s],
                  'P(SR)': [self.p_sr],
                  'previous_platform': [self.env.previous_platform_state],
                  'platform': [self.env.get_goal_state()],
                  }

    def save(self, t, s):
        self.results['time'].append(t)
        self.results['state'].append(s)
        self.results['P(SR)'].append(self.p_sr)
        self.results['previous_platform'].append(self.env.previous_platform_state)
        self.results['platform'].append(self.env.get_goal_state())

    def take_decision(self, s, orientation):

        # computing of different Q-values tables
        Q_combined, _, _, _ = self.compute_Q(s)

        # selection of preferred action
        allo_a = self.softmax_selection(state_index=s, Q=Q_combined, nbr_actions=6, inv_temp=self.inv_temp)
        ego_a = self.DLS.get_ego_action(allo_a, orientation)
        return allo_a, ego_a

    def update(self, previous_state, reward, s, allo_a, ego_a, orientation):
        RPE = self.DLS.update(reward, s, ego_a)
        if not self.lesion_hippocampus:

            SPE = self.HPC.update(reward, previous_state, s, allo_a)

        # Reliability updates
        if self.env.is_terminal(s):
            self.DLS.update_reliability(RPE)
            if not self.lesion_hippocampus:
                self.HPC.update_reliability(SPE, previous_state)

    def compute_Q(self, state_idx):
        # if orientation is None:
        #     orientation = 0
        # compute Q_SR
        if self.lesion_hippocampus:
            Q_sr = np.array([0.,0.,0.,0.,0.,0.])
        else:
            Q_sr = self.HPC.compute_Q(state_idx)

        # compute Q_MF
        visual_rep = self.DLS.get_feature_rep(state_idx)
        Q_ego = self.DLS.compute_Q(visual_rep)
        Q_allo = self.DLS.compute_Q_allo(Q_ego)

        Q_combined = self.p_sr * np.array(Q_sr) + (1-self.p_sr) * np.array(Q_allo)

        return Q_combined, Q_ego, Q_allo, Q_sr

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
        B = np.log((alpha1 ** -1) * A - 1)
        return A / (1 + np.exp(B * chi_mf))

    def get_beta(self, chi_mb):
        beta1 = self.beta1
        A = self.A_beta
        B = np.log((beta1 ** -1) * A - 1)
        return A / (1 + np.exp(B * chi_mb))
