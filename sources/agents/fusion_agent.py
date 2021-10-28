import numpy as np
import pandas as pd
from agents.agent import Agent
from agents.sr_agent import SRTD
from agents.mb_agent import RTDP
from agents.landmark_learning_agent import LandmarkLearningAgent


class CombinedAgent(Agent):
    """
    CombinedAgent is a class which implementents the uncertainty-based, fusion coordination model of Geert 2020.
    It regulate behavioral control between two spatial navigation experts: A model of the Dorsolateral Striatum (DLS)
    and a model of the Hippocampus (HPC). The DLS is modeled in this work and in Geerts original work using an associative learning strategy,
    which is here implemented by the LandmarkLearningAgent class in the landmark_learning_agent module.
    The HPC is modeled in both Geerts and our work by the Successor Representation, which is implemented here by the SRTD class in the fusion_agent module.
    CombinedAgent also support a model-based algorithm as its HPC module. (See RTDP class in mb_agent module)
    The CombinedAgent class is a fusion coordination model. It maintain a push-pull relationship over behavioral control between the HPC and the DLS module.
    The proportion of control given to one of the two spatial navigation experts over behavior is function of their reliability, which
    is computed based on the difference between expected and observed outcome.
    Several functions in the following lines of code were entirely copied or strongly inspired from the original code of Geerts 2020.

    :param env: the environment
    :type env: Environment
    :param gamma: discount factor of value propagation
    :type gamma: float
    :param q_lr: learning rate of the DLS model
    :type q_lr: float
    :param sr_lr: learning rate of the HPC model (SR or MB)
    :type sr_lr: float
    :param inv_temp: softmax exploration's inverse temperature
    :type inv_temp: int
    :param eta: used to update the HPC and DLS models' reliability
    :type eta: float
    :param A_alpha: steepness of transition curve MF to SR
    :type A_alpha: float
    :param A_beta: steepness of transition curve SR to MF
    :type A_beta: float
    :param alpha1: used to compute the transition rate from MF to SR
    :type alpha1: float
    :param beta1: used to compute the transition rate from SR to MF
    :type beta1: float
    :param init_sr: how the SR weights must be initialized (either "zero", "rw", "identity" or "opt")
    :type init_sr: str
    :param HPCmode: to choose the model of the HPC, either "SR" or "MB"
    :type HPCmode: str
    :param mf_allo: whether the DLS module is in the allocentric or egocentric frame of reference
    :type mf_allo: boolean
    :param lesion_dls: Whether the DLS module is inactivated or not (full control of HPC if True)
    :type lesion_dls: boolean
    :param lesion_hpc: Whether the HPC module is inactivated or not (full control of DLS if True)
    :type lesion_hpc: boolean
    """

    def __init__(self, env, gamma, q_lr, hpc_lr, inv_temp, eta, A_alpha, A_beta, alpha1, beta1,
                init_sr='zero', HPCmode="SR", mf_allo=False, lesion_dls=False,
                lesion_hpc=False):

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
            self.HPC = SRTD(self.env, gamma=gamma, learning_rate=hpc_lr, inv_temp=inv_temp, eta=eta, init_sr=init_sr)
        elif self.HPCmode == "MB":
            self.HPC = RTDP(self.env, gamma=gamma, learning_rate=hpc_lr, inv_temp=inv_temp, eta=eta)
        else:
            raise Exception("HPCmode should either be MB or SR")

        self.DLS = LandmarkLearningAgent(self.env, gamma=gamma, learning_rate=q_lr, inv_temp=inv_temp, eta=eta, allo=mf_allo)

        self.max_psr = 1
        self.p_sr = .9 # comprised between 0 and 1

    def setup(self): # is called at the beginning of each episode
        self.update_p_sr()

    def init_saving(self, t, s):
        """
        Is called at the beginning of each episode to declare the agent's and environment's variables to store.
        Stores a first element in each list.
        :param t: the current timestep (should be 0)
        :type t: int
        :param s: the current state of the agent
        :type s: int
        """
        self.results = {'time': [t],
                  'state': [s],
                  'P(SR)': [self.p_sr],
                  'previous_platform': [self.env.previous_platform_state],
                  'platform': [self.env.get_goal_state()],
                  }

    def save(self, t, s):
        """
        Is called at each timestep of an episode to store predefined agent's and environment's variables
        :param t: the current timestep
        :type t: int
        :param s: the current state of the agent
        :type s: int
        """
        self.results['time'].append(t)
        self.results['state'].append(s)
        self.results['P(SR)'].append(self.p_sr)
        self.results['previous_platform'].append(self.env.previous_platform_state)
        self.results['platform'].append(self.env.get_goal_state())

    def take_decision(self, s, orientation):
        """
        Compute the preferred action of the fusion model at a given timestep.
        According to the model's Q-values and explorative behavior.

        :param s: the current state of the agent
        :type s: int
        :param orientation: the current orientation of the agent
        :type orientation: int

        :returns: the preferred action in the allocentric and egocentric frame of reference
        :return type: int and int
        """
        # computing of the fusion model Q-values
        Q_combined, _, _, _ = self.compute_Q(s)

        # selection of the preferred action in the allocentric and egocentric frame
        allo_a = self.softmax_selection(state_index=s, Q=Q_combined, nbr_actions=6, inv_temp=self.inv_temp)
        ego_a = self.DLS.get_ego_action(allo_a, orientation)
        return allo_a, ego_a

    def update(self, previous_state, reward, s, allo_a, ego_a, orientation):
        """
        Triggers the update of the DLS and HPC models. (associative weights, reward function, transition function...)
        Triggers the update of the reliability measurement of the DLS and HPC models

        :param reward: reward obtained by transitioning to the current state s
        :type reward: float
        :param previous_state: the previous state
        :type previous_state: int
        :param s: the current state of the agent
        :type s: int
        :param allo_a: the last performed action (in the allocentric frame)
        :type allo_a: int
        :param ego_a: the last performed action (in the egocentric frame)
        :type ego_a: int
        :param orientation: the current orientation of the agent
        :type orientation: int
        """
        RPE = self.DLS.update(previous_state, reward, s, allo_a, ego_a, orientation)
        if not self.lesion_hippocampus:
            SPE = self.HPC.update(previous_state, reward, s, allo_a, ego_a, orientation)

        # Reliability updates
        if self.env.is_terminal(s):
            self.DLS.update_reliability(RPE)
            if not self.lesion_hippocampus:
                self.HPC.update_reliability(SPE, previous_state)

    def compute_Q(self, state_idx):
        """
        Computes the Q-values of the HPC model, then the Q-values of the DLS model in the egocentric and allocentric
        frame of reference. Then computes the fusion of both Q-tables, weighted by the p_sr parameter, which is
        function of the reliability of both modules (see update_p_sr()). Returns the four created Q-tables.

        :param state_idx: the current state of the agent
        :type state_idx: int

        :returns: four sets of Q-values
        :return type: tuple of arrays of floats
        """

        # compute HPC Q
        if self.lesion_hippocampus:
            Q_sr = np.array([0.,0.,0.,0.,0.,0.])
        else:
            Q_sr = self.HPC.compute_Q(state_idx)

        # compute DLS Q
        visual_rep = self.DLS.get_feature_rep(state_idx)
        Q_ego = self.DLS.compute_Q(visual_rep)
        Q_allo = self.DLS.compute_Q_allo(Q_ego)

        # compute fusionned Q
        Q_combined = self.p_sr * np.array(Q_sr) + (1-self.p_sr) * np.array(Q_allo)

        return Q_combined, Q_ego, Q_allo, Q_sr

    def update_p_sr(self):
        """
        Used to update the proportion of influence of the HPC on behavioral control, in competition with the DLS.
        The control of the SR (or MB) model over behavior is function of the p_sr parameter. This parameter values
        are comprised between 0. and 1., 1. being total control of the HPC over behavior and 0. being total control
        of the DLS over behavior. The p_sr value is function of the DLS and HPC models reliability.
        The following lines of code updates the p_sr parameter using the navigation experts reliability.
        (see Geerts 2020 for corresponding equation)
        """
        if self.lesion_hippocampus: # all behavioral control is given to the DLS
            self.p_sr = 0.
            return
        if self.lesion_striatum: # all behavioral control is given to the HPC
            self.p_sr = 1.
            return

        alpha = self.get_alpha(self.DLS.reliability) # computes transition rates from MF to SR
        beta = self.get_beta(self.HPC.reliability) # computes transition rates from SR to MF

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
        """
        Compute the transition rate from MF to SR, which is function of the reliability of the MF
        (see Geerts 2020 for corresponding equation)

        :param chi_mb: reliability of the MF
        :type chi_mb: float
        :returns: the transition rate from MF to SR
        :return type: float
        """
        alpha1 = self.alpha1
        A = self.A_alpha # steepness of transition curve
        B = np.log((alpha1 ** -1) * A - 1) # transition rate
        return A / (1 + np.exp(B * chi_mf))

    def get_beta(self, chi_mb):
        """
        Compute the transition rate from SR to MF, which is function of the reliability of the SR
        (see Geerts 2020 for corresponding equation)

        :param chi_mb: reliability of the SR
        :type chi_mb: float
        :returns: the transition rate from SR to MF
        :return type: float
        """
        beta1 = self.beta1
        A = self.A_beta # steepness of transition curve
        B = np.log((beta1 ** -1) * A - 1) # transition rate
        return A / (1 + np.exp(B * chi_mb))
