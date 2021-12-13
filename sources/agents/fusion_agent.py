import random
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
    :param ag_params: Contains all the parameters to set the different RL modules of the agent (see AgentsParams for details)
    :type ag_params: AgentsParams
    :param init_sr: how the SR weights must be initialized (either "zero", "rw", "identity" or "opt")
    :type init_sr: str

    """

    def __init__(self, env, ag_params, init_sr=None):

        super().__init__(env=env, gamma=ag_params.gamma, learning_rate=None, inv_temp=ag_params.inv_temp)

        self.A_alpha = ag_params.A_alpha
        self.A_beta = ag_params.A_beta
        self.alpha1 = ag_params.alpha1
        self.beta1 = ag_params.beta1

        self.HPCmode = ag_params.HPCmode
        self.mf_allo = ag_params.mf_allo

        self.Qprox = np.zeros(6)
        self.Qdist = np.zeros(6)

        self.learning = True # deactivate any model update if False

        self.lesion_striatum = ag_params.lesion_DLS
        self.lesion_hippocampus = ag_params.lesion_HPC

        if self.HPCmode == "SR":
            self.HPC = SRTD(self.env, gamma=ag_params.gamma, learning_rate=ag_params.hpc_lr,
                            inv_temp=ag_params.inv_temp, eta=ag_params.eta, init_sr=init_sr)
        elif self.HPCmode == "MB":
            self.HPC = RTDP(self.env, gamma=ag_params.gamma, learning_rate=ag_params.hpc_lr,
                            inv_temp=ag_params.inv_temp, eta=ag_params.eta)
        else:
            raise Exception("HPCmode should either be MB or SR")

        self.DLS = LandmarkLearningAgent(self.env, gamma=ag_params.gamma, learning_rate=ag_params.q_lr,
                                        inv_temp=ag_params.inv_temp, eta=ag_params.eta, allo=ag_params.mf_allo)

        self.max_psr = 1
        self.p_sr = .9 # comprised between 0 and 1

    def setup(self): # is called at the beginning of each episode
        if self.learning:
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
                  'rew_func_sum': [self.HPC.R_hat.sum()],

                  #"synaptic_w_mf": [self.weights.mean()],
                  "Q_max": [0.],
                  "Qsr_max": [0.],
                  "Qcombmf": [0.],
                  "Qcombsr": [0.],
                  "Qcombdist": [0.],
                  "Qcombprox": [0.],

                  "Qsr0": [0.],
                  "Qsr1": [0.],
                  "Qsr2": [0.],
                  "Qsr3": [0.],
                  "Qsr4": [0.],
                  "Qsr5": [0.],

                  "Qprox0": [0.],
                  "Qprox1": [0.],
                  "Qprox2": [0.],
                  "Qprox3": [0.],
                  "Qprox4": [0.],
                  "Qprox5": [0.],

                  "Qdist0": [0.],
                  "Qdist1": [0.],
                  "Qdist2": [0.],
                  "Qdist3": [0.],
                  "Qdist4": [0.],
                  "Qdist5": [0.],

                  "Qmf0": [0.],
                  "Qmf1": [0.],
                  "Qmf2": [0.],
                  "Qmf3": [0.],
                  "Qmf4": [0.],
                  "Qmf5": [0.],

                  "Qcomb0": [0.],
                  "Qcomb1": [0.],
                  "Qcomb2": [0.],
                  "Qcomb3": [0.],
                  "Qcomb4": [0.],
                  "Qcomb5": [0.],

                  'syn_prox_mean': [self.DLS.weights[0:80].mean()],
                  'syn_dist_mean': [self.DLS.weights[80:160].mean()],

                  #"f_mean": [f.mean()],
                  #'state': [s],
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



    def save_internal(self, Q_allo, Q_sr, Q_combined):

        #self.results["synaptic_w_mf"].append(self.weights.mean())
        self.results["Q_max"].append(Q_allo.max())
        #self.results["f_mean"].append(f.mean()),
        self.results["Qsr_max"].append(np.array(Q_sr).max())

        self.results["Qsr0"].append(Q_sr[0])
        self.results["Qsr1"].append(Q_sr[1])
        self.results["Qsr2"].append(Q_sr[2])
        self.results["Qsr3"].append(Q_sr[3])
        self.results["Qsr4"].append(Q_sr[4])
        self.results["Qsr5"].append(Q_sr[5])

        self.results["Qmf0"].append(Q_allo[0])
        self.results["Qmf1"].append(Q_allo[1])
        self.results["Qmf2"].append(Q_allo[2])
        self.results["Qmf3"].append(Q_allo[3])
        self.results["Qmf4"].append(Q_allo[4])
        self.results["Qmf5"].append(Q_allo[5])

        self.results["Qprox0"].append(self.Qprox[0])
        self.results["Qprox1"].append(self.Qprox[1])
        self.results["Qprox2"].append(self.Qprox[2])
        self.results["Qprox3"].append(self.Qprox[3])
        self.results["Qprox4"].append(self.Qprox[4])
        self.results["Qprox5"].append(self.Qprox[5])

        self.results["Qdist0"].append(self.Qdist[0])
        self.results["Qdist1"].append(self.Qdist[1])
        self.results["Qdist2"].append(self.Qdist[2])
        self.results["Qdist3"].append(self.Qdist[3])
        self.results["Qdist4"].append(self.Qdist[4])
        self.results["Qdist5"].append(self.Qdist[5])

        self.results["Qcomb0"].append(Q_combined[0])
        self.results["Qcomb1"].append(Q_combined[1])
        self.results["Qcomb2"].append(Q_combined[2])
        self.results["Qcomb3"].append(Q_combined[3])
        self.results["Qcomb4"].append(Q_combined[4])
        self.results["Qcomb5"].append(Q_combined[5])

        self.results['syn_prox_mean'].append(self.DLS.weights[0:80].mean())
        self.results['syn_dist_mean'].append(self.DLS.weights[80:160].mean())

        if Q_allo.argmax() == np.array(Q_combined).argmax():
            self.results["Qcombmf"].append(1)
        else:
            self.results["Qcombmf"].append(0)

        if np.array(Q_sr).argmax() == np.array(Q_combined).argmax():
            self.results["Qcombsr"].append(1)
        else:
            self.results["Qcombsr"].append(0)


        if np.array(self.Qdist).argmax() == np.array(self.Qc).argmax():
            self.results["Qcombdist"].append(1)
        else:
            self.results["Qcombdist"].append(0)

        if np.array(self.Qprox).argmax() == np.array(self.Qc).argmax():
            self.results["Qcombprox"].append(1)
        else:
            self.results["Qcombprox"].append(0)

        self.results['rew_func_sum'].append(self.HPC.R_hat.sum())


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
        Q_combined, Q_ego, Q_allo, Q_sr = self.compute_Q(s)

        self.save_internal(Q_allo, Q_sr, Q_combined)

        # selection of the preferred action in the allocentric and egocentric frame
        allo_a = self.softmax_selection(Q=Q_combined, inv_temp=self.inv_temp)
        ego_a = self.DLS.get_ego_action(allo_a, orientation)

        if self.lesion_striatum and self.lesion_hippocampus:
            allo_a = random.randint(0, 5)
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
        if self.learning:
            RPE = self.DLS.update(previous_state, reward, s, allo_a, ego_a, orientation)
        if not self.lesion_hippocampus:
            SPE = self.HPC.update(previous_state, reward, s, allo_a, ego_a, orientation)
        if self.learning:
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
        self.Qprox = self.DLS.weights[0:80].T @ visual_rep[0:80]
        self.Qdist = self.DLS.weights[80:160].T @ visual_rep[80:160]
        self.Qc = self.DLS.weights[0:160].T @ visual_rep[0:160]
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
        inact_dls = 0. # comes from the previous implementation, always 0. now
        fixedpoint = (alpha + inact_dls * beta) * tau

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

        :param chi_mf: reliability of the MF
        :type chi_mf: float
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
