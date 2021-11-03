import numpy as np
import pandas as pd
from agents.sr_agent import SRTD
from agents.mb_agent import RTDP
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from agents.agent import Agent, AssociativeAgent
from agents.landmark_learning_agent import LandmarkLearningAgent


class DolleAgent(Agent, AssociativeAgent):
    """
    DolleAgent is a class which implementents the associative-learning coordination model of Dolle 2010.
    It regulates behavioral control between two spatial navigation experts: A model of the Dorsolateral Striatum (DLS)
    and a model of the Hippocampus (HPC). The DLS is modeled in this work and in Dolle original work using an associative
    learning strategy, which is here implemented by the LandmarkLearningAgent class in the landmark_learning_agent module.
    The HPC is modeled in both Dolle's and our work by a model-based algorithm. In our work, the goal-directed model
    is implemented using the RTDP class in the mb_agent module. DolleAgent also support the Successor-Representation
    as its HPC model (see the SRTD class in the fusion_agent module).
    At each decision time the DolleAgent model gives total control of behavior to a single navigation expert
    (either DLS or HPC) for the next action to be taken. The coordination model learns to select a given navigation strategy
    when encountering certain stimuli. The input of the coordination model is multimodal and consists of visual proximal
    landmark neurons, visual distal landmark neurons (from DLS model) and place-cells (from HPC model). When the HPC is
    lesioned, the coordination model can adopt two different behaviors (selectable by the user using the lesion_PFC parameter).
    Either it always select the DLS strategy, bypassing the associative-learning agent, or in the other case, it still
    need to learn to select between the HPC and DLS strategies over time, with the HPC module proposing only random actions.

    :param env: the environment
    :type env: Environment
    :param gamma: discount factor of value propagation
    :type gamma: float
    :param q_lr: learning rate of the DLS model
    :type q_lr: float
    :param sr_lr: learning rate of the HPC model (SR or MB)
    :type sr_lr: float
    :param arbi_inv_temp: arbitrator's inverse temperature for softmax explorative behavior
    :type arbi_inv_temp: int
    :param inv_temp_mf: DLS model's inverse temperature for softmax explorative behavior
    :type inv_temp_mf: int
    :param inv_temp_gd: HPC model's inverse temperature for softmax explorative behavior
    :type inv_temp_gd: int
    :param learning_rate: learning_rate of the arbitrator
    :type learning_rate: float
    :param init_sr: how the SR weights must be initialized (either "zero", "rw", "identity" or "opt")
    :type init_sr: str
    :param HPCmode: to choose the model of the HPC,either "SR" or "MB"
    :type HPCmode: str
    :param mf_allo: whether the DLS module is in the allocentric or egocentric frame of reference
    :type mf_allo: boolean
    :param lesion_dls: Whether the DLS module is inactivated or not (full control of HPC if True)
    :type lesion_dls: boolean
    :param lesion_hpc: Whether the HPC module is inactivated or not (full control of DLS if True)
    :type lesion_hpc: boolean
    :param lesion_PFC: Arbitrator will always select the DLS strategy if True. If False the arbitrator will still
                       need to select between HPC and DLS strategies when the HPC is lesioned (HPC Q-values all set to 0).
    :type lesion_PFC: boolean
    """

    image = {}

    def __init__(self, env, gamma, q_lr, hpc_lr, arbi_inv_temp, inv_temp_mf,
                inv_temp_gd, learning_rate, init_sr='zero', HPCmode="SR", mf_allo=False,
                lesion_dls=False, lesion_hpc=False, lesion_PFC=False):

        if init_sr != 'zero' and init_sr != 'rw' and init_sr != 'identity':
            raise Exception("init_sr should be set to either 'zero', 'rw' or 'identity'")

        super().__init__(env=env, gamma=gamma, learning_rate=learning_rate, inv_temp=arbi_inv_temp)

        self.inv_temp_gd = inv_temp_gd
        self.inv_temp_mf = inv_temp_mf
        self.arbi_inv_temp = arbi_inv_temp

        self.HPCmode = HPCmode
        self.mf_allo = mf_allo

        self.lesion_striatum = lesion_dls
        self.lesion_hippocampus = lesion_hpc
        self.lesion_PFC = lesion_PFC

        if self.HPCmode == "SR":
            self.HPC = SRTD(self.env, init_sr=init_sr, gamma=gamma, learning_rate=hpc_lr, inv_temp=inv_temp_gd, eta=None)
        elif self.HPCmode == "MB":
            self.HPC = RTDP(self.env, gamma=gamma, learning_rate=hpc_lr, inv_temp=inv_temp_gd, eta=None)
        else:
            raise Exception("HPCmode should either be MB or SR")

        self.DLS = LandmarkLearningAgent(self.env, gamma=gamma, learning_rate=q_lr, inv_temp=inv_temp_mf, eta=None, allo=mf_allo)
        self.weights = np.zeros((240, 2)) # 80 proximal landmark neurons + 80 distal landmark neurons + 80 place-cells
        self.last_observation = np.zeros(240) # 80 proximal landmark neurons + 80 distal landmark neurons + 80 place-cells
        self.last_decision_arbi = 0

    def setup(self): # not used yet
        pass

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
                  'arbitrator_choice': [1],
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
        self.results['arbitrator_choice'].append(self.last_decision_arbi)
        self.results['previous_platform'].append(self.env.previous_platform_state)
        self.results['platform'].append(self.env.get_goal_state())

    def take_decision(self, s, orientation):
        """
        Compute the preferred action of the associative coordination model at a given timestep.
        According to the model's Q-values and explorative behavior.

        :param s: the current state of the agent
        :type s: int
        :param orientation: the current orientation of the agent
        :type orientation: int

        :returns: the preferred action in the allocentric and egocentric frame of reference
        :return type: int and int
        """

        # computing of different Q-values tables
        Q, _, _, _, _, decision_arbi = self.compute_Q(s)

        # selection of preferred action
        if decision_arbi == 0 or self.lesion_PFC == True:
            allo_a = self.softmax_selection(Q=Q, inv_temp=self.DLS.inv_temp)
        else:
            allo_a = self.softmax_selection(Q=Q, inv_temp=self.HPC.inv_temp)

        self.last_decision_arbi = decision_arbi
        ego_a = self.DLS.get_ego_action(allo_a, orientation)
        return allo_a, ego_a


    def update(self, previous_state, reward, s, allo_a, ego_a, orientation):
        """
        Triggers the update of the DLS and HPC models. (associative weights, reward function, transition function...)
        Updates the weights of the associative-learning agent in charge of selecting which navigation expert can drive behavior
        at each decision time.

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
        decision_arbi = self.last_decision_arbi

        self.DLS.update(previous_state, reward, s, allo_a, ego_a, orientation)
        if not self.lesion_hippocampus:
            self.HPC.update(previous_state, reward, s, allo_a, ego_a, orientation)

        features_arb = self.get_feature_rep(s) # place-cells + visual signal

        Qa = self.weights.T @ self.last_observation
        RPEa = self.compute_error(Qa, decision_arbi, features_arb, s, reward)
        self.update_weights(RPEa, decision_arbi, self.last_observation)
        self.last_observation = features_arb

        return RPEa

    def get_feature_rep(self, state=None):
        """
        Compute both the landmark neuron's activity and the place-cells activity. Concatenate
        both signal into one. Returns it.
        80 neurons encodes the proximal beacon distance and orientation relative to the agent
        80 other neurons encodes the distal beacon distance and orientation relative to the agent
        80 neurons encode the place-cells activity (tuned to the position of the agent in the simulated water-maze)
        The first 160 neurons activity might either encode visual features in the egocentric or allocentric frame of reference.
        The last 80 place-cells activity encodes spatial information in the allocentric frame.

        :param state: current state of the agent
        :type state: int

        :returns: The landmark neurons activity concatenated to place-cells activity
        :returns type: float array of dim 240
        """

        visual_rep = self.DLS.get_feature_rep() # visual
        features_dist = self.get_feature_dist(state) # place-cells
        features_rep = np.concatenate((visual_rep, features_dist), axis=None)
        return features_rep


    def get_feature_dist(self, state_idx):
        """
        Compute and returns the place-cells activity.
        80 neurons encodes the position in X and Y axis of the agent, with each neurons tuned to a specific location.
        Neurons response is higher the closer the agent gets to their preferred location.

        :param state_idx: current state of the agent
        :type state_idx: int

        :returns: The place-cells activity
        :returns type: float array of dim 80
        """
        if self.lesion_hippocampus: # no signal if HPC lesion
            features_dist = np.zeros(80)

        if state_idx in DolleAgent.image: # if the signal generated by place-cells at state_idx location has already been experienced and saved
            blurred = DolleAgent.image[state_idx]
        else:
            empty = np.array([[0.]*60]*60) # 60*60 image
            coord = self.env.get_state_location(state_idx)
            empty[int(coord[1]*2.5)+30, int(coord[0]*2.5)+30] += 100
            blurred = gaussian_filter(empty, sigma=2) # smoothing the image so that the single coord pixel stretches to neighboring states
            blurred = resize(blurred, (8, 10)) # resize image to fit a 80 neurons signal
            DolleAgent.image[state_idx] = blurred

        flattened = blurred.flatten()
        return flattened

    def compute_Q(self, state_idx):
        """
        Computes the Q-values of the HPC model, then the Q-values of the DLS model in the egocentric and allocentric
        frame of reference. Then computes the Q-values of the arbitrator and returns the Q-values of the
        preferred navigation expert of the arbitrator.

        :param state_idx: the current state of the agent
        :type state_idx: int

        :returns: the preferred model of the arbitrator and five sets of Q-values
        :return type: int tuple of arrays of floats
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

        # Select navigation expert
        features_arb = self.get_feature_rep(state_idx)
        Q_arbi = self.weights.T @ features_arb
        decision_arbitrator = self.softmax_selection(Q=Q_arbi, inv_temp=self.inv_temp)

        if decision_arbitrator == 0 or self.lesion_PFC == True:
            Q = Q_allo
        else:
            Q = Q_sr

        return Q, Q_ego, Q_allo, Q_sr, Q_arbi, decision_arbitrator

    def compute_error(self, Q, a, next_f, next_state, reward):
        """
        Compute the TD error of the arbitrator for a given transition

        :param Q: the Q-values of the arbitrator at pre-transition state
        :type Q: float array
        :param a: model (1 for HPC, 0 for DLS) chosen by the model at pre-transition state
        :type a: int
        :param next_f: visual landmark neurons + place cells input (240 dim array) at post-transition state
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
