import math
import random
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
    :param ag_params: Contains all the parameters to set the different RL modules of the agent (see AgentsParams for details)
    :type ag_params: AgentsParams
    :param init_sr: how the SR weights must be initialized (either "zero", "rw", "identity" or "opt")
    :type init_sr: str
    """

    pc_pop_activity = {} # associate each state of the maze to place-cells population activity

    def __init__(self, env, ag_params, init_sr=None):

        super().__init__(env=env, gamma=ag_params.gamma, learning_rate=ag_params.arbi_learning_rate, inv_temp=ag_params.arbi_inv_temp)

        self.inv_temp_gd = ag_params.inv_temp_gd
        self.inv_temp_mf = ag_params.inv_temp_mf
        self.arbi_inv_temp = ag_params.arbi_inv_temp

        self.HPCmode = ag_params.HPCmode
        self.mf_allo = ag_params.mf_allo

        self.learning=True

        self.lesion_striatum = ag_params.lesion_DLS
        self.lesion_hippocampus = ag_params.lesion_HPC
        self.lesion_PFC = ag_params.lesion_PFC

        if self.HPCmode == "SR":
            self.HPC = SRTD(self.env, init_sr=init_sr, gamma=ag_params.gamma,
                            learning_rate=ag_params.hpc_lr, inv_temp=ag_params.inv_temp_gd, eta=None)
        elif self.HPCmode == "MB":
            self.HPC = RTDP(self.env, gamma=ag_params.gamma,
                            learning_rate=ag_params.hpc_lr, inv_temp=ag_params.inv_temp_gd, eta=None)
        else:
            raise Exception("HPCmode should either be MB or SR")

        self.DLS = LandmarkLearningAgent(self.env, gamma=ag_params.gamma, learning_rate=ag_params.q_lr,
                                        inv_temp=ag_params.inv_temp_mf, eta=None, allo=ag_params.mf_allo)
        #self.weights = np.zeros((240, 2)) # 80 proximal landmark neurons + 80 distal landmark neurons + 80 place-cells
        self.weights = np.array([[0., 0.0]]*240) # Early preference of the coordination model for the MB. Set to 0. for the no intrinsic cost of MB mode
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
                  'rew_func_sum': [self.HPC.R_hat.sum()],
                  'syn_prox_mean': [self.DLS.weights[0:80].mean()],
                  'syn_dist_mean': [self.DLS.weights[80:160].mean()],
                  'syn_prox_mean_arbi': [self.weights[0:80].mean()],
                  'syn_dist_mean_arbi': [self.weights[80:160].mean()],
                  'syn_pc_arbi': [self.weights[160:240].mean()],

                  }

    def blur_state(self, s):

        dim1 = np.array([*range(-10,12,2)])
        dim2 = np.array([*range(-10,12,2)])
        sdim1 = self.env.grid.cart_coords[s][0]
        sdim2 = self.env.grid.cart_coords[s][1]
        closest1 = dim1[np.abs(dim1 - sdim1).argmin()]
        closest2 = dim2[np.abs(dim2 - sdim2).argmin()]

        def calculateDistance(p1,p2):
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return dist

        lowest = 100
        idlowest = -1
        for i in range(0,271):

            p2 = self.env.grid.cart_coords[i]
            dist = calculateDistance((closest1,closest2),p2)
            if lowest > dist:
                lowest = dist
                idlowest = i

        # neighbors = [self.env.get_next_state_and_reward(s,a)[0] for a in range(6)]
        # res = []
        # for n in neighbors:
        #     res += [self.env.get_next_state_and_reward(n,a)[0] for a in range(6)]
        # res = list(set(res))
        # neighbors.append(s)
        # s2 = random.choice(res)
        return idlowest

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
        self.results['rew_func_sum'].append(self.HPC.R_hat.sum())
        self.results['syn_prox_mean'].append(self.DLS.weights[0:80].mean())
        self.results['syn_dist_mean'].append(self.DLS.weights[80:160].mean())
        self.results['syn_prox_mean_arbi'].append(self.weights[0:80].mean())
        self.results['syn_dist_mean_arbi'].append(self.weights[80:160].mean())
        self.results['syn_pc_arbi'].append(self.weights[160:240].mean())

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
        if self.lesion_hippocampus == True and self.lesion_striatum == True:
            allo_a = random.randint(0, len(Q)-1)
        elif self.lesion_PFC == True and self.lesion_striatum == True:
            allo_a = self.softmax_selection(Q=Q, inv_temp=self.HPC.inv_temp)
            decision_arbi = 1
        elif self.lesion_PFC == True and self.lesion_hippocampus == True:
            allo_a = self.softmax_selection(Q=Q, inv_temp=self.DLS.inv_temp)
            decision_arbi = 0
        elif decision_arbi == 1:
            allo_a = self.softmax_selection(Q=Q, inv_temp=self.HPC.inv_temp)
        elif decision_arbi == 0:
            allo_a = self.softmax_selection(Q=Q, inv_temp=self.DLS.inv_temp)
        else:
            raise Exception("Unknown case")

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
        if self.learning:
            decision_arbi = self.last_decision_arbi
            if not self.lesion_striatum:
                self.DLS.update(previous_state, reward, s, allo_a, ego_a, orientation)
            if not self.lesion_hippocampus:
                s2 = self.blur_state(s)
                s3 = self.blur_state(previous_state)
                #s2 = s
                #s3 = previous_state

                if self.HPCmode == "SR" or decision_arbi == 1:
                    self.HPC.update(s3, reward, s2, allo_a, ego_a, orientation)
                else:
                    # set replay to True for the no intrinsic cost of MB mode
                    self.HPC.update(s3, reward, s2, allo_a, ego_a, orientation, replay=True)

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
            return features_dist

        if state_idx in DolleAgent.pc_pop_activity: # if the signal generated by place-cells at state_idx location has already been experienced and saved
            blurred = DolleAgent.pc_pop_activity[state_idx]
        else:
            empty = np.array([[0.]*60]*60) # 60*60 image
            coord = self.env.get_state_location(state_idx)
            empty[int(coord[1]*2.5)+30, int(coord[0]*2.5)+30] += 100
            blurred = gaussian_filter(empty, sigma=2) # smoothing the image so that the single coord pixel stretches to neighboring states
            blurred = resize(blurred, (8, 10)) # resize image to fit a 80 neurons signal
            DolleAgent.pc_pop_activity[state_idx] = blurred

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
            s2 = self.blur_state(state_idx)
            #s2 = state_idx
            Q_sr = self.HPC.compute_Q(s2)

        # compute DLS Q

        if self.lesion_striatum:
            visual_rep = np.zeros(160)
            Q_ego = np.array([0.,0.,0.,0.,0.,0.])
        else:
            visual_rep = self.DLS.get_feature_rep(state_idx)
            Q_ego = self.DLS.compute_Q(visual_rep)
        Q_allo = self.DLS.compute_Q_allo(Q_ego)

        # Select navigation expert
        features_arb = self.get_feature_rep(state_idx)
        Q_arbi = self.weights.T @ features_arb
        decision_arbitrator = self.softmax_selection(Q=Q_arbi, inv_temp=self.inv_temp)

        if self.lesion_hippocampus == True and self.lesion_striatum == True:
            Q = np.zeros(6)
        elif self.lesion_PFC == True and self.lesion_striatum == True:
            Q = Q_sr
            decision_arbitrator = 1
        elif self.lesion_PFC == True and self.lesion_hippocampus == True:
            Q = Q_allo
            decision_arbitrator = 0
        elif decision_arbitrator == 1:
            Q = Q_sr
        elif decision_arbitrator == 0:
            Q = Q_allo
        else:
            raise Exception("Unknown case")

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

        if a == 1:
            reward = reward - (0.) # set to 0. for the no intrinsic cost of MB mode

        next_Q = self.weights.T @ next_f
        if self.env.is_terminal(next_state):
            RPE = reward - Q[a]
        else:
            RPE = reward + self.gamma * np.max(next_Q) - Q[a]
        return RPE
