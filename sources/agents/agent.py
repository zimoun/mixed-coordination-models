import utils
import numpy as np
from abc import ABC, abstractmethod

# contains generic and abstract methods of RL agents
class Agent(ABC):

    def __init__(self, env, gamma, learning_rate, inv_temp):
        self.env = env # the environment
        self.gamma = gamma # discount factor of value propagation
        self.learning_rate = learning_rate
        self.inv_temp = inv_temp # softmax exploration's inverse temperature
        self.results = None # to temporarily store agent's and environment's variables at each episode

    @abstractmethod
    def setup(self): # is called at the beginning of any episode
        pass

    # compute the preferred action of the model at a given timestep,
    # according to the model's Q-values and explorative behavior.
    @abstractmethod
    def take_decision(self):
        pass

    @abstractmethod
    def update(self): # update the model weights
        pass

    @abstractmethod
    def init_saving(self): # is called at the beginning of each episode to declare the agent's and environment's variables to store
        pass

    @abstractmethod
    def save(self): # called at each timestep to temporarily store variables defined in init_saving
        pass

    @abstractmethod
    def compute_Q(self): # returns the Q-values of the model for the input in parameter
        pass

    # used to model exploitative/explorative behavior choice using a softmax function
    def softmax_selection(self, Q, inv_temp):
        try:
            probabilities = utils.softmax(Q, inv_temp)
            action_idx = np.random.choice(list(range(len(Q))), p=probabilities)
        except Exception:
            raise Exception("Exception occured while selecting between exploitative and explorative behavior")

        return action_idx

# for strategies that select actions in the spatial domain
# in opposition with second order strategies (arbitrators/coordination models) which select strategies
class FirstOrderAgent(ABC):

    # reliability is used by uncertainty-based arbitrators as a parameter to the
    # push-pull relationship between first-order strategies
    @abstractmethod
    def update_reliability(self):
        pass

    @abstractmethod
    def compute_error(self): # to compute the RPE (MF/MB) and SPE (Successor-representation)
        pass

# any associative-learning (or model-free/habitual) agent
class AssociativeAgent(ABC):

    def update_weights(self, RPE, action, features): # classical TD learning update
        self.weights[:, action] = self.weights[:, action] + self.learning_rate * RPE * features

    @abstractmethod
    def get_feature_rep(self): # compute and return the input accessible to the associative agent at a given timestep
        pass

    @abstractmethod
    def compute_error(self): # to compute the RPE (MF/MB) or SPE (Successor-Representation)
        pass

# any agent with a reward function (SR/MB)
class ModelBasedAgent(ABC):

    def update_R(self, next_state, reward): # update the reward function
        RPE = reward - self.R_hat[next_state]
        self.R_hat[next_state] += 1. * RPE


class AgentsParams():
    """
    :ivar mf_allo: whether the DLS module is in the allocentric or egocentric frame of reference
    :vartype mf_allo: boolean
    :ivar hpc_lr: learning rate of the HPC model (SR or MB)
    :vartype hpc_lr: float
    :ivar q_lr: learning rate of the DLS model
    :vartype q_lr: float
    :ivar gamma: discount factor of value propagation (shared between coordination, SR, MB and MF models)
    :vartype gamma: float
    :ivar eta: used to update the HPC and DLS models' reliability (Geerts coordination model only)
    :vartype eta: float
    :ivar arbi_learning_rate: learning rate of the arbitrator associative learning agent (Dolle coordination model only)
    :vartype arbi_learning_rate: float
    :ivar alpha1: used to compute the transition rate from MF to SR (Geerts coordination model only)
    :vartype alpha1: float
    :ivar beta1: used to compute the transition rate from SR to MF (Geerts coordination model only)
    :vartype beta1: float
    :ivar A_alpha: steepness of transition curve MF to SR (Geerts coordination model only)
    :vartype A_alpha: float
    :ivar A_beta: steepness of transition curve SR to MF (Geerts coordination model only)
    :vartype A_beta: float
    :ivar HPCmode: to choose the model of the HPC, either "SR" or "MB"
    :vartype HPCmode: str
    :ivar lesion_HPC: Whether the DLS module is inactivated or not (full control of HPC if True)
    :vartype lesion_HPC: boolean
    :ivar lesion_DLS: Whether the HPC module is inactivated or not (full control of DLS if True)
    :vartype lesion_DLS: boolean
    :ivar dolle: whether the coordination model is based on associative learning (Dolle) or fusion (Geerts)
    :vartype dolle: boolean
    :ivar inv_temp: DLS inverse temperature for softmax exploration (Geerts model only)
    :vartype inv_temp: int
    :ivar inv_temp_gd: HPC inverse temperature for softmax exploration (Dolle model only)
    :vartype inv_temp_gd: int
    :ivar inv_temp_mf: DLS inverse temperature for softmax exploration (Dolle model only)
    :vartype inv_temp_mf: int
    :ivar arbi_inv_temp: Coordination model inverse temperature for softmax exploration (Dolle model only)
    :vartype arbi_inv_temp: int
    :ivar lesion_PFC: (only for Dolle model) Dolle arbitrator will always select the DLS strategy if True.
                        If False the arbitrator will still need to select between HPC and DLS strategies when the HPC
                        is lesioned (but with HPC Q-values all set to 0).
    :vartype lesion_PFC: boolean
    """

    def __init__(self):

        self.mf_allo = None
        self.hpc_lr = None
        self.q_lr = None
        self.inv_temp = None
        self.gamma = None
        self.arbi_learning_rate = None
        self.eta = None # reliability learning rate
        self.alpha1 = None
        self.beta1 = None
        self.A_alpha = None # Steepness of transition curve MF to SR
        self.A_beta = None # Steepness of transition curve SR to MF
        self.HPCmode = None
        self.lesion_HPC = None
        self.lesion_DLS = None
        self.dolle = None
        self.arbi_inv_temp = None
        self.inv_temp_mf = None
        self.inv_temp_gd = None
        self.lesion_PFC = None
