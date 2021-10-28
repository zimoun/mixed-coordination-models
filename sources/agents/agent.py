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
