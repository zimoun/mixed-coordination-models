from abc import ABC, abstractmethod
import utils
import numpy as np

class Agent(ABC):

    def __init__(self, env, gamma, learning_rate, inv_temp, eta):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.inv_temp = inv_temp
        self.eta = eta
        self.results = None

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def take_decision(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def init_saving(self):
        pass

    @abstractmethod
    def save(self):
        pass



    @abstractmethod
    def compute_Q(self):
        pass

    def softmax_selection(self, state_index, Q, nbr_actions, inv_temp):
        try:
            probabilities = utils.softmax(Q, inv_temp)
            action_idx = np.random.choice(list(range(nbr_actions)), p=probabilities)
        except Exception:
            raise Exception()
            action_idx = np.array(Q).argmax()

        return action_idx

class FirstOrderAgent(Agent):

    @abstractmethod
    def update_reliability(self):
        pass

    @abstractmethod
    def compute_error(self):
        pass

# abstract compute error ?

class AssociativeAgent(ABC):

    def update_weights(self, RPE, action, features):
        self.weights[:, action] = self.weights[:, action] + self.learning_rate * RPE * features

    @abstractmethod
    def get_feature_rep(self):
        pass

    @abstractmethod
    def compute_error(self):
        pass

class ModelBasedAgent(ABC):

    def update_R(self, next_state, reward):
        RPE = reward - self.R_hat[next_state]
        self.R_hat[next_state] += 1. * RPE
