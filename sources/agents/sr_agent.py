import utils
import numpy as np
from utils import value_iteration
from agents.agent import Agent, ModelBasedAgent, FirstOrderAgent


class SRTD(Agent, ModelBasedAgent, FirstOrderAgent):

    def __init__(self, env, gamma, learning_rate, inv_temp, eta, init_sr='identity'):
        """
        :param env: the environment
        :type env: Environment
        :param gamma: discount factor of value propagation
        :type gamma: float
        :param learning_rate: learning rate of the model
        :type learning_rate: int
        :param inv_temp: softmax exploration's inverse temperature
        :type inv_temp: int
        :param eta: used to update the model's reliability
        :type eta: float
        :param init_sr: how the SR weights must be initialized (either "zero", "rw", "identity" or "opt")
        :type init_sr: str
        """

        Agent().__init__(env, gamma, learning_rate, inv_temp)

        self.reliability = .8
        self.eta = eta # to compute the model's reliability (see update_reliability())
        self.M_hat = self.init_M(init_sr) # SR initialisation
        self.identity = np.eye(self.env.nr_states) # to compute the TD error (see compute_SPE())
        self.R_hat = np.zeros(self.env.nr_states) # reward function

    def init_M(self, init_sr):
        """
        Initialize the Successor-Representation M matrix

        :param init_sr: how the sr weights must be initialized (either "zero", "rw", "identity" or "opt")
        :type init_sr: str

        :returns: the matrix M, representing the future discounted probability to visit any state S' when in state S, for all states S of the environment
        :return type: float array
        """
        M_hat = np.zeros((self.env.nr_states, self.env.nr_states))
        random_policy = utils.generate_random_policy(self.env)
        init_sr="rw"
        if init_sr == 'zero': # all weights are set to zero
            return M_hat
        if init_sr == 'identity': # only self transitions are set to 1, others to 0
            M_hat = np.eye(self.env.nr_states)
        # random walk initialisation (not used in this work, but originally used in Geerts 2020)
        # no initial random walk stage was allowed during Pearce original experiment, so we removed it from the protocol
        # no actual random walk is performed by the simulated agent using 'rw', instead, M values are set so as to encode the true transition function
        # this initialization allows M's transition probabilies to propagate dramatically faster with exploration than if M weights were all set to 0
        elif init_sr == 'rw':
            random_policy = utils.generate_random_policy(self.env)
            M_hat = self.env.get_successor_representation(random_policy, gamma=self.gamma)
        elif init_sr == 'opt': # doesn't work, not used
            optimal_policy, _ = value_iteration(self.env)
            M_hat = self.env.get_successor_representation(optimal_policy, gamma=self.gamma)

        return M_hat

    def setup(self): # not used yet
        pass

    def init_saving(self, t, s): # not used yet
        pass

    def save(self, t, s): # not used yet
        pass

    def take_decision(self): # not used yet
        pass

    def update(self, reward, last_state, s, allo_a):
        """
        Triggers the M matrix and reward function updates

        :param reward: reward obtained by transitioning to the current state s
        :type reward: float
        :param last_state: the previous state
        :type last_state: int
        :param s: the current state
        :type s: int
        :param allo_a: the last performed action (in the allocentric frame)
        :type allo_a: int

        :returns: The vector-valued error signal (SPE), indicating whether states are visited more or less often than expected
        :return type: float array
        """
        SPE = self.compute_error(last_state, s)
        delta_M = self.learning_rate * SPE
        self.M_hat[last_state, :] += delta_M
        self.update_R(s, reward)
        return SPE

    def update_M(self, SPE):
        delta_M = self.learning_rate * SPE
        return delta_M

    def update_reliability(self, SPE, s):
        """
        Update the model's reliability (used by superordinate, uncertainty driven coordination model)

        :param SPE: the TD error of the model
        :type SPE: float array
        :param s: current state of the agent
        :type s: int
        """
        self.reliability += self.eta * (1 - abs(SPE[s]) / 1 - self.reliability)

    def compute_error(self, last_state, s):
        """
        Compute and returns the vector-valued TD error of the SR, for a given transition (see Dayan 1993 for the original equation)

        :param last_state: pre-transition state
        :type last_state: int
        :param s: post-transition state
        :type s: int

        :returns: The vector-valued error signal (SPE), indicating whether states are visited more or less often than expected
        :return type: float array
        """
        if self.env.is_terminal(next_state):
            SPE = self.identity[s, :] + self.identity[next_state, :] - self.M_hat[s, :]
        else:
            SPE = self.identity[s, :] + self.gamma * self.M_hat[next_state, :] - self.M_hat[s, :]
        return SPE

    def compute_Q(self, state_idx):
        """
        Compute and returns the Q-values of the agent at state state_idx
        :type state_idx: int
        :return type: float array
        """
        V = self.M_hat @ self.R_hat
        next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
        Q_sr = [V[s] for s in next_state]
        return Q_sr

    def get_SR(self):
        return self.M_hat
