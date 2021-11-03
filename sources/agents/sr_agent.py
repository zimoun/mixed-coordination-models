import utils
import numpy as np
from utils import value_iteration
from agents.agent import Agent, ModelBasedAgent, FirstOrderAgent


class SRTD(Agent, ModelBasedAgent, FirstOrderAgent):
    """
    Successor-Representation model based on Dayan 1993 article.
    Used in this work as a model of the hippocampus and goal-directed learning in the coordination model of Geerts 2020.
    The SR represent a middle ground between MF and MB algorithms in the tradeoff between computational complexity and learning flexibility.
    As any MB agent, the SR agent possess a reward function representation, however it doesn't implement
    an estimation of the transition function. It rather updates a matrix M that encodes the future discounted probability
    to visit any state S' when in state S, for all states S of the environment. This makes the agent less flexible to transition
    function changes than MB, however it allows to decrease the complexity of the model's updates, making it dramatically less computationaly
    expensive than MB, but still higher than MF (see Gershman 2018 for a review).
    The input of the model is represented by a single int representing the discrete state of the environment.
    The state of the environment is omnisciently known by the agent, and not inferred from visual,
    vestibular or proprioceptive information like in more biologically plausible models. This represent a limitation of this model.
    Apart from some modifications and deletions, most of the following lines were taken from the original code of Geerts 2020.

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

    def __init__(self, env, gamma, learning_rate, inv_temp, eta, init_sr='identity'):

        super().__init__(env, gamma, learning_rate, inv_temp)

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

    def update(self, previous_state, reward, s, allo_a, ego_a, orientation):
        """
        Triggers the M matrix and reward function updates
        Returns an error signal to allows second-order arbitrator to infer the reliability of the model over time

        :param previous_state: the previous state
        :type previous_state: int
        :param reward: reward obtained by transitioning to the current state s
        :type reward: float
        :param s: the current state of the agent
        :type s: int
        :param allo_a: the last performed action (in the allocentric frame)
        :type allo_a: int
        :param ego_a: the last performed action (in the egocentric frame)
        :type ego_a: int
        :param orientation: the current orientation of the agent
        :type orientation: int

        :returns: The vector-valued error signal (SPE), indicating whether states are visited more or less often than expected
        :return type: float array
        """
        SPE = self.compute_error(previous_state, s)
        delta_M = self.learning_rate * SPE
        self.M_hat[previous_state, :] += delta_M
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

    def compute_error(self, previous_state, s):
        """
        Compute and returns the vector-valued TD error of the SR, for a given transition (see Dayan 1993 for the original equation)

        :param previous_state: pre-transition state
        :type previous_state: int
        :param s: post-transition state
        :type s: int

        :returns: The vector-valued error signal (SPE), indicating whether states are visited more or less often than expected
        :return type: float array
        """
        if self.env.is_terminal(s):
            SPE = self.identity[previous_state, :] + self.identity[s, :] - self.M_hat[previous_state, :]
        else:
            SPE = self.identity[previous_state, :] + self.gamma * self.M_hat[s, :] - self.M_hat[previous_state, :]
        return SPE

    def compute_Q(self, state_idx):
        """
        Compute and returns the Q-values of the agent at state state_idx (see Dayan 1993 for original equations)
        To retrieve the value of a neighbor state of state_idx, the reward function is multiplied to the vector at
        index S (next state) and the resulting vector is summed to obtain the value of the state S.

        :type state_idx: int
        :return type: float array
        """
        V = self.M_hat @ self.R_hat
        next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
        Q_sr = [V[s] for s in next_state]
        return Q_sr

    def get_SR(self):
        return self.M_hat
