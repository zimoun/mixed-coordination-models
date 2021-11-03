import numpy as np
from agents.agent import Agent, ModelBasedAgent, FirstOrderAgent

class RTDP(Agent, ModelBasedAgent, FirstOrderAgent):
    """
    Model-based agent based on the Real-Time Dynamic Programming of (Barto, Bradtke, and Singh 1995).
    Used in this work as a model of the hippocampus and goal-directed learning in the coordination model of Dolle 2010.
    The input of the model is represented by a single int representing the discrete state of the environment.
    The state of the environment is omnisciently known by the agent, and not inferred from visual,
    vestibular or proprioceptive information like in more biologically plausible models. This represent a limitation of this model.
    The MB agent possess a model of the world constituted from a transition and a reward function.
    These are both continuously updated with exloration of the environment. This model of the world is then used to
    compute a value function using replay/planning.

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
    """

    def __init__(self, env, gamma, learning_rate, inv_temp, eta):

        super().__init__(env, gamma, learning_rate, inv_temp)

        self.eta = eta # to compute the model's reliability (see update_reliability())
        self.reliability = .8
        self.max_RPE = 1 # see update_reliability()

        self.Q = np.zeros((self.env.nr_states,self.env.nr_actions)) # the value function
        self.hatP = np.ones((self.env.nr_states, self.env.nr_actions,self.env.nr_states))/self.env.nr_states # the transition function
        self.N = np.ones((self.env.nr_states,self.env.nr_actions)) # counter of every occurence of action a in state s
        self.R_hat = np.zeros(self.env.nr_states) # reward function of the model

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
        Updates the transition and reward functions updates. See (Barto, Bradtke, and Singh 1995) for original equations
        Updates the value function using replay (single full value backup)
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

        :returns: The TD error signal (RPE)
        :return type: float
        """
        Q_values = self.compute_Q(previous_state) # Q-values at previous state
        # update the model of the environment
        for y in range(len(self.hatP[previous_state,allo_a,:])):
            self.hatP[previous_state,allo_a,y] = (1-1/self.N[previous_state,allo_a])*self.hatP[previous_state,allo_a,y] + (1/self.N[previous_state,allo_a]*(s==y))
        # keeping track of the number of time allo_a was performed in previous_state
        self.N[previous_state,allo_a] = self.N[previous_state,allo_a] + 1

        self.update_R(s, reward) # reward function udate
        Qmax = self.Q.max(axis=1)
        # execution of a single full backup (replay/planning)
        for rs in range(0,271): # ss = replay state
            for ra in range(0,6): # ra = replay action
                # updating of the value function using the model of the environment (reward+transition function)
                self.Q[rs,ra] =  self.R_hat[self.env.get_next_state_and_reward(rs,ra)[0]] + self.gamma*(np.dot(self.hatP[rs,ra,:], Qmax))

        # for uncertainty based arbitrator
        RPE = self.compute_error(previous_state, s, reward, Q_values[allo_a])
        return RPE

    def update_reliability(self, RPE, s):
        """
        Update the model's reliability (used by superordinate, uncertainty driven coordination model)

        :param RPE: the TD error of the model
        :type RPE: float
        :param s: current state of the agent
        :type s: int
        """
        self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)

    def compute_error(self, previous_state, s, reward, Q_value):
        """
        Compute the TD error for a given transition

        :param previous_state: pre-transition state
        :type previous_state: int
        :param s: post-transition state
        :type s: int
        :param reward: reward obtained transitioning to next_state
        :type reward: float
        :param Q_value: Q-value of the last action chosen
        :type Q_value: float

        :returns type: float
        """
        if self.env.is_terminal(s):
            RPE = reward + self.gamma * np.max(self.Q[previous_state,:]) - Q_value
        else:
            RPE = reward + self.gamma * np.max(self.Q[previous_state,:]) - Q_value
        return RPE

    def compute_Q(self, state_idx):
        """
        Compute and returns the Q-values of the agent at state state_idx
        :type state_idx: int
        :return type: float array
        """
        return self.Q[state_idx,:]
