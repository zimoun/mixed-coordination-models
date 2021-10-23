from agents.agent import Agent, ModelBasedAgent, FirstOrderAgent
import numpy as np
import utils

class SRTD(Agent, ModelBasedAgent, FirstOrderAgent):
    def __init__(self, env, gamma, learning_rate, inv_temp, eta, init_sr='identity', beta=20):

        super().__init__(env, gamma, learning_rate, inv_temp, eta)

        self.epsilon = .1
        self.beta = beta
        self.reliability = .8
        self.omega = 1.  # np.ones(self.env.nr_states)

        # SR initialisation
        self.M_hat = self.init_M(init_sr)

        self.identity = np.eye(self.env.nr_states)
        self.R_hat = np.zeros(self.env.nr_states)

    def init_M(self, init_sr):
        M_hat = np.zeros((self.env.nr_states, self.env.nr_states))
        random_policy = utils.generate_random_policy(self.env)
        M_hatu = self.env.get_successor_representation(random_policy, gamma=self.gamma)

        if init_sr == 'zero':
            return M_hat
        if init_sr == 'identity':
            M_hat = np.eye(self.env.nr_states)
        elif init_sr == 'rw':  # Random walk initalisation
            random_policy = utils.generate_random_policy(self.env)
            M_hat = self.env.get_successor_representation(random_policy, gamma=self.gamma)
        elif init_sr == 'opt':
            optimal_policy, _ = value_iteration(self.env)
            M_hat = self.env.get_successor_representation(optimal_policy, gamma=self.gamma)

        return M_hat

    def setup(self):
        pass

    # def one_episode(self, time_limit, random_policy=False):
    #     #print("sr lr: ", self.learning_rate)
    #
    #     self.env.reset()
    #     t = 0
    #     s = self.env.get_current_state()
    #     cumulative_reward = 0
    #
    #     results = pd.DataFrame({'time': [],
    #                             'reward': [],
    #                             'RPE': [],
    #                             'reliability': [],
    #                             'state': []})
    #
    #     while not self.env.is_terminal(s) and t < time_limit:
    #         if random_policy:
    #             a = np.random.choice(list(range(self.env.nr_actions)))
    #         else:
    #             a = self.select_action(s)
    #
    #         next_state, reward = self.env.act(a)
    #
    #         SPE = self.compute_error(next_state, s)
    #
    #         self.update_reliability(SPE, s)
    #         self.M_hat[s, :] += self.update_M(SPE)
    #         self.update_R(next_state, reward)
    #
    #         s = next_state
    #         t += 1
    #         cumulative_reward += reward
    #
    #         results = results.append({'time': t, 'reward': reward, 'SPE': SPE, 'reliability': self.reliability,
    #                                   'state': s}, ignore_index=True)
    #
    #     return results

    def init_saving(self, t, s):
        pass

    def save(self, t, s):
        pass

    def take_decision(self):
        pass

    def update(self, reward, last_state, s, allo_a):
        SPE = self.compute_error(s, last_state)
        delta_M = self.learning_rate * SPE
        self.M_hat[last_state, :] += delta_M
        self.update_R(s, reward)
        return SPE

    # def update_R(self, next_state, reward):
    #     RPE = reward - self.R_hat[next_state]
    #     self.R_hat[next_state] += 1. * RPE

    def update_M(self, SPE):
        delta_M = self.learning_rate * SPE
        return delta_M

    def update_reliability(self, SPE, s):
        self.reliability += self.eta * (1 - abs(SPE[s]) / 1 - self.reliability)

    def compute_error(self, next_state, s):
        if self.env.is_terminal(next_state):
            SPE = self.identity[s, :] + self.identity[next_state, :] - self.M_hat[s, :]
        else:
            SPE = self.identity[s, :] + self.gamma * self.M_hat[next_state, :] - self.M_hat[s, :]
        return SPE

    # def select_action(self, state_idx, softmax=True):
    #     # TODO: get categorical dist over next state
    #     # okay because it's local
    #     # gradient-based (hill-climbing) gradient ascent
    #     # graph hill climbing
    #     # Maybe change for M(sa,sa). potentially over state action only in two step
    #     V = self.M_hat @ self.R_hat
    #     next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
    #     Q = [V[s] for s in next_state]
    #     probabilities = utils.softmax(Q, self.beta)
    #     return np.random.choice(list(range(self.env.nr_actions)), p=probabilities)

    def compute_Q(self, state_idx):
        V = self.M_hat @ self.R_hat
        next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
        Q_sr = [V[s] for s in next_state]
        return Q_sr

    def get_SR(self):
        return self.M_hat
