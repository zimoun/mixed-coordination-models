import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from agents.agent import Agent, ModelBasedAgent

class RTDP(Agent, ModelBasedAgent):

    def __init__(self, env, gamma, learning_rate, inv_temp, eta=.03):

        super().__init__(env, gamma, learning_rate, inv_temp, eta)

        self.reliability = .8
        self.max_RPE = 1

        self.Q = np.zeros((self.env.nr_states,self.env.nr_actions))
        self.hatP = np.ones((self.env.nr_states, self.env.nr_actions,self.env.nr_states))/self.env.nr_states
        self.N = np.ones((self.env.nr_states,self.env.nr_actions))
        self.R_hat = np.zeros(self.env.nr_states)


    # def update_R(self, next_state, reward):
    #     RPE = reward - self.R_hat[next_state]
    #     self.R_hat[next_state] += 1. * RPE


    # def update_reliability(self, RPE):
    #     self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)
    def setup(self):
        pass

    def init_saving(self, t, s):
        pass

    def save(self, t, s):
        pass

    def take_decision(self):
        pass

    def update(self, reward, s, next_state, allo_a):
        Q_sr = self.compute_Q(s)
        for y in range(len(self.hatP[s,allo_a,:])):
            self.hatP[s,allo_a,y] = (1-1/self.N[s,allo_a])*self.hatP[s,allo_a,y] + (1/self.N[s,allo_a]*(next_state==y))
        self.N[s,allo_a] = self.N[s,allo_a]+1


        self.update_R(next_state, reward)
        Qmax = self.Q.max(axis=1)
        #print(self.Q)
        for ss in range(0,271):
            for a in range(0,6):
                self.Q[ss,a] =  self.R_hat[self.env.get_next_state_and_reward(ss,a)[0]] + self.gamma*(np.dot(self.hatP[ss,a,:], Qmax))

        SPE = self.compute_error(s,next_state, allo_a, reward, Q_sr[allo_a])
        return SPE

    def update_reliability(self, RPE, s):
        self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)

    def compute_error(self, x, next_state, u, reward, Q_value):
        if self.env.is_terminal(next_state):
            RPE = reward + self.gamma * np.max(self.Q[x,:]) - Q_value
        else:
            RPE = reward + self.gamma * np.max(self.Q[x,:]) - Q_value
        return RPE

    def compute_Q(self, state_idx):
        return self.Q[state_idx,:]
