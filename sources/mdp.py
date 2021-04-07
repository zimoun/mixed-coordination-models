import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

##########-ACTIONS DEFINITION-###################
N = 0
S = 1
E = 2
W = 3
NoOp = 4

class RTDP(object):

    def __init__(self, env, gamma=.95, learning_rate=0.07, eta=.03):

        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eta = eta
        self.reliability = .8
        self.max_RPE = 1


        self.Q = np.zeros((self.env.nr_states,self.env.nr_actions))
        self.hatP = np.ones((self.env.nr_states, self.env.nr_actions,self.env.nr_states))/self.env.nr_states
        self.N = np.ones((self.env.nr_states,self.env.nr_actions))



        #self.r = np.zeros((self.nX, self.nU))
        self.R_hat = np.zeros(self.env.nr_states)
        #self.R_hat = np.zeros((self.env.nr_states, self.env.nr_actions))

    # def update_R(self, next_state, reward):
    #     RPE = reward - self.R_hat[next_state]
    #     self.R_hat[next_state] += 1. * RPE
    def update_R(self, next_state, reward):
        RPE = reward - self.R_hat[next_state]
        self.R_hat[next_state] += 1. * RPE

    def update_reliability(self, RPE,s):
        self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)

    def compute_error(self, x, next_state, u, reward, Q_value):
        if self.env.is_terminal(next_state):
            #RPE = reward - Q_value
            RPE = reward + self.gamma * np.max(self.Q[x,:]) - Q_value

        else:
            RPE = reward + self.gamma * np.max(self.Q[x,:]) - Q_value
        #print(RPE)
        return RPE

    def update(self, x, u, next_state):

        # self.hatP[x,u,:] = self.hatP[x,u,:] * (1-1/self.N[x,u])
        # self.hatP[x,u, next_state] = self.hatP[x,u, next_state] + (1/self.N[x,u]) * 1
        for y in range(len(self.hatP[x,u,:])):

            self.hatP[x,u,y] = (1-1/self.N[x,u])*self.hatP[x,u,y] + (1/self.N[x,u]*(next_state==y))

        self.N[x,u] = self.N[x,u]+1















    def discreteProb(self,p):
        # Draw a random number using probability table p (column vector)
        # Suppose probabilities p=[p(1) ... p(n)] for the values [1:n] are given, sum(p)=1 and the components p(j) are nonnegative. To generate a random sample of size m from this distribution imagine that the interval (0,1) is divided into intervals with the lengths p(1),...,p(n). Generate a uniform number rand, if this number falls in the jth interval give the discrete distribution the value j.
        r = np.random.random()
        cumprob=np.hstack((np.zeros(1),p.cumsum()))
        sample = -1
        for j in range(p.size):
            if (r>cumprob[j]) & (r<=cumprob[j+1]):
                sample = j
                break
        return sample

    # This version works for both first and last part of the TD
    # noise parameter might be None or a float
    def MDPStep(self,x,u, noise = None):
        # This function executes a step on the MDP M given current state x and action u.
        # It returns a next state y and a reward r

        y = self.discreteProb(self.P[x, u]) # y should be sampled according to the discrete distribution self.P[x,u,:]
        if noise is None:
            r = self.r[x,u] # r should be the reward of the transition
        else:
            r = np.random.normal(self.r[x,u], noise)
        return [y,r]


    def softmax(self,Q,x,tau):
        # Returns a soft-max probability distribution over actions
        # Inputs :
        # - Q : a Q-function reprensented as a nX times nU matrix
        # - x : the state for which we want the soft-max distribution
        # - tau : temperature parameter of the soft-max distribution
        # Output :
        # - p : probabilty of each action according to the soft-max distribution
        #(column vector of length nU)

        e_x = np.exp(Q[x] / tau)
        p =  e_x / e_x.sum()

        return p


    def RTDP(self, tau, noise=None):

        for iterr in range(nbIter):


            hatP[x,u,:] = hatP[x,u,:] * (1-1/N[x,u])
            hatP[x,u,y] = hatP[x,u,y] + (1/N[x,u]) * 1
            Qmax = Q.max(axis=1)
            # update of Q-values
            Q[x,u] =  r + self.gamma*(np.dot(hatP[x,u,:], Qmax))

            N[x,u] = N[x,u]+1
