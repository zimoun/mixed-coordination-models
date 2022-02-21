"""
What needs to happen here:
- simulation of devaluation on plus maze


"""
from hippocampus.environments import DevaluationPlusMaze, Environment
from hippocampus.agents import CombinedAgent, LandmarkLearningAgent
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from hippocampus import utils
from definitions import ROOT_FOLDER
import os


groups = {0: 'control',
          1: 'inactivate_HPC'}


group = groups[1]


if group == 'inactivate_HPC':
    inactivate_HPC = True
else:
    inactivate_HPC = False
if group =='inactivate_DLS':
    inactivate_DLS = True
else:
    inactivate_DLS = False


# save location
results_folder = os.path.join(ROOT_FOLDER, 'results', 'plusmaze_deval', group)
figure_folder = os.path.join(results_folder, 'figures')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    os.makedirs(figure_folder)


class LM(LandmarkLearningAgent):
    def __init__(self, environment=Environment(), learning_rate=.1, gamma=.9, eta=.03, beta=10):
        super().__init__(environment=environment, learning_rate=learning_rate, gamma=gamma, eta=eta, beta=beta)

    def get_feature_rep(self, state, orientation):
        distance = self.get_distance_to_landmark(state)
        angle = self.angle_to_landmark(state, orientation)
        response = self.features.compute_response(distance, angle)
        return np.append(response, state == 6)


class CA(CombinedAgent):
    def __init__(self, env=Environment(), init_sr='rw', lesion_dls=False, lesion_hpc=False, gamma=.95, eta=.03,
                 inv_temp=10, learning_rate=.1, inact_hpc=0., inact_dls=0., A_alpha=1., A_beta=.5,
                 alpha1=.01, beta1=.1):
        super().__init__(env=env, init_sr=init_sr, lesion_dls=lesion_dls, lesion_hpc=lesion_hpc, gamma=gamma, eta=eta,
                 inv_temp=inv_temp, learning_rate=learning_rate, inact_hpc=inact_hpc, inact_dls=inact_dls, A_alpha=A_alpha, A_beta=A_beta,
                 alpha1=alpha1, beta1=beta1)
        self.DLS = LM(self.env, eta=self.eta)
        self.weights = np.zeros((self.DLS.features.n_cells + 1, self.env.nr_actions))

    def one_episode(self, random_policy=False, setp_sr=None, random_start_loc=True):
        if self.lesion_striatum and self.lesion_hippocampus:
            random_policy = True
        time_limit = 1000
        self.env.reset(random_loc=random_start_loc)
        t = 0
        s = self.env.get_current_state()
        cumulative_reward = 0

        possible_orientations = np.round(np.degrees(self.env.action_directions))
        angles = []
        for i, o in enumerate(possible_orientations):
            angle = utils.angle_to_landmark(self.env.get_state_location(s), self.env.landmark_location, np.radians(o))
            angles.append(angle)
        orientation = possible_orientations[np.argmin(np.abs(angles))]

        # get MF system features
        f = self.DLS.get_feature_rep(s, orientation)
        Q_mf = self.weights.T @ f

        results = pd.DataFrame({})

        results = results.append({'time': t,
                                  'reward': 0,
                                  'SPE': 0,
                                  'RPE': 0,
                                  'HPC reliability': self.HPC.reliability,
                                  'DLS reliability': self.DLS.reliability,
                                  'alpha': self.get_alpha(self.DLS.reliability),
                                  'beta': self.get_beta(self.DLS.reliability),
                                  'state': s,
                                  'P(SR)': self.p_sr,
                                  'choice': self.current_choice,
                                  'M_hat': self.HPC.M_hat.flatten(),
                                  'R_hat': self.HPC.R_hat.copy(),
                                  'Q_mf': Q_mf,
                                  'platform': self.env.get_goal_state()}, ignore_index=True)

        while not self.env.is_terminal(s) and t < time_limit:
            if setp_sr is None:
                self.update_p_sr()
            else:
                self.p_sr = setp_sr

            # select action
            Q_combined, Q_allo = self.compute_Q(s, orientation, self.p_sr)
            if random_policy:
                allo_a = np.random.choice(list(range(self.env.nr_actions)))
            else:
                allo_a = self.softmax_selection(s, Q_combined)
            ego_a = self.get_ego_action(allo_a, orientation)

            if s == 6:
                allo_a = 0
                ego_a = self.get_ego_action(allo_a, orientation)

            # act
            next_state, reward = self.env.act(allo_a)

            # get MF state representation
            orientation = self.DLS.get_orientation(s, next_state, orientation)
            next_f = self.DLS.get_feature_rep(next_state, orientation)

            # SR updates
            SPE = self.HPC.compute_error(next_state, s)
            delta_M = self.HPC.learning_rate * SPE
            self.HPC.M_hat[s, :] += delta_M
            self.HPC.update_R(next_state, reward)

            # MF updates
            next_Q = self.weights.T @ next_f
            if self.env.is_terminal(next_state):
                RPE = reward - Q_mf[ego_a]
            else:
                RPE = reward + self.gamma * np.max(next_Q) - Q_mf[ego_a]

            self.weights[:, ego_a] = self.weights[:, ego_a] + self.learning_rate * RPE * f

            # Reliability updates
            if self.env.is_terminal(next_state):
                self.DLS.update_reliability(RPE)
                self.HPC.update_reliability(SPE, s)

            s = next_state
            f = next_f
            Q_mf = next_Q
            t += 1
            cumulative_reward += reward

            results = results.append({'time': t,
                                      'reward': reward,
                                      'SPE': SPE,
                                      'RPE': RPE,
                                      'HPC reliability': self.HPC.reliability,
                                      'DLS reliability': self.DLS.reliability,
                                      'alpha': self.get_alpha(self.DLS.reliability),
                                      'beta': self.get_beta(self.DLS.reliability),
                                      'state': s,
                                      'P(SR)': self.p_sr,
                                      'choice': self.current_choice,
                                      'M_hat': self.HPC.M_hat.copy(),
                                      'R_hat': self.HPC.R_hat.copy(),
                                      'Q_mf': Q_mf,
                                      'Q_allo': Q_allo,
                                      'Q': Q_combined,
                                      'features': f.copy(),
                                      'weights': self.weights.copy(),
                                      'platform': self.env.get_goal_state(),
                                      'landmark': self.env.landmark_location}, ignore_index=True)
        return results


n_agents = 20
n_trials = 30

pm = DevaluationPlusMaze()
behavioural_scores = pd.DataFrame({})

for ag in tqdm(range(n_agents)):
    agent = CA(env=pm, lesion_hpc=inactivate_HPC, lesion_dls=inactivate_DLS, learning_rate=.07, inv_temp=4)  #inv_temp= 5
    agent_results = []

    for trial in tqdm(range(n_trials), leave=False):
        if trial == n_trials - 3 or trial == n_trials -1:
            agent.env.toggle_probe_trial()
        elif trial == n_trials - 2:
            agent.env.toggle_training_trial()
            agent.env.toggle_devaluation()
        else:
            agent.env.toggle_training_trial()

        res = agent.one_episode(random_policy=False)
        res['trial'] = trial
        res['escape time'] = res.time.max()
        res['goal location'] = agent.env.get_goal_state()
        res['total reward'] = res['reward'].sum()
        last_state = res['state'].iloc[-2]
        res['last state'] = last_state
        res['trial type'] = agent.env.trial_type

        if agent.env.trial_type == 'probe':
            if last_state == agent.env.rewarded_terminal:
                res['score'] = 'place'
            elif last_state == 5:
                res['score'] = 'response'
            else:
                raise ValueError('dkfjkdf')

            behavioural_scores = behavioural_scores.append({'agent': ag,
                                                            'trial': trial,
                                                            'score': res['score'].iloc[0],
                                                            'group': None}, ignore_index=True)
        else:
            if last_state == agent.env.rewarded_terminal:
                res['score'] = 'correct'
            elif last_state == 5:
                res['score'] = 'incorrect'
        agent_results.append(res)

    df = pd.concat(agent_results)
    df['agent'] = ag
    df.to_csv(os.path.join(results_folder, 'agent{}.csv'.format(ag)))


behavioural_scores.to_csv(os.path.join(results_folder, 'summary.csv'))

agg = behavioural_scores.pivot_table(index=['trial', 'score'], aggfunc=len, margins=True)

plt.figure()
ax = sns.barplot(x='trial', y='agent', hue='score', data=agg.reset_index())

plt.xticks([0, 1, 2], ['non-deval', 'deval', '_'])
plt.show()
