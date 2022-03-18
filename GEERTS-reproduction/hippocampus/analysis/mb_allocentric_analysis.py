import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from definitions import RESULTS_FOLDER
from hippocampus.analysis.daw_analysis import add_relevant_columns
from hippocampus.environments import HexWaterMaze


def get_first_trial_info(data):
    d2 = data.pivot_table(index='total trial')
    d2['previous platform'] = d2['platform'].shift(1)
    first_trials = d2[d2['trial']==0]
    first_trials = first_trials.drop(0).pivot_table(index='session')
    return first_trials


def get_surrounding_states(state, env, rec_depth=4):
    surrounding_states = [state]
    for i in range(rec_depth):
        added_states = []
        for s in surrounding_states:
            neighbours = np.flatnonzero(env.adjacency_graph[s])
            for n in neighbours:
                if n not in surrounding_states and n not in added_states:
                    added_states.append(n)
        surrounding_states += added_states
    return surrounding_states


def get_allo_index(agent_data, env):
    """Get the allocentricness index, defined as the amount of time spent around the previous platform location during
    first trials of sessions.

    :param agent_data:
    :param env:
    :return:
    """
    first_trials = get_first_trial_info(agent_data)
    prop_times = []
    for ses in range(1, 11):
        states = np.sort(agent_data[(agent_data.session == ses) & (agent_data.trial == 0)]['state'])
        previous_platform = first_trials['previous platform'][ses]
        surrounding_states = np.sort(np.array(get_surrounding_states(int(previous_platform), env)))

        time_spent = np.isin(states, surrounding_states).sum()
        prop_times.append(time_spent)

    return np.mean(prop_times)


def get_distance_err(agent_data, val_func_data, env):
    """Get the (neg) allocentricness index, defined as distance error between the top of the value function and the
    previous platform location during the first trials of sessions.

    :param agent_data:
    :param env:
    :return:
    """
    distances = []

    #  value_func_tops = val_func_data.max(axis=2).argmax(axis=1)
    platform_locs = agent_data.groupby('session')['platform'].mean().astype('int').to_numpy()

    val_funcs = val_func_data.max(axis=2)

    n = 7
    for ses in range(1, 11):
        top_n_states = val_funcs[ses].argsort()[-n:]

        loc = np.array([0., 0.])
        for s in top_n_states:
            loc += np.array(en.grid.cart_coords[s])
        loc = loc / n

        dist = np.linalg.norm(loc - en.grid.cart_coords[platform_locs[ses-1]])
        distances.append(dist)

    #val_funcs = val_func_data.max(axis=2)

    #for ses in range(10):
    #    env.set_platform_state(platform_locs[ses])
    #    opt_pol, val = value_iteration(env)
    #
    #    dist = np.linalg.norm(val_funcs[ses+1] - val)
    #    distances.append(dist)


    #distances = []
    #for ses in range(1, 11):
    #    distances.append(env.grid.distance(value_func_tops[ses], platform_locs[ses-1]))

    return np.mean(distances) #np.sort(distances)[8:].mean()


def get_model_weights(data):
    add_relevant_columns(data)
    data['Stay'] = data['Stay'].astype('int')
    data = data[['Stay', 'PreviousReward', 'PreviousTransition']]
    mod = smf.logit(formula='Stay ~ PreviousTransition * PreviousReward', data=data)
    res = mod.fit()
    model_based_weight = -res.params['PreviousTransition[T.rare]:PreviousReward']
    model_free_weight = res.params['PreviousReward']
    return model_based_weight, model_free_weight


def compute_scores():
    allocentric_scores = []
    allo_time = []
    for a in tqdm(range(n_agents)):
        df = pd.read_csv(os.path.join(res_dir, 'spatial_agent{}'.format(a)))
        val_funcs = np.load(os.path.join(res_dir, 'spatial_agent{}value_funcs.npy'.format(a)))
        allocentric_scores.append(get_distance_err(df, val_funcs, en))
        allo_time.append(get_allo_index(df, en))

    mb_scores = []
    for a in range(n_agents):
        df = pd.read_csv(os.path.join(res_dir, 'twostep_agent{}'.format(a)))
        mb_weight, mf_weight = get_model_weights(df)
        mb_scores.append(mb_weight)

    allocentric_scores_lesion = []
    allo_time_lesion = []
    for a in tqdm(range(n_agents)):
        df = pd.read_csv(os.path.join(res_dir, 'spatial_partial_lesion_agent{}'.format(a)))
        val_funcs = np.load(os.path.join(res_dir, 'spatial_partial_lesion_agent{}value_funcs.npy'.format(a)))
        allocentric_scores_lesion.append(get_distance_err(df, val_funcs,  en))
        allo_time_lesion.append(get_allo_index(df, en))

    mb_scores_lesion = []
    for a in range(n_agents):
        df = pd.read_csv(os.path.join(res_dir, 'twostep_partial_lesion_agent{}'.format(a)))
        mb_weight, mf_weight = get_model_weights(df)
        mb_scores_lesion.append(mb_weight)

    score_data = pd.DataFrame({})
    score_data['model based'] = np.concatenate([mb_scores, mb_scores_lesion])
    score_data['allocentric'] = np.concatenate([allocentric_scores, allocentric_scores_lesion])
    score_data['allo time'] = np.concatenate([np.log(allo_time), np.log(allo_time_lesion)])
    score_data['group'] = ['control'] * n_agents + ['lesion'] * n_agents

    score_data.to_csv(os.path.join(res_dir, 'score_data.csv'))
    return score_data


def get_correlation_diff_z(r1, r2, n1, n2):
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    denom = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return (z1 - z2) / denom


def compute_correlations(ctrl, lesion):
    # correlations
    r_ctrl, p_ctrl = stats.pearsonr(ctrl['model based'], ctrl['allocentric'])
    r_les, p_les = stats.pearsonr(lesion['model based'], lesion['allocentric'])

    z = get_correlation_diff_z(r_ctrl, r_les, len(ctrl), len(lesion))
    p_diff = stats.norm.sf(abs(z)) * 2
    results = {'z_control': np.arctanh(r_ctrl),
               'p_control': p_ctrl,
               'z_lesion': np.arctanh(r_les),
               'p_lesion': p_les,
               'z_diff': z,
               'p_diff': p_diff}
    return results


def fisher_z(pearson_r):
    z = np.log((1+pearson_r) / (1-pearson_r))
    return z


if __name__ == '__main__':
    import matplotlib
    from scipy import stats

    font = {'family': 'normal',
            'size': 15}

    matplotlib.rc('font', **font)


    n_agents = 20
    colpal = sns.color_palette()

    en = HexWaterMaze(6)

    res_dir = os.path.join(RESULTS_FOLDER, 'mb_spatialmemory')
    params = pd.read_csv(os.path.join(res_dir, 'params.csv'))

    score_data_file = os.path.join(res_dir, 'score_data.csv')

    if not os.path.exists(score_data_file):
        score_data = compute_scores()
    else:
        score_data = pd.read_csv(score_data_file)

    fig, ax = plt.subplots()

    lesion_data = score_data[score_data.group == 'lesion']
    control_data = score_data[score_data.group == 'control']

    # do stats
    r_cont, p_cont = stats.pearsonr(control_data['allocentric'], control_data['model based'])
    r_les, p_les = stats.pearsonr(lesion_data['allocentric'], lesion_data['model based'])
    z_cont = fisher_z(r_cont)
    z_les = fisher_z(r_les)

    z_diff = (z_les - z_cont) / (np.sqrt(
        1/(n_agents-3) + 1/(n_agents-3)
    ))
    p_diff = (1 - stats.norm.cdf(np.abs(z_diff))) * 2

    stats_df = pd.DataFrame({'control': [z_cont, p_cont], 'lesion': [z_les, p_les], 'difference': [z_diff, p_diff]},
                            index=['z', 'p']).T
    stats_df.to_csv(os.path.join(res_dir, 'stats.csv'))


    plt.xlim([0, 6])
    plt.ylim([-1, 4])
    sns.regplot(control_data['allocentric'], control_data['model based'], color=colpal[1], ci=80)
    sns.regplot(lesion_data['allocentric'], lesion_data['model based'], color=colpal[4], ci=80)

    plt.ylabel('Model-based estimate')
    plt.xlabel('Boundary distance error (a.u.)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'figures', 'mb_spatial_cov.pdf'))
    plt.show()
    print('done')

