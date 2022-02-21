import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from definitions import ROOT_FOLDER, FIGURE_FOLDER
from hippocampus.environments import HexWaterMaze
import matplotlib

en = HexWaterMaze(6)


def classify_strategy(trial_data, previous_platform):
    """Classify strategy as place, response or neither.

    :return:
    """
    threshold = 60
    last_state = trial_data.state.iloc[-1]
    if last_state == previous_platform:
        strategy = 'place'
    elif last_state == trial_data.platform.iloc[0]:
        strategy = 'response'

    if len(trial_data) > threshold:
        strategy = 'neither'
    return strategy


results_folder = os.path.join(ROOT_FOLDER, 'results', 'cue_vs_place_watermaze')
figure_dir = os.path.join(FIGURE_FOLDER, 'cue_vs_place_watermaze')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

all_strategies = []
for group in ['sham', 'hpc', 'dls', 'double']:

    escape_times = []
    res = []
    for agent_nr in range(15):

        df = pd.DataFrame.from_csv(os.path.join(results_folder, '{}_agent{}.csv'.format(group, agent_nr)))

        probe = df[df['trial type'] == 'probe']
        previous_platform = int(df.platform.iloc[0])
        probe['previous platform'] = np.repeat(previous_platform, len(probe))

        strategy = classify_strategy(probe, previous_platform)
        res.append(strategy)
        escape_times.append(len(probe))

    all_strategies.append(res)

#en.plot_occupancy_on_grid(probe)


df = pd.DataFrame(columns=['Sham', 'HPC', 'DLS', 'HPC + DLS'], data=np.array(all_strategies).T)
counts = df.stack().reset_index().pivot_table(index=['level_1', 0], aggfunc='count')

counts = counts.reset_index()
counts.columns = ['Lesion', 'Strategy', 'Count']

reshaped = counts.pivot(index='Lesion', columns='Strategy')
reshaped = reshaped.fillna(0)

reshaped = reshaped.reindex(['Sham', 'HPC', 'DLS', 'HPC + DLS'])

reshaped = reshaped['Count'] / 15 * 100

fig, ax = plt.subplots(figsize=(6,4))

bar = reshaped[['place', 'response', 'neither']].plot.bar(stacked=True, edgecolor='k',
                                        color=['black', 'white', 'white'], ax=ax,
                                                          width=.7)

hatches = [''] * 4 + [''] * 4 + ['.....'] * 4

for i, thisbar in enumerate(bar.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i])

font = {'size': 12, 'weight': 'bold'}

matplotlib.rc('font', **font)


plt.title('Probe test (simulation)')
plt.ylabel('agents (%)')
plt.legend(['Place', 'Cue', 'Neither'], bbox_to_anchor=(1.05, .4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel('Agents (%)')

plt.tight_layout()
plt.xlabel('')

plt.savefig(os.path.join(figure_dir, 'watermaze_lesions_stackedbar.pdf'))
plt.show()

final_df = pd.DataFrame(columns=['Lesion', 'Place', 'Response', 'Neither'])



#locs = np.array([en.get_state_location(s) for s in probe['state']])
#plt.plot(locs[:, 0],  locs[:, 1])


