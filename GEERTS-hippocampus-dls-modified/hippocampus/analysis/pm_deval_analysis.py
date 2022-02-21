import pandas as pd
import os.path as op
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from definitions import RESULTS_FOLDER, FIGURE_FOLDER, ROOT_FOLDER


figure_location = op.join(FIGURE_FOLDER, 'pm_deval')
if not op.exists(figure_location):
    os.makedirs(figure_location)


def load_deval_model_result():
    results_folder = op.join(RESULTS_FOLDER, 'plusmaze_deval')
    control_results = pd.read_csv(op.join(results_folder, 'control', 'summary.csv'))
    control_results['group'] = 'control'
    hpc_lesion_results = pd.read_csv(op.join(results_folder, 'inactivate_HPC', 'summary.csv'))
    hpc_lesion_results['group'] = 'HPC'

    df = pd.concat([control_results, hpc_lesion_results])
    df = df.pivot_table(index=['trial', 'group', 'score'], aggfunc=len, margins=True)
    df['total'] = df['agent'].groupby(['trial', 'group']).sum()
    df['Percentage'] = df['agent'] / df['total'] * 100
    df = df.drop('All')
    df = df.reset_index()
    return df[df['score'] == 'place']


def performance_barchart(df):
    ax = sns.catplot(x="Injection site", y="Percentage", hue="Treatment", data=df,
                     kind="bar", col="Test day")
    ax.set_ylabels('% place strategy')
    plt.savefig(op.join(figure_location, 'pm_barchart.pdf'), format='pdf')
    return ax


if __name__ == "__main__":

    df = load_deval_model_result()

    fig = plt.figure()
    performance_barchart(df)
    plt.show()
