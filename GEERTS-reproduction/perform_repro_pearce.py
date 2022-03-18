#!/usr/bin/python

import sys

import warnings
warnings.filterwarnings('ignore')

from pearce import launch_pearce, fuse_plot_and_save

path = "pearce_from_script"
n_agents = 100

if len(sys.argv) >= 2:
    path = sys.argv[1]

if len(sys.argv) >= 3:
    n_agents = int(sys.argv[2])

print()
print("Simulating control agents")
launch_pearce(path, n_agents, lesion_hippocampus=False)
print()
print("Simulating lesioned agents")
launch_pearce(path, n_agents, lesion_hippocampus=True)
print()
print("Merging results")
fuse_plot_and_save(path)
