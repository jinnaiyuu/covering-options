#!/bin/bash

source activate base

python -m cProfile rl_experiments.py --experiment offline --nepisodes 2 --nsteps 2 --noepisodes 2 --nosteps 2 --freqs 2 --noptions 2 --nsepisodes 2 --nssteps 2
