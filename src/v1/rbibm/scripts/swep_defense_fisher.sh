#!/bin/bash

rbibm -m name=main_sweeps defense=fisher_trace task=$1 model=$2 train.N_test=1000 sweeper=tpe_mo sweeper.objective=[rob_value,test_loss] sweeper.direction=[minimize,minimize] "defense.params.beta=interval(0.0001,100.)" hydra.sweeper.n_trials=200 run_eval_approx=false