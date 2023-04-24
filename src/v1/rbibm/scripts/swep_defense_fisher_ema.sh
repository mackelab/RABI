#!/bin/bash

rbibm -m name=main_sweeps defense=fisher_trace task=$1 eval_rob.eps=$2 model=maf_pyro train.N_test=1000 sweeper=tpe_mo sweeper.objective=[test_loss,rob_value] sweeper.direction=[minimize,minimize] defense.params.algorithm=ema "defense.params.beta=interval(0.0001,20.)" hydra.sweeper.n_trials=200 run_eval_approx=false partition=cuda device=cuda