#!/bin/bash

rbibm -m name=main_sweeps defense=l2pgdAdvTrain task=$1 model=maf_pyro sweeper=tpe_mo sweeper.objective=[metric_approx_tilde,test_loss] sweeper.direction=[minimize,minimize] "defense.params.eps=interval(0.1,2.)" "defense.params.nb_iter=range(1,20)" "defense.params.eps_iter=interval(0.01,0.5)" hydra.sweeper.n_trials=500 eval_rob.eps=$2 eval_approx/metric=coverage