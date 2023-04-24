#!/bin/bash

rbibm -m name=main_sweeps idx=$1 train.N_test=1000 sweeper=tpe_mo sweeper.objective=[rob_value,metric_approx_tilde,runtime_rob] sweeper.direction=[maximize,maximize,minimize] eval_rob/attack=$2 eval_rob.eps=$3 eval_approx/metric=coverage "eval_rob.attack.params.eps_iter=interval(0.00001,2)" "eval_rob.attack.params.nb_iter=range(1,200)" "eval_rob.attack.attack_mc_budget=range(1,20)"