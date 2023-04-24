#!/bin/bash

rbibm -m name=main_sweeps defense=l2pgdTrades task=$1 model=$2 sweeper=tpe_mo sweeper.objective=[rob_value,test_loss] sweeper.direction=[minimize,minimize] "defense.params.beta=interval(0.001,20.)" run_eval_approx=false