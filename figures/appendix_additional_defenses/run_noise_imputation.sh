#!/bin/sh


rbibm +experiment=train_small_flows_noise train.N_train=100000 & rbibm +experiment=train_large_flows_noise train.N_train=100000


IDX2=$(python -c 'from rbibm.utils.utils_data import query_main; df = query_main("benchmark", defense="L2UniformNoiseTraining", N_train=100000); idx = str(list(df.index)); idx = idx.replace("[", "").replace("]", "").replace(" ", ""); print(idx)')
echo $IDX2

rbibm +experiment=eval_l2 idx=$IDX2


