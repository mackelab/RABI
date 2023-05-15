#!/bin/sh

IDX2=$(python -c 'from rbibm.utils.utils_data import query_main; df = query_main("benchmark", model="maf",task=["gaussian_linear", "lotka_volterra", "sir", "vae_task"], defense="FIMTraceRegularizer", N_train=100000); idx = str(list(df.index)); idx = idx.replace("[", "").replace("]", "").replace(" ", ""); print(idx)')
echo $IDX2
rbibm +experiment=eval_true idx=$IDX2 partition=cuda





