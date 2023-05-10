#!/bin/sh

rbibm +experiment=train_small_others_fim
rbibm +experiment=train_vae_others_fim
rbibm +experiment=train_hh_others_fim
rbibm +experiment=train_spatial_sir_others_fim
IDX=$(python -c 'from rbibm.utils.utils_data import query_main; df = query_main("benchmark", defense="FIMTraceRegularizer", model_name=["gaussian", "mixture_gaussian","multivariate_gaussian"]); idx = str(list(df.index)); idx = idx.replace("[", "").replace("]", "").replace(" ", ""); print(idx)')
echo $IDX
rbibm +experiment=eval_l2 idx=$IDX