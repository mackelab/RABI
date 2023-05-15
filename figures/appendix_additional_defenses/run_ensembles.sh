#!/bin/sh

rbibm +experiment=train_ensembles_small & rbibm +experiment=train_ensembles_lv & rbibm +experiment=train_ensembles_large  

IDX2=$(python -c 'from rbibm.utils.utils_data import query_main; df = query_main("benchmark", model_name="maf_ensemble", N_train=100000); idx = str(list(df.index)); idx = idx.replace("[", "").replace("]", "").replace(" ", ""); print(idx)')
echo $IDX2

rbibm +experiment=eval_l2 idx=$IDX2


