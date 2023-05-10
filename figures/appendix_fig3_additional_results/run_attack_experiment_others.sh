#!/bin/sh
# Required experiments for this to work

rbibm +experiment=train_small_others
rbibm +experiment=train_large_others
IDX=$(python -c 'from rbibm.utils.utils_data import query_main; df = query_main("benchmark", defense="None", model_name=["gaussian", "mixture_gaussian","multivariate_gaussian"]); idx = str(list(df.index)); idx = idx.replace("[", "").replace("]", "").replace(" ", ""); print(idx)')
echo $IDX
rbibm +experiment=eval_l2 idx=$IDX