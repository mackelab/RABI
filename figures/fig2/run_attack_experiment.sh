#!/bin/sh
# Required experiments for this to work

rbibm +experiment=train_small_flows & sleep 5 & rbibm +experiment=train_large_flows
IDX=$(python -c 'from rbibm.utils.utils_data import query_main; df = query_main("benchmark", defense="None"); idx = str(list(df.index)); idx = idx.replace("[", "").replace("]", "").replace(" ", ""); print(idx)')
echo $IDX
rbibm +experiment=eval_l2 idx=$IDX