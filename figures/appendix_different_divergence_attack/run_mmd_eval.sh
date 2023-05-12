#!/bin/sh


IDX=$(python -c 'from rbibm.utils.utils_data import query_main; df = query_main("benchmark", defense="None"); idx = str(list(df.index)); idx = idx.replace("[", "").replace("]", "").replace(" ", ""); print(idx)')
echo $IDX
rbibm +experiment=eval_l2_mmd idx=$IDX
