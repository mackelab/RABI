#!/bin/sh

rbibm +experiment=train_small_others_adversarial
rbibm +experiment=train_large_others_adversarial
IDX=$(python -c 'from rbibm.utils.utils_data import query_main; df = query_main("benchmark", defense="L2PGDTargetedAdversarialTraining", model_name=["gaussian", "mixture_gaussian","multivariate_gaussian"]); idx = str(list(df.index)); idx = idx.replace("[", "").replace("]", "").replace(" ", ""); print(idx)')
echo $IDX
rbibm +experiment=eval_l2 idx=$IDX