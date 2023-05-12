#!/bin/sh


rbibm +experiment=appendix_vae & rbibm +experiment=appendix_vae1  & rbibm +experiment=appendix_vae2
rbibm +experiment=eval_l2 "idx=range(0,24)"

