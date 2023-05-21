# **R**obustness of **A**mortized **B**ayesian **I**nference (RABI)

This package contains the code to reproduce results from the paper: TODO LINK

### Installation

Installation of all dependencies should work by just running
```bash
conda create -n rbi1 python=3.9
python src/install.py -r
```

This should also install a script  **rbibm** to your command line (tested on Linux and Windows), which serves as main interface to run experiments on large scales.

### Usage

We use [hydra](https://hydra.cc/docs/intro/), to run or extend different parameterizations.

For example one can use:
```bash
rbibm task=lotka_volterra model=maf_pyro
```
you train and amortized posterior approximation network for the Lotka Volterra task using an Masked Autoregressive Flow. Additionally it also runs a bunch of metrics for evaluating the approximation quality or the robustness. You now can do the same i.e. using FIM regularization
```bash
rbibm task=lotka_volterra model=maf_pyro defense=fisher_trace
```
or for comparission we may want to compute the results for Trades or another defense we might run
```bash
rbibm task=lotka_volterra model=maf_pyro defense=l2pgdTrades_rKL
```

The default for multirun is slurm-submitit, which settings are likely incompatible with your system. You can adjust the setting in [here](https://github.com/mackelab/robustness_ai/tree/main/src/rbibm/config). 

### Reproducing results

In the [figures](https://github.com/mackelab/robustness_ai/tree/main/figures) folder you typically find a "run_****_experiment.sh" file (which runs the necessary experiments to create the figure) as well as one or two notebook which creates the plots. 

### Extending work

TODO
