# **R**obustness of **A**mortized **B**ayesian **I**nference (RABI)

This package contains the code to reproduce results from the paper: [Adversarial robustness of amortized Bayesian Inference](link)

## Installation

To install all dependencies, simply run the following commands:
```bash
conda create -n rbi python=3.9
conda activate rbi
python src/install.py -r
```
This will create a conda environment named "rbi" with Python 3.9 and install all the necessary packages.  Additionally, it will install a script called  **rbibm** into your command line (tested on Linux and Windows), which serves as main interface to run experiments on large scales.

## Usage

We use [hydra](https://hydra.cc/docs/intro/), to run or extend different parameterizations.

For example, you can use the following command to train and amortized posterior approximation network for the Lotka-Volterra task using a Masked Autoregressive Flow model:
```bash
rbibm task=lotka_volterra model=maf_pyro
```
This command not only trains the network but also runs various metrics to evaluate the approximation quality and robustness. To perform the same task using FIM regularization, you can run:
```bash
rbibm task=lotka_volterra model=maf_pyro defense=fisher_trace
```
or for comparission we may want to compute the results for Trades or another defense using
```bash
rbibm task=lotka_volterra model=maf_pyro defense=l2pgdTrades_rKL
```

### Multirun
Please note that the default configuration for multirun is set to "slurm-submitit", which settings are likely incompatible with your system. You can adjust the setting in [here](https://github.com/mackelab/robustness_ai/tree/main/src/rbibm/config).

If these setting are properly adjusted you should be able to run multiple experiments in parallel e.g. see for example [here](https://github.com/mackelab/RABI/blob/main/src/rbibm/config/experiment/train_small_flows.yaml) for an parameterization of an experiment. You can then run the full batch of jobs by using:
```bash
rbibm +experiment=train_small_flows
```
Alternatively you can explicitly run multiple experiments by common hydra syntax i.e. to investigate different hyperparameters i.e.
```bash
rbibm train.params.lr=1e-3,1e-4,1e-5
```
to train with different learning rates.

### Database

By default all results are stored in a database (by default a "data" folder is created [here](https://github.com/mackelab/RABI/tree/main/src/rbibm)). You can easily access the data through  python`rbibm.utils.utils_data.query`, which also is used by the implemented plotting capabilities see e.g. [here](https://github.com/mackelab/RABI/blob/main/figures/fig3/fig3.ipynb) for an example.

### Reproducing results

In the [figures](https://github.com/mackelab/robustness_ai/tree/main/figures) folder you typically find a "run_****_experiment.sh" file (which runs the necessary experiments to create the figure) as well as one or two notebook which creates the plots.

## Extending the work

TODO
