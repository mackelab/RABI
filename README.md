# Welcome to this repo

You can install the required code base using pip

### Installation

Install the two packages using 
```bash
python src/install.py v1
```

This should install a script, which is added to your global PATH (tested on Linux and Windows). You can call it with the keyword **rbibm**. With that one could evaluate our results and more!  

### Usage

We use [hydra](https://hydra.cc/docs/intro/), so one also could test different parameterizations.

For example one can use:
```bash
rbibm task=lotka_volterra model=maf_pyro
```
you could run the Lotka Volterra task using an Masked Autoregressive Flow with robustness evaluation afterwards. You now can do the same i.e. using FIM regularization
```bash
rbibm task=lotka_volterra model=maf_pyro defense=fisher_trace
```
or to compare one could run Trades using
```bash
rbibm task=lotka_volterra model=maf_pyro defense=l2pgdTrades_rKL
```

The default for multirun is slurm-submitit, which settings are likely incompatible with your system. Thus one could switch to the local launcher.