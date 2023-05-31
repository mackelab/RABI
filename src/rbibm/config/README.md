
## Configurate your experiments

The scripts are organized into several config groups. Most relevant to parameterize an experiment are

## 1) Tasks

The config group *tasks* parameterize a task and only contains a "name" and "params" that can be adjusted. See [here](https://github.com/mackelab/RABI/tree/main/src/rbibm/rbibm/tasks) for examples of creating a new task. Additionally, you would have to add a new task [here](https://github.com/mackelab/RABI/blob/main/src/rbibm/rbibm/utils/get_task.py) to connect the *name* with the *implementation*.

## 2) Model

This configures the (neural) density estimator. For this, you need to specify the following:
* Name: Name of the model
* module_path: Path to the module from which we can import the implementation.
* class_name: Name of the class to import
* params: Parms to initialize the model.
* embedding_net: See below.

Roughly, the model must output a PyTorch distribution object when calling the forward pass and should follow [this](https://github.com/mackelab/RABI/blob/main/src/rbi/rbi/models/base.py) abstract base classes.
### 2.1) Embedding net

The sub-config group does define an embedding net. The default is "identity," i.e., no embedding net. But you can, e.g., use an "mp" or specific other neural network architectures. In the config file, you must specify a "name" and some parameters optionally. Similar to the tasks, you have to link the *name* with the implementation [here](https://github.com/mackelab/RABI/blob/main/src/rbibm/rbibm/utils/embedding_nets.py).

## 3) Train
This parameterizes the training loop. Implemented are fKL and rKL (experimental), which do minimize the forward KL divergence or reverse KL divergence, respectively.
* class_name: The class of a loss function.
* loss_module: The module from which we can import the class.
* params: Additional hyperparameters
Implementation should follow this [base class](https://github.com/mackelab/RABI/blob/main/src/rbi/rbi/loss/base.py). Additionally, one can set a lot of standard hyperparameters, such as the learning rate ...

## 4) Defense
This parameterizes a defense method. The main config attributes are:
* defense_module: The module from which we can import the class.
* Defense: Defense class that implements the method.
* params: Additional hyperparameters.
The implementation should follow one of these [base classes](https://github.com/mackelab/RABI/blob/main/src/rbi/rbi/defenses/base.py).

## 5) Eval approx.
The "eval_approx" configuration group does evaluate a trained (or specified model). This config group roughly specifies which "metrics" should be evaluated to quantify how well the posterior is approximated.

### 5.1) Metric
This sub-config group links to an implementation of a metric. And should follow the following convention:
* Name:
  * metric_module: Module in which the class implementation is
  * metric_class: Class
  * params: Parameters.

There are already quite a lot implemented, e.g., "coverage" (expected coverage), "nll" (negative loglikelihood), or "ppl2d"(posterior predictive l2 distance). Some have additional requirements on the task, i.e., "c2st" (Classifier Two Sample Test), which requires a tractable "true posterior" implementation.

## 6) Eval rob.

The "eval_rob" configuration group does evaluate the adversarial robustness of the model empirically. The evaluation is done by generating adversarial examples using the chosen *attack* and evaluating a metric of the model on these examples.
There are three sub config group

### 6.1) Attack
Defines an attack and consists majorly of the following points

* attack_module: From where we can import the attack
* loss_module: From where we can import the loss (adversarial objective)
* attack_class: Ahe class of the class which can be imported from the "attack_module"
* attack_loss_fn: Loss fn
* attack_mc_budget: Monte Carlo budget for evaluating the loss if necessary
* params: Additional hyperparameters

### 6.2) Metric
A metric that compares the adversarial perturbed with the original prediction. This config group requires the following points:
* metric_module: From where we can import the class.
* metric_class: The class
* eval_mc_budget: MC budget for evaluating the metric
* num_adversarial_examples: Number of adversarial examples that should be generated.
* params: Additional hyperparameters. Implementation should follow this [interface](https://github.com/mackelab/RABI/tree/main/src/rbibm/rbibm/metric).

### 6.3) Metric_attack

This ensures that the adversarial objective of the attack matches the evaluation metric. For example, if we chose the KL divergence for evaluation, it overwrites the loss_fn of the attack to also be the KL divergence.

## 7) Eval_true

This is experimental and only loss implemented, but it aims to obtain the "true posterior" with traditional methods to be able to compare it with the approximations.

## 8) Launcher

This sets the default launcher for multi-runs. This typically uses "slurm" but can be switched to "local"- 

## 9) Partition
This can be used to select certain partitions and other SLURM-related parameters (i.e., number of GPUs, CPUs, TIMEOUT ...)
## 10) Sweeper

Typically the "sweeping" mode is set to "none," i.e., no sweeps. Then the script is executed once, and all results are saved to the database. In sweeping mode, it instead does not save all results but only runs the script, evaluates a loss, and saves the configurations that minimize/maximize the loss. Several Bayesian optimization algorithms are available,  also "multi-objective" optimization is possible. For example, this allows minimizing both an **approximation** and a **robustness** metric.
