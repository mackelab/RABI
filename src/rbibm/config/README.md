
## Configurate your experiments

The scripts is organized into several config-groups. Most relevant to parameterize a experiment are

## 1) Tasks

The config group *tasks* parameterize a task and only contains a "name" and "params" that can be adjusted. See [here](https://github.com/mackelab/RABI/tree/main/src/rbibm/rbibm/tasks) for some examples, on how to create a new task. Additionally you would have to add a new task [here](https://github.com/mackelab/RABI/blob/main/src/rbibm/rbibm/utils/get_task.py) to connect the *name* with the *implementation*.

## 2) Model

This configurates the (neural) density estimator. For this you need to specify the following:
* name: Name of the model
* module_path: Path to the module from which we can import the implementation.
* class_name: Name of the class to import
* params: Parms to initialize the model.
* embedding_net: See below.

Roughly, the model must output a PyTorch distribution object when calling the forward pass and should roughly follow [this](https://github.com/mackelab/RABI/blob/main/src/rbi/rbi/models/base.py) abstract base classes.
### 2.1) Emebdding net

The sub-config group does define an embedding net. Default is "identity" i.e. no embedding net. But you can e.g. use an "mlp" or specific other neural network architectures. In the config file you have to specify a "name" and optionally som parameters. Similar to the tasks you have to link the *name* with the implementation [here](https://github.com/mackelab/RABI/blob/main/src/rbibm/rbibm/utils/embedding_nets.py).

## 3) Train
This parameterizes the training loop. Implemented is fKL and rKL (experimental) which does minimize the forward KL divergence or reverse KL divergence respectively.
* class_name: The class of a loss function.
* loss_module: The module from which we can import the class.
Implementation should follow this [base class](https://github.com/mackelab/RABI/blob/main/src/rbi/rbi/loss/base.py). Additionally one can set a lot of standard hyperparameters such as the learning rate ...

## 4) Defense


* **train**:
* **defense**:
* **eval_approx**:
* **eval_rob**:
* **eval_true**:

Relevent to adjust launcher configuration to your system are:
* **partition**:
* **sweeper**:
* config.yaml
