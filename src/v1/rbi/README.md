

# Robust Bayesian Inference

`rbi` is a PyTorch package for adversarially robust *amortized* Bayesian inference i.e. learning a function $f$ which given data estimates the posterior distribution $f(x) = p(\theta| x)$ induced by a fixed generative model $p(x, \theta) = p(x|\theta) p(\theta)$. 

## Installation

Can be easily installed via pip.

```commandline
pip install rbi
```

## Structure

The package is structured in 3 main parts.

### 1) Models

This implements certain conditional density estimators that are nicely suited for amortized inference.

```python
from rbi.models import InverseAffineAutoregressiveModel

model = InverseAffineAutoregressiveModel(10, 10)
x = torch.randn(100, 10)
q = model(x)                          # An distribution, batch_shape is 100 event_dim is 10

samples = q.sample((50,))             # Samples of shape (sampling_shape, batch_shape, event_dim)
log_prob = q.log_prob(samples)        # Log probabilities of shape (sampling_shape, batch_shape)
```

### 2) Loss

This implements certain loss functions for amortized Bayesian inference and evaluation.

```python
from rbi.loss import ReverseKLDivergence

loss = ReverseKLDivergence()
y = torch.randn(100, 10)
p = model(y)

val = loss(p,q)  # KL divergence estimate
```

### 2) Attacks

Attacks can be used to estimate the adversarial robustness of a model. Every attack has a method `perturb`, which perturbs a given input adversarially with the goal to distorted the approximate posterior as much as possible, typically limited by a certain tolerance level `eps`

```python
from rbi.attacks import L2PGDAttack

attack = L2PGDAttack(model, loss_fn = loss, eps=0.1)
x_tilde = attack.perturb(x)                         # Untargeted attack: maximizes D_KL(q(t|x) || q(t|x_tilde))
x_tilde = attack.perturb(x, p)                      # Targeted attack: minimizes D_KL( p || q(t|x_tilde))
```

### 3) Defense

Defends a model against adversarial attacks.

```python
from rbi.loss import NLLLoss
from rbi.defense import FIMTraceRegularizer

train_loss = NLLLoss(model)
defense = FIMTraceRegularizer(model, train_loss)
defense.activate()                                 # Training with train_loss now regularized.
```