from numpy import isin
import pytest
import torch

from rbi.defenses import (
    FIMTraceRegularizer,
    TransformTraceRegularizer,
    JacobiRegularizer,
    FIMLargestEigenvalueRegularizer,
    L2PGDTrades,
    L1PGDTrades,
    LinfPGDTrades,
    DoublyRandomizedGaussianSmoothing,
    L1PGDAdversarialTraining,
    L2PGDAdversarialTraining,
    LinfPGDAdversarialTraining,
    L1UniformNoiseTraining,
    L2UniformNoiseTraining,
    LinfUniformNoiseTraining,
)


from rbi.loss.train_loss import NLLLoss, NegativeElboLoss
from rbi.models.parametetric_families import BernoulliNet, CategoricalNet
from tests.conftest import INPUT_DIMS

#from .test_models import MODELS, FLOWS, OUTPUT_DIMS, PARAMETRIC_FAMILIES


DEFENSES = [
    FIMTraceRegularizer,
    FIMLargestEigenvalueRegularizer,
    JacobiRegularizer,
    DoublyRandomizedGaussianSmoothing,
    L2PGDAdversarialTraining,
    L1PGDAdversarialTraining,
    LinfPGDAdversarialTraining,
    L1UniformNoiseTraining,
    L2UniformNoiseTraining,
    LinfUniformNoiseTraining,
    L2PGDTrades,
    L1PGDTrades,
    LinfPGDTrades,
]



def test_defense_general_continuous_initialize(model, defense):

    
    dim = model.input_dim
    net = model
    loss_fn = NLLLoss(net)

    defs = defense(net, loss_fn)
    defs.activate()

    if hasattr(net, "net_constraints"):
        assert len(net.net_constraints) > 0, "Defense seems not to be active"
    else:
        assert (
            len(loss_fn._pre_loss_regularizers) > 0
            or len(loss_fn._post_loss_regularizers) > 0
        ), "Regularizers seems to be not activated..."

    defs.deactivate()

    if hasattr(net, "net_constraints"):
        assert len(net.net_constraints) == 0, "Defense seems not to be deactivated"
    else:
        assert (
            len(loss_fn._pre_loss_regularizers) == 0
            and len(loss_fn._pre_loss_regularizers) == 0
        ), "Regularizers seems to be not deactivated..."

    defs.activate()
    q = net(torch.randn(10, dim))
    theta = q.sample()
   
    # Only if compatible models
    try:
        eval_loss_reg = loss_fn(torch.randn(10, dim), theta)
    except:
        eval_loss_reg = None 

    if eval_loss_reg is not None:
        assert torch.isfinite(eval_loss_reg).all(), "Not finite..."
    

