from pyparsing import removeQuotes
import pytest 
import torch
from torch import nn
import random

from rbi.models import (
    BernoulliNet,
    CategoricalNet,
    IndependentGaussianNet,
    MultivariateGaussianNet,
    MixtureDiagGaussianModel,
    AffineAutoregressiveModel,
    SplineAutoregressiveModel,
    SplineCouplingModel,
    InverseAffineAutoregressiveModel,
    InverseSplineAutoregressiveModel,
    AffineCouplingModel,
    MaskedAutoregressiveFlow,
    NeuralSplineFlow,
)

from rbi.models.base import ParametricProbabilisticModel


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
    LipschitzNeuralNet, 
    LipschitzEmbeddingNet,

)

from rbi.loss.train_loss import NLLLoss, NegativeElboLoss
from rbi.loss import (
    ForwardKLLoss,
    LogLikelihoodLoss,
    ReverseKLLoss,
    NegativeLogLikelihoodLoss,
    C2ST,
    C2STBayesOptimal,
    C2STKnn,
    MMDsquared,
    SymKLLoss,

)
from rbi.attacks import (
    GaussianNoiseAttack,
    SpectralKLAttack,
    SpectralTransformAttack,
    TruncatedGaussianNoiseAttack,
    L1UniformNoiseAttack,
    L2UniformNoiseAttack,
    LinfUniformNoiseAttack,
    RandomSearchAttack,
    IterativeRandomSearchAttack,
    L2MomentumIterativeAttack,  # type: ignore
    LinfMomentumIterativeAttack,  # type: ignore
    L2PGDAttack,  # type: ignore
    L1PGDAttack,  # type: ignore
    LinfPGDAttack,  # type: ignore
    FGSM,  # type: ignore
    FGM,  # type: ignore
    GradientSignAttack,  # type: ignore
    GradientAttack,  # type: ignore
    L2BasicIterativeAttack,  # type: ignore
    LinfBasicIterativeAttack,  # type: ignore
)


# Test variables --------------------------------------------------

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES += ["cuda"]


INPUT_DIMS = [1] + [random.randint(2, 10) for _ in range(1)]
OUTPUT_DIMS = [1] + [random.randint(2, 5) for _ in range(1)]

BATCH_SHAPES = [(),(10,), (2, 1, 3), (10, 1)]

# ATTACK variables ------------------------------------------------

EPS = [random.random()*5 for _ in range(2)]

ATTACKS = [
    GaussianNoiseAttack,
    SpectralKLAttack,
    TruncatedGaussianNoiseAttack,
    L1UniformNoiseAttack,
    L2UniformNoiseAttack,
    LinfUniformNoiseAttack,
    L2MomentumIterativeAttack,
    LinfMomentumIterativeAttack,
    L2PGDAttack,
    L1PGDAttack,
    LinfPGDAttack,
    FGSM,
    FGM,
    GradientSignAttack,
    GradientAttack,
    L2BasicIterativeAttack,
    LinfBasicIterativeAttack,
]

# Model variables ----------------------------------------------------

PARAMETRIC_FAMILIES = [
    BernoulliNet,
    CategoricalNet,
    IndependentGaussianNet,
    MultivariateGaussianNet,
    MixtureDiagGaussianModel,
]

FLOWS = [
    AffineAutoregressiveModel,
    SplineAutoregressiveModel,
    SplineCouplingModel,
    InverseAffineAutoregressiveModel,
    InverseSplineAutoregressiveModel,
    AffineCouplingModel,
    NeuralSplineFlow,
    MaskedAutoregressiveFlow,
]

MODELS = PARAMETRIC_FAMILIES + FLOWS

CONTINUOUS_MODEL = PARAMETRIC_FAMILIES[2:] + FLOWS

# Train loss--------------------------------------------------------
TRAIN_LOSS = [NLLLoss, NegativeElboLoss]
EVAL_LOSS = [
    ForwardKLLoss,
    ReverseKLLoss,
    LogLikelihoodLoss,
    NegativeLogLikelihoodLoss,
    C2ST,
    C2STKnn,
    C2STBayesOptimal,
    MMDsquared,
]
DIVERGENCES = [ReverseKLLoss, ForwardKLLoss, SymKLLoss]
C2STS = [C2ST, C2STKnn, C2STBayesOptimal]
MC_SAMPLES = [1, 2,3]

Q1 = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
Q2 = torch.distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
Q3 = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2) * 0.1)

# Defense --------------------------------------------------------------

DEFENSES = [
    FIMTraceRegularizer,
    FIMLargestEigenvalueRegularizer,
    TransformTraceRegularizer,
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
    LipschitzNeuralNet,
    LipschitzEmbeddingNet
]

# Autograd tools--------------------------------------------------------------------

class TestLinear:
    def __init__(self, dim=2) -> None:
        self.dim = dim
        self.W = torch.randn(dim, dim)
        self.jac = self.W

    def __call__(self, x):
        return (self.W @ x.T).T


class TestQuadratic:
    def __init__(self, dim) -> None:
        self.dim = dim
        self.W = torch.randn(dim, dim)
        self.jac = lambda x: ((self.W + self.W.T) @ x.T).T.unsqueeze(-2)
        self.hessian = lambda x: (self.W + self.W.T).unsqueeze(0)

    def __call__(self, x):
        return torch.einsum("bi, ij, bj -> b", x, self.W, x).unsqueeze(-1)




NEURAL_NETS_IN = [2,2, 100, 2]
NEURAL_NETS_OUT = [2,100, 2, 2]

NEURAL_NETS = list(zip([nn.Sequential(nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 2)), nn.Sequential(nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 100)), nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 2)), nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Identity(),
        nn.Linear(10, 2),
    )], NEURAL_NETS_IN, NEURAL_NETS_OUT))

#----------------------------------------------------------------------------------

@pytest.fixture(params=NEURAL_NETS)
def mlp(request):
    return request.param


@pytest.fixture(params=[2,5])
def linear(request):
    return TestLinear(request.param)


@pytest.fixture(params=[2,5])
def quadratic(request):
    return TestQuadratic(request.param)


@pytest.fixture(params=DEVICES, ids=[f"device={d}" for d in DEVICES])
def device(request):
    return request.param

@pytest.fixture(params=EPS, ids=[f"eps={e}, " for e in EPS])
def eps(request):
    return request.param

@pytest.fixture(params=INPUT_DIMS)
def dims(request):
    return request.param


@pytest.fixture(params=[(2, 2, m)  for m in MODELS], ids = [f"{m.__name__}(2, 2), "  for m in MODELS])
def model_2d(request):
    i,o, m = request.param
    if m == BernoulliNet:
        o = 1

    if m in PARAMETRIC_FAMILIES:
        return m(i,o, hidden_dims=[5])
    else:
        return m(i, o ,hidden_dims=[5], num_transforms=1)

@pytest.fixture(params=[(i, o, m)  for i in INPUT_DIMS for o in OUTPUT_DIMS for m in MODELS], ids = [f"{m.__name__}({i}, {o}), "  for i in INPUT_DIMS for o in OUTPUT_DIMS for m in MODELS])
def model(request):
    i,o, m = request.param
    if m == BernoulliNet:
        o = 1

    hidden_dims = [5,5]

    if m in PARAMETRIC_FAMILIES:
        return m(i,o, hidden_dims=hidden_dims)
    else:
        return m(i, o ,hidden_dims=hidden_dims, num_transforms=1)

@pytest.fixture(params=[(i, o, m)  for i in INPUT_DIMS for o in OUTPUT_DIMS for m in FLOWS], ids = [f"{m.__name__}({i}, {o}), "  for i in INPUT_DIMS for o in OUTPUT_DIMS for m in FLOWS])
def flow(request):
    i,o, m = request.param
    if m == BernoulliNet:
        o = 1
        
    return m(i,o, hidden_dims = [20], num_transforms=1)

@pytest.fixture(params=[(i, o, m)  for i in INPUT_DIMS for o in OUTPUT_DIMS for m in PARAMETRIC_FAMILIES], ids = [f"{m.__name__}({i}, {o}), "  for i in INPUT_DIMS for o in OUTPUT_DIMS for m in PARAMETRIC_FAMILIES])
def parametric_family(request):
    i,o, m = request.param
    if m == BernoulliNet:
        o = 1
        
    return m(i,o)

@pytest.fixture(params=[(i, o, m)  for i in INPUT_DIMS for o in OUTPUT_DIMS for m in CONTINUOUS_MODEL], ids = [f"{m.__name__}({i}, {o}), "  for i in INPUT_DIMS for o in OUTPUT_DIMS for m in CONTINUOUS_MODEL])
def continuous_model(request):
    i,o, m = request.param
    if m == BernoulliNet:
        o = 1
        
    if m in PARAMETRIC_FAMILIES:
        return m(i,o, hidden_dims=[20])
    else:
        return m(i, o ,hidden_dims=[20], num_transforms=1)

@pytest.fixture(params= BATCH_SHAPES, ids = [str(p) for p in BATCH_SHAPES])
def batch_shape(request):
    return request.param

@pytest.fixture(params= BATCH_SHAPES, ids = [str(p) for p in BATCH_SHAPES])
def sampling_batch(request):
    return request.param

@pytest.fixture(params= MC_SAMPLES, ids = [str(p) for p in MC_SAMPLES])
def mc_samples(request):
    return request.param

@pytest.fixture(params= EVAL_LOSS, ids = [str(p.__name__) for p in EVAL_LOSS])
def eval_loss(request):
    return request.param


@pytest.fixture()
def nllloss(model):
    return NLLLoss(model)

@pytest.fixture()
def negative_elbo_loss(continuous_model, device):
    input_dim = continuous_model.input_dim
    output_dim = continuous_model.output_dim
    model = continuous_model.to(device)
    prior = torch.distributions.MultivariateNormal(torch.zeros(output_dim, device = device), torch.eye(output_dim, device = device))
    loglikelihood_fn = lambda x: torch.distributions.Independent(
        torch.distributions.Uniform(
            torch.ones((input_dim,), device=device) * -1000, torch.ones((input_dim,), device=device) * 1000
        ),
        1,
    )

    potential_fn = lambda x, theta: prior.log_prob(theta) + loglikelihood_fn(
        theta
    ).log_prob(x)

    return model, NegativeElboLoss, prior, loglikelihood_fn, potential_fn, device

@pytest.fixture(params=DIVERGENCES)
def divergence(request):
    return request.param


@pytest.fixture(params=C2STS)
def c2sts(request):
    return request.param

@pytest.fixture()
def example_distributions():
    return [Q1, Q2, Q3]

@pytest.fixture(params = ATTACKS)
def attack(request):
    return request.param

@pytest.fixture(params = DEFENSES)
def defense(request):
    return request.param