import pytest
from rbibm.tasks import (
    GaussianLinearTask,
    LotkaVolterraTask,
    VAETask,
    SquareTask,
    HHTask,
    SIRTask,
    RBFRegressionTask,
)

from rbi.models import (
    IndependentGaussianNet,
    MultivariateGaussianNet,
    MixtureDiagGaussianModel,
    InverseAffineAutoregressiveModel,
    InverseSplineAutoregressiveModel,
)

from rbibm.metric.approximation_metric import (
    ReverseKL2GroundTruthMetric,
    ForwardKL2GroundTruthMetric,
    C2STBayesOptimal2GroundTruthMetric,
    ExpectedCoverageMetric,
    SimulationBasedCalibrationMetric,
    R2LinearFit2Potential,
)
from rbibm.metric.robustness_metric import (
    ReverseKLRobMetric,
    ForwardKLRobMetric,
    MMDsquaredRobMetric,
)


import torch

#
APPROX_METRIC = [
    ReverseKL2GroundTruthMetric,
    ForwardKL2GroundTruthMetric,
    C2STBayesOptimal2GroundTruthMetric,
    ExpectedCoverageMetric,
    SimulationBasedCalibrationMetric,
    R2LinearFit2Potential,
]
ROB_METRIC = [ReverseKLRobMetric, ForwardKLRobMetric, MMDsquaredRobMetric]

# Task specific stuff ------------------------------------------------

BASE_DIMS_TO_TEST = [1, 5, 10]
BASE_PRIOR_MEAN = [-1.0, 0.0, 10.0]
BASE_PRIOR_SCALES = [0.1, 1.0]
BASE_LIKELIHOOD_SCALES = [0.1, 1.0, 10.0]

# If they are very large the ode becomes hard to solve... -> Very spiky
LV_PRIOR_MEAN = [-0.2, 0.0, 0.2]
LV_PRIOR_SCALES = [0.1, 0.2, 0.5]

OBSERVED_POINTS = [10, 50]

VAE_LATENT_DIMS = [2, 5]

TASKS = [LotkaVolterraTask, GaussianLinearTask, VAETask, HHTask, SIRTask, RBFRegressionTask]
TAKS_WITH_POTENTIAL = [LotkaVolterraTask, GaussianLinearTask, VAETask, HHTask, SIRTask, RBFRegressionTask]

MODELS = [
    IndependentGaussianNet,
    MultivariateGaussianNet,
    MixtureDiagGaussianModel,
    InverseAffineAutoregressiveModel,
    InverseSplineAutoregressiveModel,
]

# ---------------------------------------------------------------------

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES += ["cuda"]


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


@pytest.fixture(params=TASKS)
def task(request):
    return request.param()



@pytest.fixture(params = APPROX_METRIC)
def approximation_metric(request):
    return request.param

@pytest.fixture(params = ROB_METRIC)
def rob_metric(request):
    return request.param


@pytest.fixture(params=TAKS_WITH_POTENTIAL)
def task_with_potential(request):
    return request.param()


@pytest.fixture(params=MODELS)
def model(request):
    return request.param


@pytest.fixture()
def gaussian_linear_default():
    return GaussianLinearTask()


@pytest.fixture()
def lotka_volterra_default():
    return LotkaVolterraTask()


@pytest.fixture()
def hh_default():
    return HHTask()


@pytest.fixture()
def vae_default():
    return VAETask()


@pytest.fixture(
    params=[
        (d, p_m, p_s, l_s)
        for d in BASE_DIMS_TO_TEST
        for p_m in BASE_PRIOR_MEAN
        for p_s in BASE_PRIOR_SCALES
        for l_s in BASE_LIKELIHOOD_SCALES
    ],
    ids=[
        f"(dim={d}, prior_mean={p_m}, prior_scale={p_s}, likelihood_scale={l_s})"
        for d in BASE_DIMS_TO_TEST
        for p_m in BASE_PRIOR_MEAN
        for p_s in BASE_PRIOR_SCALES
        for l_s in BASE_LIKELIHOOD_SCALES
    ],
)
def gaussian_linear(request):
    return GaussianLinearTask(*request.param)

@pytest.fixture
def glr_rbf():
    return RBFRegressionTask()


@pytest.fixture(
    params=[
        (t_obs, p_m, p_s, l_s)
        for t_obs in OBSERVED_POINTS
        for p_m in LV_PRIOR_MEAN
        for p_s in LV_PRIOR_SCALES
        for l_s in BASE_LIKELIHOOD_SCALES
    ],
    ids=[
        f"(time_points_obs={t_obs}, prior_mean={p_m}, prior_scale={p_s}, observation_noise={l_s})"
        for t_obs in OBSERVED_POINTS
        for p_m in LV_PRIOR_MEAN
        for p_s in LV_PRIOR_SCALES
        for l_s in BASE_LIKELIHOOD_SCALES
    ],
)
def lotka_volterra(request):
    params = request.param
    return LotkaVolterraTask(
        t_max=5.0,
        time_points_observed=params[0],
        prior_mean=params[1],
        prior_scale=params[2],
    )


@pytest.fixture(
    params=[
        (t_obs, l_s) for t_obs in OBSERVED_POINTS for l_s in BASE_LIKELIHOOD_SCALES
    ],
    ids=[
        f"(time_points_obs={t_obs}, observation_noise={l_s})"
        for t_obs in OBSERVED_POINTS
        for l_s in BASE_LIKELIHOOD_SCALES
    ],
)
def hh(request):
    params = request.param
    return HHTask(
        t_max=5.0,
        I_on=1.0,
        I_off=4.0,
        time_points_observed=params[0],
        observation_noise=params[1],
    )


@pytest.fixture(params=VAE_LATENT_DIMS, ids=["latent_dim=2", "latent_dim=5"])
def vae(request):
    param = request.param
    return VAETask(latent_dim=param)


@pytest.fixture()
def sir():
    return SIRTask()


@pytest.fixture(
    params=[
        (p_m, p_s, l_s)
        for p_m in BASE_PRIOR_MEAN
        for p_s in BASE_PRIOR_SCALES
        for l_s in BASE_LIKELIHOOD_SCALES
    ],
    ids=[
        f"(prior_mean={p_m}, prior_scale={p_s}, likelihood_scale={l_s})"
        for p_m in BASE_PRIOR_MEAN
        for p_s in BASE_PRIOR_SCALES
        for l_s in BASE_LIKELIHOOD_SCALES
    ],
)
def square_task(request):
    param = request.param
    return SquareTask(
        prior_mean=param[0], prior_scale=param[1], likelihood_scale=param[2]
    )


# Models -------------------------------------------------------------------
