from numpy import isin
import pytest
import torch

from rbi.utils.mcmc import MCMC
from rbi.utils.mcmc_kernels import (
    GaussianKernel,
    AdaptiveGaussianKernel,
    AdaptiveMultivariateGaussianKernel,
    LangevianKernel,
    HMCKernel,
    LearnableIndependentKernel,
    SliceKernel,
    LatentSliceKernel,
    IndependentKernel,
    KernelScheduler,
)

from sbi.utils.metrics import c2st


SIMPEL_KERNEL = [GaussianKernel, AdaptiveGaussianKernel, AdaptiveMultivariateGaussianKernel]
GRADIENT_BASED_KERNEL = [LangevianKernel, HMCKernel]
SLICE_KERNEL = [SliceKernel, LatentSliceKernel]
INDEPENDENT_KERNEL = [IndependentKernel, LearnableIndependentKernel]
COMBINED_KERNELS = [KernelScheduler]


@pytest.fixture(params=[1, 2,5])
def normal_potential_task(request):
    dim = request.param
    p = torch.distributions.Independent(torch.distributions.Normal(2*torch.ones(dim), 0.5*torch.ones(dim)),-1)

    def potential_fn(theta):
        return p.log_prob(theta)

    return potential_fn, p, dim



@pytest.fixture(params=[1, 2,5])
def normal_potential_task_gpu(request):
    dim = request.param
    p = torch.distributions.Independent(torch.distributions.Normal(2*torch.ones(dim, device="cuda"), 0.5*torch.ones(dim, device="cuda")),-1)

    def potential_fn(theta):
        return p.log_prob(theta)

    return potential_fn, p, dim

@pytest.fixture
def conditional_nomrmal_potential_task():
    dim = 1
    context = torch.ones(1) * 4
    def potential_fn(x, theta):
            p = torch.distributions.Independent(torch.distributions.Normal(theta**2, 0.5*torch.ones(dim)),1)
            return p.log_prob(x).squeeze()
    
    return potential_fn, context, dim, _GROUND_TRUTH

@pytest.fixture(
    params=SIMPEL_KERNEL + GRADIENT_BASED_KERNEL + SLICE_KERNEL + INDEPENDENT_KERNEL + COMBINED_KERNELS
)
def kernel(request):
    return request.param


def test_conditional_mcmc(conditional_nomrmal_potential_task, kernel):
    potential_fn, x, dim, true_samples = conditional_nomrmal_potential_task
    proposal =  torch.distributions.Normal(torch.zeros(1), 2*torch.ones(1))
    if kernel in SIMPEL_KERNEL:
        k = kernel()
    elif kernel == IndependentKernel:
        # This is to bad...
        return
    elif kernel == LearnableIndependentKernel:
        k = kernel()
    elif kernel == KernelScheduler:
        k = KernelScheduler([IndependentKernel(proposal), GaussianKernel(), AdaptiveGaussianKernel()], [10, 20, 50])
    else:
        k = kernel(potential_fn, context=x)
    mcmc = MCMC(k, potential_fn, proposal, context=x)
    
    samples = mcmc.run(1000)

    acc = c2st(samples.reshape(1000,dim), true_samples.reshape(1000,dim))

    assert acc < 0.75, f"We have a c2st of {acc}, which however should be close to 0.5..."

def test_conditional_mcmc_batched(conditional_nomrmal_potential_task, kernel):
    potential_fn, x, dim, true_samples = conditional_nomrmal_potential_task
    x = x.repeat(2,1)
    proposal =  torch.distributions.Normal(torch.zeros(1), 2*torch.ones(1))

    if kernel in SIMPEL_KERNEL:
        k = kernel()
    elif kernel == IndependentKernel:
        # This is to bad...
        return
    elif kernel == LearnableIndependentKernel:
        k = kernel()
    elif kernel == KernelScheduler:
        k = KernelScheduler([IndependentKernel(proposal), GaussianKernel(), AdaptiveGaussianKernel()], [10, 20, 50])
    else:
        k = kernel(potential_fn, context=x)
    mcmc = MCMC(k, potential_fn, proposal, context=x)
    
    samples = mcmc.run(1000)

    samples1 = samples[:, 0]
    samples2 = samples[:, 1]

    acc = c2st(samples1.reshape(1000,dim), true_samples.reshape(1000,dim))

    assert acc < 0.75, f"We have a c2st of {acc}, which however should be close to 0.5..."

    acc = c2st(samples2.reshape(1000,dim), true_samples.reshape(1000,dim))

    assert acc < 0.75, f"We have a c2st of {acc}, which however should be close to 0.5..."


def test_mcmc(normal_potential_task, kernel):
    potential_fn, p, dim = normal_potential_task
    proposal =  torch.distributions.Independent(torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)),1)
    if kernel in SIMPEL_KERNEL:
        k = kernel()
    elif kernel == IndependentKernel:
        k = kernel(p)
    elif kernel == LearnableIndependentKernel:
        k = kernel()
    elif kernel == KernelScheduler:
        k = KernelScheduler([IndependentKernel(proposal), GaussianKernel(), AdaptiveGaussianKernel()], [10, 20, 50])
    else:
        k = kernel(potential_fn)
    mcmc = MCMC(k, potential_fn, proposal)
    
    samples = mcmc.run(1000)
    true_samples = p.sample((1000,))

    acc = c2st(samples.reshape(1000,dim), true_samples.reshape(1000,dim))

    assert acc < 0.75, f"We have a c2st of {acc}, which however should be close to 0.5..."


def test_mcmc_gpu(normal_potential_task_gpu, kernel):
    potential_fn, p, dim = normal_potential_task_gpu
    proposal =  torch.distributions.Independent(torch.distributions.Normal(torch.zeros(dim, device="cuda"), torch.ones(dim, device="cuda")),1)
    if kernel in SIMPEL_KERNEL:
        k = kernel()
    elif kernel == IndependentKernel:
        k = kernel(p)
    elif kernel == LearnableIndependentKernel:
        k = kernel()
    elif kernel == KernelScheduler:
        k = KernelScheduler([IndependentKernel(proposal), GaussianKernel(), AdaptiveGaussianKernel()], [10, 20, 50])
    else:
        k = kernel(potential_fn)
    mcmc = MCMC(k, potential_fn, proposal, device="cuda")
    
    samples = mcmc.run(1000)
    true_samples = p.sample((1000,))

    acc = c2st(samples.reshape(1000,dim), true_samples.reshape(1000,dim))

    assert acc < 0.75, f"We have a c2st of {acc}, which however should be close to 0.5..."


def test_mcmc_jit(normal_potential_task, kernel):
    potential_fn, p, dim = normal_potential_task

    proposal =  torch.distributions.Independent(torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)),1)
    if kernel in SIMPEL_KERNEL:
        k = kernel()
    elif kernel == IndependentKernel:
        k = kernel(p)
    elif kernel == LearnableIndependentKernel:
        k = kernel()
    elif kernel == KernelScheduler:
        k = KernelScheduler([IndependentKernel(proposal), GaussianKernel(), AdaptiveGaussianKernel()], [10, 20, 50])
    else:
        k = kernel(potential_fn)

    mcmc = MCMC(k, potential_fn, proposal, jit=True)
    
    samples = mcmc.run(1000)
    true_samples = p.sample((1000,))

    acc = c2st(samples.reshape(1000,dim), true_samples.reshape(1000,dim))

    assert acc < 0.75, f"We have a c2st of {acc}, which however should be close to 0.5..."













_GROUND_TRUTH = torch.tensor([-2.1970,  1.8402,  2.0309, -1.9218, -1.8909,  1.9087, -1.9086,  2.1310,
         2.0330, -1.9735, -2.0762,  1.9192, -2.0276,  1.9527, -1.9362,  1.8796,
        -2.0395,  2.1148, -1.8631, -1.9576, -2.2008,  2.0914, -2.1252, -1.9929,
        -1.9008,  2.1819, -2.1757,  2.0489,  1.9171,  2.0563,  1.9329, -2.0501,
         2.0395,  2.0806,  2.0700, -1.9872, -1.9664, -1.9256,  1.8592,  1.8414,
        -1.9705, -2.0633,  1.8801, -2.0288,  2.0655, -2.0280,  2.0682,  1.9176,
         2.1036,  1.7743, -2.1995,  2.1226,  1.9838, -2.0509,  2.0660, -2.1311,
        -1.9069, -1.8531,  2.0434,  1.8241,  2.0035, -1.8989,  2.0286, -2.0059,
         2.0377, -2.2187, -2.0259,  2.0276, -1.9472, -2.0128,  1.8926,  2.0521,
        -1.9709,  2.1635, -2.0863, -1.9711, -2.1010,  2.0131, -2.0785, -1.9875,
         1.9637, -1.7550, -1.9533,  2.0011,  2.1035, -2.0513, -2.0699,  1.9157,
         1.9659,  1.8844,  1.9544,  1.8956, -1.7753, -1.8341,  1.8223,  2.0817,
        -2.0618, -2.0624,  1.8374, -2.0179, -2.0014, -2.2886, -2.1648,  1.8804,
         2.0452, -2.0552, -2.2471, -1.8452,  1.9748,  1.8233, -1.8631,  2.1528,
         1.8901,  1.8968,  2.1926, -1.9447,  1.9673, -1.9823, -1.9743,  2.0789,
        -2.0391, -1.9833,  1.9599,  2.0635, -2.1268, -2.0637, -2.1368, -2.1584,
         2.1136, -1.9878,  1.9934, -1.6135, -1.9575, -2.0525, -2.0689, -1.9615,
         2.1801,  1.7560, -2.0812,  2.0167,  2.0689,  1.8757,  1.6412,  1.7704,
        -2.1147,  1.8342, -1.9939,  1.9984,  2.2711,  1.9020, -2.0773,  2.0341,
         1.9069, -2.1178, -1.9694, -1.9673, -1.9974, -2.0806,  2.1758,  1.9531,
        -1.8469,  1.8614,  1.7595, -1.9829,  2.0536,  2.1118,  2.1270,  1.9466,
         1.9595,  1.8909,  1.9239, -1.8843, -2.0865,  1.8040,  1.8280, -2.2156,
        -1.8742, -1.8462, -1.9096,  1.8836, -2.0871,  2.0020,  2.0922, -1.9944,
        -2.0892, -2.2068,  1.8550, -1.9999, -2.0142,  2.1366, -2.0249, -1.9970,
        -1.9730, -2.0157,  1.9883, -1.7735,  2.0345,  2.0297, -1.7760,  1.9879,
        -1.7580, -1.9317, -2.0720, -1.9257, -1.9341, -1.9870,  2.0855, -2.0937,
         1.9690,  1.9091, -1.9670,  2.0496,  1.8563,  2.0679, -1.9987,  1.9032,
         2.0818,  2.0125, -2.0514,  2.0782,  2.1203, -2.1716, -1.8884, -1.9297,
        -2.0631,  2.0037,  1.9885, -2.1372, -1.6208, -2.0732,  2.0954,  1.9449,
        -1.8428, -1.9617, -2.0574,  2.0651, -1.8942, -2.0195,  1.8966, -1.8640,
        -2.1044, -2.0911, -2.0909,  2.0741, -1.9611, -2.0048,  2.0438, -1.9199,
         2.1234,  1.9018, -1.8257, -2.1395,  2.1275,  2.2887,  2.0157,  2.0912,
        -2.0222,  2.2443, -1.9546,  2.0697, -2.0279,  2.0244,  1.9477, -1.9471,
         1.8488,  1.8228, -1.9517, -1.9758, -2.0586, -1.8595, -2.0567, -1.8218,
         2.0962,  1.9117, -2.0561, -1.9377,  2.1967,  1.9329,  2.0926, -2.1120,
        -1.7966,  1.9577, -2.1443,  2.2194, -1.7887,  1.9696, -2.1234,  1.9180,
        -2.0002,  2.0308, -2.1919, -1.9448, -1.9497, -1.9514, -1.8574, -1.8162,
         2.0943,  1.9553, -1.9014,  2.0549,  2.0368,  1.8138, -1.8600, -2.1213,
        -2.0891,  2.1235,  1.6843,  1.8162, -2.0430,  1.9118, -2.0203,  1.9626,
         2.1013,  1.8486,  2.0585, -1.9268,  1.9758, -2.0222, -2.1874, -2.0771,
         2.1549, -2.0614,  1.9790,  1.6362,  1.8537,  1.9540, -1.9334, -1.9731,
         1.9718,  2.0731, -2.2236, -2.0454,  2.1367,  1.9569, -2.1082, -2.1133,
         1.7778,  2.0859,  1.9675,  1.8912,  2.0770, -1.9025,  1.8437,  2.0278,
         1.7686,  1.8378,  1.8889, -2.0056,  1.9733, -2.0922, -2.0716,  2.0129,
        -1.8890, -1.9382,  1.9524, -1.8471, -1.9516, -1.8990, -1.8441, -2.0447,
        -2.0089, -1.8452,  1.8943,  2.1599,  2.1127,  1.8782,  2.0124,  2.0594,
         1.9666, -1.9355,  2.0004,  2.0173, -1.9794, -2.0104,  1.9701,  1.7581,
         1.9245, -2.0109,  1.8769, -2.0523, -1.8832, -2.0650, -1.7603,  1.9184,
        -2.1201,  2.1795,  2.0620,  1.9788,  2.0892, -2.0220,  2.1862, -1.8524,
         1.6877, -2.0206,  1.7932,  1.9762,  1.9830, -1.8612, -2.0382,  1.9353,
        -1.9929,  2.0688, -1.9543,  2.1146,  1.8983,  2.0630,  1.9361,  2.0281,
         2.0281, -2.1561,  2.0942,  2.0980, -1.8065,  2.1677,  1.9050,  2.0930,
        -1.9727, -2.1660, -1.7587, -1.8721,  2.0454, -2.0075, -1.9235, -1.7851,
         1.9651, -1.9473,  1.8078,  2.0785,  2.0133, -1.8414, -2.2160,  2.0554,
        -1.8670,  2.0104, -1.9663, -2.0128, -2.0812, -1.9674,  2.0041, -1.9963,
        -2.0556, -2.2752,  2.0431,  1.8879, -1.7951, -1.9666, -1.8401, -1.9046,
         1.7330, -1.6719,  2.1058,  1.8765,  2.0577, -1.8302,  1.9900,  2.0516,
         2.0000, -2.0617, -2.0250, -2.1127, -2.0856, -2.0451,  1.9795,  1.8222,
         2.0617,  2.1015,  1.7238,  1.9574, -2.0082, -1.9355,  2.1772,  1.7775,
        -1.7696,  2.0157, -2.0025, -2.1607, -1.9919,  2.0427, -1.9586, -1.8831,
        -1.9018, -1.9528,  2.1892, -1.7531, -1.9522,  2.0451,  1.5493,  1.9652,
        -2.0111, -1.7111,  2.0023, -2.0679, -2.1524,  2.0618, -1.9393, -1.9343,
        -2.0160,  1.7982, -2.1416,  1.9437,  1.9167, -2.0572,  1.9621, -1.9435,
        -2.0853, -1.9183, -1.9881,  2.1294,  2.0157,  1.9039, -1.7730,  2.1954,
         1.9582,  1.6286,  1.7552, -2.0669, -2.0456, -2.1204,  1.9446, -1.9593,
         2.0314,  2.0903, -1.9716,  1.5148,  2.1025, -2.1780, -2.0987,  1.9667,
        -2.1232, -1.8711, -2.0887, -1.8506,  1.8683,  2.1254, -2.0164, -1.8541,
         1.9984,  2.0078, -1.9930,  1.9100, -2.0755,  2.0131, -2.1461, -1.8179,
         1.9713,  1.9373, -1.8124, -2.1630,  2.1317, -2.0064, -2.0769, -2.0345,
        -2.0454,  1.8932,  1.9868, -2.0852, -2.0514, -1.9143, -2.0369,  2.0102,
        -2.1594, -2.1311,  2.1068, -2.0495,  2.0204,  2.0091,  2.0875, -2.0643,
        -1.8369, -1.8652, -2.0957, -2.1168, -2.0132, -2.1950,  1.6162, -1.9156,
        -2.0449,  1.8744,  1.9426, -2.0654, -2.0848, -1.8593,  1.9031, -1.8025,
        -2.0143, -1.9679,  1.9859,  2.1351, -1.9173,  2.0881, -2.0451, -2.1262,
        -1.8752, -1.9183, -2.0259, -1.9323,  1.8276,  1.7563, -2.0450,  2.2526,
         1.9182, -1.9903, -2.0335,  1.9354,  1.8059,  2.0276, -2.0181,  2.1685,
         1.9393,  2.2037,  1.8373,  2.0104, -1.8956,  2.2000, -2.0218, -1.9323,
        -2.0362,  2.0514, -1.7957, -1.9522, -2.1559,  2.1418, -1.9312, -1.9398,
        -1.9093, -1.9990,  2.1031,  2.0000,  1.9288, -1.9195,  2.0987,  2.1547,
         2.0290, -1.9596, -2.1838, -1.9725, -1.9904, -2.0999, -1.8383, -1.8345,
         2.0533,  2.0866,  2.0472, -1.7800,  1.9691, -2.1127, -2.0136,  1.9168,
        -1.9714, -1.7794,  2.1809, -2.0032,  1.8796, -1.9860,  2.1321, -2.0039,
         2.2875, -1.9751,  1.8944, -2.0041, -1.8884,  1.7626, -1.9082, -1.7755,
         1.9340,  2.0375,  1.7785,  2.1806,  2.0562, -2.1077,  2.1030,  1.7434,
         1.7939, -2.1927,  1.9097,  2.0868, -2.0853,  2.0238,  1.8357,  1.9732,
        -1.9338,  1.7161, -1.8141,  1.8102,  1.9129, -1.9636, -1.8627, -2.0841,
        -1.9451,  2.1127,  1.9506, -1.9115, -1.9975,  1.8328, -1.9276,  1.8145,
         1.8612, -1.9730, -1.8781, -2.0463, -2.0431, -1.8957,  2.0560, -1.9589,
         1.9910, -1.9894, -1.9883,  1.9705,  2.1232, -1.8964,  2.0687, -1.9124,
        -2.0765, -2.0154, -1.9017,  1.9379, -1.8391, -1.7962,  1.8643, -2.1304,
         2.0203, -1.9002, -1.9929, -1.7937, -2.1168, -1.9915, -2.1617,  2.0831,
         1.8973, -1.8739,  2.0734, -1.9664,  2.0671, -2.0360, -1.8567, -2.0302,
         2.0490,  2.1080,  1.8978,  1.9349,  1.8636, -2.0290,  2.1018, -2.0714,
        -2.0941,  2.1401, -2.2060, -1.9273, -1.9886, -2.1046,  2.0708,  2.0963,
         2.0461,  2.0138,  1.9355, -2.0217, -2.0755, -1.7533, -1.8077, -1.9667,
        -1.8882, -1.9362, -1.7884, -1.9027, -1.9329,  1.8788,  1.9585,  2.0585,
        -1.9665,  2.2107,  1.8527,  1.9978, -2.1611,  1.9649,  1.9205, -2.1292,
         2.2116, -1.9657, -2.0181,  1.9352,  1.9015, -2.0759,  2.0796,  1.9932,
         1.9088,  1.8844, -1.9818, -1.8876, -1.9613, -1.8494,  2.0448, -2.1558,
         1.8631,  2.0635,  2.0204, -1.9526,  2.0696, -1.9234, -2.0351,  1.8903,
         2.0534,  2.1711,  2.1203, -1.8281,  1.9711, -2.1941,  1.7052, -1.9535,
         2.1072, -1.8857,  2.1531,  2.1027,  2.2091,  2.0738, -1.7053,  1.7520,
        -2.2531, -1.8389, -1.9775,  2.1045,  2.0778, -1.9005,  2.0811,  2.0423,
         1.8045, -1.7767,  1.8902, -1.8023,  1.9896,  1.9193,  1.9347, -2.1473,
        -2.0960, -2.0497,  2.0142, -1.9112, -2.2715, -1.9745, -2.1115,  2.0077,
         1.7614, -2.1420, -2.1256, -1.9323,  1.9965, -2.0067, -1.8878,  2.3232,
        -2.0784,  2.0353,  1.9468,  2.1584, -1.8068,  1.9342,  2.1132,  1.7749,
        -1.9137,  1.9812,  2.0808,  2.1076, -1.7870, -2.0498, -2.0884,  2.1148,
        -2.1031,  1.9256, -2.2197, -2.1070, -2.0373,  1.7537, -2.0765, -2.1322,
         2.0448, -1.9809, -1.9562, -1.7479,  1.8569, -2.1280,  1.9934,  2.0147,
         1.8558, -2.2941,  2.0096,  2.2169, -2.0042, -2.0018,  2.0531,  1.8919,
        -1.9972,  2.1058,  2.0230,  2.0216, -1.7416,  1.7103,  1.8417, -2.0624,
        -1.7907, -2.0301,  1.8671,  2.0546,  1.7059,  2.0554, -2.0767,  2.0119,
         1.9197, -1.8728, -2.0730,  1.6863, -2.2407, -1.9816,  1.9106,  2.0176,
         1.9237, -1.8698,  1.9259, -1.9898, -2.0276,  2.0540, -2.1118,  1.6461,
         1.8071,  2.1866, -2.0374, -2.0764,  2.0428, -1.7201, -1.9062, -2.0994,
        -1.8803, -2.0408,  1.7735, -2.1470,  1.9441, -1.9443, -1.9431,  1.9142,
         1.9544,  2.0281, -2.2262, -2.1445, -2.1160, -1.9699,  1.9443, -2.0275,
         2.0388,  2.0073,  2.0205,  2.1368,  1.9250, -1.8127, -1.9367,  1.9763,
         1.8122,  2.1759,  2.1323, -2.0067,  2.0429, -2.0024,  2.1693, -1.9019,
        -1.9540,  1.9801,  1.8641,  1.8957,  2.1198,  2.0107, -1.9562, -1.9625,
        -1.8497,  1.9198, -2.0610, -2.0884,  2.1322, -1.7403,  2.0746, -2.0364,
        -2.0072,  1.9364, -2.0432, -2.2372, -1.9672,  2.0371,  1.9177,  1.9257,
         2.1320,  1.8570,  2.0669,  2.1879, -1.9264, -2.0495,  1.9646, -2.1143,
        -1.7609,  1.9434,  1.9605, -1.8447,  1.9457,  2.0607, -2.0716, -1.8550])