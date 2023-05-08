
import torch 

from rbi.utils.transforms import sampling_transform_jacobian
from rbi.models import MixtureDiagGaussianModel


def test_get_transform_jacobian(continuous_model):
    if isinstance(continuous_model , MixtureDiagGaussianModel):
        # Skipped for now...
        return 
    input_dim = continuous_model.input_dim

    x = torch.randn((10, input_dim))
    matrix = sampling_transform_jacobian(continuous_model, x)

    assert matrix.shape[0] == 10, "The batch shape is wrong..."
    assert matrix.shape[1] == input_dim and matrix.shape[2] == input_dim, "The jacobian shape is wrong ..."
    assert torch.isclose(matrix, torch.transpose(matrix, -2, -1), atol=1e-2).all(), "The output must be symmetric"
    assert (torch.linalg.eigvalsh(matrix) > -1e-3).all(), "The output must be p.s.d"


