
from .flows import InverseAffineAutoregressiveModel
from .base import EnsembleModel

class InverseAffineAutoregressiveEnsemble(EnsembleModel):
    def __init__(self, *args, num_models=10, **kwargs):
        models = [InverseAffineAutoregressiveModel(*args,**kwargs) for n in range(num_models)]
        super().__init__(models)