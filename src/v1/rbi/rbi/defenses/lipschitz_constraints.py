
from rbi.defenses.base import ArchitecturalRegularizer
from rbi.utils.lipschitz_tools import lipschitz_neural_net
from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel

from torch.nn import Module

from typing import Callable

class LipschitzNeuralNet(ArchitecturalRegularizer):

    def __init__(self, model: Module, loss_fn: Callable, L: float = 2., ord1: float= 2., ord2: float = 2., **kwargs) -> None:
        """The imposes a bounded Lipschitz constant constraint to the neural netork.

        Args:
            model (Module): Model
            loss_fn (Callable): Loss fn
            L (float, optional): Lipschitz bound. Defaults to 2..
            ord1 (float, optional): Input order of assumed metric. Defaults to 2..
            ord2 (float, optional): Output order of assumed metric. Defaults to 2..
        """
        super().__init__(model, loss_fn)
        self.L = L
        self.ord1 = ord1 
        self.ord2 = ord2
        self._kwargs = kwargs

    def _add_constraints_to_model(self):
        if isinstance(self.model, ParametricProbabilisticModel):
            lipschitz_neural_net(self.model.net, self.L, self.ord1, self.ord2, **self._kwargs)
        else:
            pass
class LipschitzEmbeddingNet(ArchitecturalRegularizer):

    def __init__(self, model: Module, loss_fn: Callable, L: float = 2., ord1: float= 2., ord2: float = 2., **kwargs) -> None:
        """The imposes a bounded Lipschitz constant constraint to the embedding net.

        Args:
            model (Module): Model
            loss_fn (Callable): Loss fn
            L (float, optional): Lipschitz bound. Defaults to 2..
            ord1 (float, optional): Input order of assumed metric. Defaults to 2..
            ord2 (float, optional): Output order of assumed metric. Defaults to 2..
        """
        super().__init__(model, loss_fn)
        self.L = L
        self.ord1 = ord1 
        self.ord2 = ord2
        self._kwargs = kwargs

    def _add_constraints_to_model(self):
        lipschitz_neural_net(self.model.embedding_net, self.L, self.ord1, self.ord2, **self._kwargs) # type: ignore
