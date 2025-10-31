from typing import List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import numpy as np
from ..types import evaluationMode

config.update("jax_enable_x64", True)

class CompositeModel(nn.Module):

    models: List[nn.Module]

    @nn.compact
    def __call__(self, x, mode=evaluationMode.direct):
        if mode == evaluationMode.direct:
            for model in self.models:
                x = model(x)
        elif mode == evaluationMode.inverse:
            for model in reversed(self.models):
                x = model(x, mode=evaluationMode.inverse)
        else:
            raise ValueError("Invalid mode")
        return x

  