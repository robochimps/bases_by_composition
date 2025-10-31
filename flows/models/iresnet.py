from typing import List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import flows 
from .MLP import NormalizedMultiLayerPerceptron
config.update("jax_enable_x64", True)
from ..types import *

class Inverse(nn.Module):
    @nn.compact
    def __call__(self, x, x0, dense):
        x = x0 - dense(x)
        return x, x
    
class InvertibleResNet(nn.Module):

    archi: List[List[int]]
    no_inv_iters: Optional[int] = 100
    lip: Optional[float] = .9
    activation: Optional[activationType] = activationType.lipswish
    svd: Optional[svdType] = svdType.direct_indiv

    def setup(self):
        self.NNs = [NormalizedMultiLayerPerceptron(arch, activation=self.activation, svd=self.svd, lip=self.lip) for arch in self.archi]
        
    @nn.compact
    def __call__(self, x, mode=evaluationMode.direct):
        if mode == evaluationMode.direct:
            for block in self.NNs:
                x = block(x) + x
            return x    
        elif mode == evaluationMode.inverse:
            for block in reversed(self.NNs):
                x0 = x
                units = nn.scan(Inverse, variable_broadcast="params",
                            split_rngs={"params": True}, in_axes=0)
                x, _ = units()(x, jnp.array([x0]*self.no_inv_iters), block)
        return x


