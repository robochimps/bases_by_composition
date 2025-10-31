from typing import List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import numpy as np
from ..types import evaluationMode

config.update("jax_enable_x64", True)


class Linear(nn.Module):
    """Linear transformation a * x + b"""

    a: Union[List[float], NDArray[np.float_]]
    b: Union[List[float], NDArray[np.float_]]
    opt_a: Optional[bool] = True
    opt_b: Optional[bool] = True

    @nn.compact
    def __call__(self, x, mode=evaluationMode.direct):
        if self.opt_a:
            a = self.param("linear_a", lambda *_: jnp.asarray(self.a), jnp.shape(self.a))
        else:
            a = jnp.asarray(self.a)
            
        if self.opt_b:
            b = self.param("linear_b", lambda *_: jnp.asarray(self.b), jnp.shape(self.b))
        else:
            b = jnp.asarray(self.b)
            
        if mode==evaluationMode.inverse:
            return (x - b) / a
        else:
            return x * a + b

