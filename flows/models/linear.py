from typing import List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import numpy as np
# from ..types import evaluationMode # Assuming evaluationMode is an Enum with members 'direct' and 'inverse'

config.update("jax_enable_x64", True)


class Linear(nn.Module):
    """
    Implements an element-wise affine transformation, $y = a \odot x + b$, 
    which is an **invertible** operation.

    This module is often used as a simple scaling and shifting layer in deep learning 
    or as a coupling layer component in Normalizing Flows.

    The parameters $a$ (scale) and $b$ (shift) can optionally be made **learnable** Flax parameters using `opt_a` and `opt_b`.

    Attributes:
        a (Union[List[float], NDArray]): The scaling factor(s). Must have a shape 
                                         compatible with the input $x$ for element-wise multiplication.
        b (Union[List[float], NDArray]): The shifting factor(s). Must have a shape 
                                         compatible with the input $x$ for element-wise addition.
        opt_a (Optional[bool]): If True, 'a' is registered as a learnable Flax parameter. 
                                Defaults to True.
        opt_b (Optional[bool]): If True, 'b' is registered as a learnable Flax parameter. 
                                Defaults to True.
    """

    a: Union[List[float], NDArray[np.float64]]
    b: Union[List[float], NDArray[np.float64]]
    opt_a: Optional[bool] = True
    opt_b: Optional[bool] = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, mode='direct') -> jnp.ndarray:
        """
        Applies the affine transformation or its inverse.

        Args:
            x (jnp.ndarray): The input data.
            mode (evaluationMode, optional): The evaluation mode. Must be 'direct' 
                                            (forward) or 'inverse' (backward). Defaults to 'direct'.

        Returns:
            jnp.ndarray: The output of the transformation.
        """
        # --- Parameter Initialization/Loading ---
        if self.opt_a:
            # Register 'a' as a learnable parameter
            a = self.param("linear_a", lambda *_: jnp.asarray(self.a), jnp.shape(self.a))
        else:
            # Treat 'a' as a static constant
            a = jnp.asarray(self.a)
            
        if self.opt_b:
            # Register 'b' as a learnable parameter
            b = self.param("linear_b", lambda *_: jnp.asarray(self.b), jnp.shape(self.b))
        else:
            # Treat 'b' as a static constant
            b = jnp.asarray(self.b)
        
        # --- Transformation Logic ---
        if mode == 'inverse':
            # Inverse: $x = (y - b) / a$
            return (x - b) / a
        else:
            # Direct: $y = x \odot a + b$
            return x * a + b