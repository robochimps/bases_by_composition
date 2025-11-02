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
    """
    A small helper module designed for use within `jax.nn.scan` to perform 
    a single iteration of the fixed-point iteration (e.g., Picard iteration) 
    required for numerical inversion of a ResNet block:

    $$x_{k+1} = x_0 - f(x_k)$$
    
    where $x_0$ is the output of the ResNet block, and $f$ is the non-linear transformation 
    (the dense network).

    This module is stateless and intended for execution on a sequence of inputs 
    to drive the iteration towards convergence.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray, x0: jnp.ndarray, dense: nn.Module) -> jnp.ndarray:
        """
        Performs one iteration of the inverse fixed-point calculation.

        Args:
            x (jnp.ndarray): The current estimate of the inverse solution ($x_k$).
            x0 (jnp.ndarray): The target value, which is the output of the direct block ($y$).
            dense (nn.Module): The non-linear transformation $f$ (the ResNet block's inner network).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The updated estimate $x_{k+1}$ (as $x$) and a duplicate 
                                            of the updated estimate (as $x$), which is required 
                                            by `jax.lax.scan` for passing intermediate results.
        """
        # Fixed-point iteration: $x_{k+1} = x_0 - f(x_k)$
        x = x0 - dense(x)
        return x, x

# ... (imports)

class InvertibleResNet(nn.Module):
    """
    Implements a sequence of Invertible Residual Network (ResNet) blocks, 
    often used as a building block in Normalizing Flows (e.g., Residual Flows).

    The direct (forward) pass is an additive residual map: $y = x + f(x)$.
    The inverse pass is computed numerically via fixed-point iteration 
    (Picard iteration) because the inverse function, $x = y - f(x)$, is implicit.

    Attributes:
        archi (List[List[int]]): A list where each inner list defines the 
                                 architecture (layer sizes) of a single 
                                 NormalizedMultiLayerPerceptron (MLP) block.
        no_inv_iters (Optional[int]): The number of fixed-point iterations to 
                                      perform for the inverse calculation. Defaults to 100.
        lip (Optional[float]): The Lipschitz constant upper bound for the MLP 
                               blocks $f(x)$ (to ensure convergence of the inverse 
                               via Banach's fixed-point theorem). Defaults to 0.9.
        activation (Optional[activationType]): The activation function type used in the MLPs. 
                                               Defaults to `activationType.lipswish`.
        svd (Optional[svdType]): The SVD type used for spectral normalization of the MLPs. 
                                 Defaults to `svdType.direct_indiv`.
    """

    archi: List[List[int]]
    no_inv_iters: Optional[int] = 100
    lip: Optional[float] = .9
    activation: Optional[activationType] = activationType.lipswish
    svd: Optional[svdType] = svdType.direct_indiv

    def setup(self):
        """
        Initializes the list of NormalizedMultiLayerPerceptron blocks (the NNs).
        """
        self.NNs = [NormalizedMultiLayerPerceptron(arch, activation=self.activation, svd=self.svd, lip=self.lip) 
                    for arch in self.archi]
        
    @nn.compact
    def __call__(self, x: jnp.ndarray, mode='direct') -> jnp.ndarray:
        """
        Applies the sequence of ResNet blocks in the direct or inverse mode.

        Args:
            x (jnp.ndarray): The input data.
            mode (evaluationMode, optional): The evaluation mode. Must be 
                                            'direct' or 'inverse'. Defaults to 'direct'.

        Returns:
            jnp.ndarray: The output of the composite transformation.
        """
        if mode == evaluationMode.direct:
            # Direct pass: $y = x + f(x)$
            for block in self.NNs:
                x = block(x) + x
            return x    
            
        elif mode == evaluationMode.inverse:
            # Inverse pass: Numerical calculation of $x = y - f(x)$
            # The blocks are inverted in reverse order.
            for block in reversed(self.NNs):
                x0 = x # $y$ becomes the fixed target for this block's inversion
                
                # JAX scan is used to perform the fixed-point iteration across 'no_inv_iters' steps.
                # The 'x' input to scan is the initial guess ($x_0$), which is $y$.
                units = nn.scan(Inverse, variable_broadcast="params",
                            split_rngs={"params": True}, in_axes=0)
                
                # The scan operation iteratively computes $x_{k+1} = x_0 - f(x_k)$.
                # The sequence of $x_0$ values (the fixed target $y$) is broadcasted.
                # The final converged value is $x$.
                x, _ = units()(x, jnp.array([x0]*self.no_inv_iters), block)
                
        return x
