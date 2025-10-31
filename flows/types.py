from  enum import Enum

class nnType(Enum):
    """
    Enumeration class representing different types of neural networks.
    """

    resnet = "Residual NN"
    recnet = "Recurrent NN"
    ode = "ODE"
    rnvp = "RNVP"

class activationType(Enum):
    """
    Enum class representing different activation types.

    Attributes:
        sigmoid (str): Sigmoid activation type.
        relu (str): Relu activation type.
        lipswish (str): Lipswish activation type.
    """
    sigmoid = "Sigmoid"
    relu = "Relu"
    lipswish = "Lipswish"
    
class evaluationMode(Enum):
    """
    Enum class representing different activation types. modes of evaluation for
    an invertible model.

    Attributes:
        direct (str): Represents the direct mode of evaluation.
        inverse (str): Represents the inverse mode of evaluation.
    """
    direct = "Direct"
    inverse = "Inverse"

class svdType(Enum):
    fourier = "Fourier"
    direct  = "Direct"
    direct_indiv = "Direct_indiv"
    flax = "Flax"  ## To be added.
