from typing import Optional, List, Dict

import tensorflow as tf
from tensorflow.keras.layers import Activation
import numpy as np

from generic_taps import MeshLayerTaps
from neurophox.meshmodel import RectangularMeshModel, TriangularMeshModel, PermutingRectangularMeshModel, ButterflyMeshModel
from neurophox.helpers import rectangular_permutation, butterfly_layer_permutation
from neurophox.config import DEFAULT_BASIS, TF_FLOAT, TF_COMPLEX, SINGLEMODE
        
class TMTaps(MeshLayerTaps):
    """Triangular mesh network layer for unitary operators implemented in tensorflow

    Args:
        units: The dimension of the unitary matrix (:math:`N`)
        hadamard: Hadamard convention for the beamsplitters
        basis: Phase basis to use
        bs_error: Beamsplitter split ratio error
        theta_init_name: Initializer name for :code:`theta` (:math:`\\boldsymbol{\\theta}` or :math:`\\theta_{n\ell}`)
        phi_init_name: Initializer name for :code:`phi` (:math:`\\boldsymbol{\\phi}` or :math:`\\phi_{n\ell}`)
        gamma_init_name: Initializer name for :code:`gamma` (:math:`\\boldsymbol{\\gamma}` or :math:`\\gamma_{n}`)
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, units: int, hadamard: bool = False, basis: str = SINGLEMODE,
                 bs_error: float = 0.0, theta_init_name: Optional[str] = "haar_tri",
                 phi_init_name: Optional[str] = "random_phi", gamma_init_name: Optional[str] = "random_gamma",
                 activation: Activation = None, **kwargs):
        super(TMTaps, self).__init__(
            TriangularMeshModel(units, hadamard, bs_error, basis,
                                theta_init_name, phi_init_name, gamma_init_name),
            activation, **kwargs
        )