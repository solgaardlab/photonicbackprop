from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, Activation
import numpy as np

from neurophox.numpy.generic import MeshPhases
from neurophox.meshmodel import MeshModel
from neurophox.helpers import pairwise_off_diag_permutation, plot_complex_matrix, inverse_permutation
from neurophox.config import TF_COMPLEX, BLOCH, SINGLEMODE
from neurophox.generic import *

class Mesh_taps:
    def __init__(self, model: MeshModel):
        """General mesh network layer defined by `neurophox.meshmodel.MeshModel`

        Args:
            model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        """
        self.model = model
        self.units, self.num_layers = self.model.units, self.model.num_layers
        self.pairwise_perm_idx = pairwise_off_diag_permutation(self.units)
        enn, enp, epn, epp = self.model.mzi_error_tensors
        self.enn, self.enp, self.epn, self.epp = tf.constant(enn, dtype=TF_COMPLEX), tf.constant(enp, dtype=TF_COMPLEX),\
                                                 tf.constant(epn, dtype=TF_COMPLEX), tf.constant(epp, dtype=TF_COMPLEX)
        self.perm_layers = []
        for layer in range(self.num_layers):
            self.perm_layers.append(PermutationLayer(self.model.perm_idx[layer]))
            self.perm_layers.append(PermutationLayer(np.arange(self.units))) ## no permutation within one MZI
        self.perm_layers.append(PermutationLayer(self.model.perm_idx[self.num_layers])) ## output permutation
        # self.perm_layers = [PermutationLayer(self.model.perm_idx[layer]) for layer in range(self.num_layers + 1)]

    def mesh_layers(self, phases: MeshPhasesTensorflow) -> List[MeshVerticalLayer]:
        """

        Args:
            phases:  The :code:`MeshPhasesTensorflow` object containing :math:`\\boldsymbol{\\theta}, \\boldsymbol{\\phi}, \\boldsymbol{\\gamma}`

        Returns:
            List of mesh layers to be used by any instance of :code:`MeshLayer`
        """
        internal_psl = phases.internal_phase_shift_layers
        external_psl = phases.external_phase_shift_layers
        # smooth trick to efficiently perform the layerwise coupling computation
        
        # pdb.set_trace()
        mask = tf.transpose(tf.convert_to_tensor(np.repeat(self.model.mask, 2, axis = 1).astype(np.complex64)))
        if self.model.hadamard:
            raise NotImplementedError()
        else:
            # diag_layers = external_psl * (s11 + s22) / 2
            # off_diag_layers = roll_tensor(external_psl) * (s21 + s12) / 2
            int_diag_layers = internal_psl * (mask*(1/np.sqrt(2)-1) + 1)
            int_off_diag_layers = 1j * roll_tensor(internal_psl) * mask/np.sqrt(2)
            ext_diag_layers = external_psl * (mask*(1/np.sqrt(2)-1) + 1)
            ext_off_diag_layers = 1j * roll_tensor(external_psl) * mask/np.sqrt(2)                     
        
        # pdb.set_trace()
        if self.units % 2:
            # diag_layers = tf.concat((diag_layers[:-1], tf.ones_like(diag_layers[-1:])), axis=0)
            raise NotImplementedError()
        
        int_diag_layers, int_off_diag_layers = tf.transpose(int_diag_layers), tf.transpose(int_off_diag_layers)
        ext_diag_layers, ext_off_diag_layers = tf.transpose(ext_diag_layers), tf.transpose(ext_off_diag_layers)
        
        mesh_layers = [MeshVerticalLayer(self.pairwise_perm_idx, int_diag_layers[0], int_off_diag_layers[0],
                                         self.perm_layers[1], self.perm_layers[0])]
        mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, ext_diag_layers[0], ext_off_diag_layers[0],
                                         self.perm_layers[2]))
        for layer in range(1, self.num_layers):
            mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, int_diag_layers[layer], int_off_diag_layers[layer],
                                                 self.perm_layers[layer*2 + 1]))
            mesh_layers.append(MeshVerticalLayer(self.pairwise_perm_idx, ext_diag_layers[layer], ext_off_diag_layers[layer],
                                                 self.perm_layers[layer*2 + 2]))
        return mesh_layers


class MeshLayer_taps(TransformerLayer):
    """Mesh network layer for unitary operators implemented in numpy

    Args:
        mesh_model: The `MeshModel` model of the mesh network (e.g., rectangular, triangular, custom, etc.)
        activation: Nonlinear activation function (:code:`None` if there's no nonlinearity)
    """

    def __init__(self, mesh_model: MeshModel, activation: Activation = None,
                 include_diagonal_phases: bool = True, **kwargs):
        self.mesh = Mesh_taps(mesh_model)
        self.units, self.num_layers = self.mesh.units, self.mesh.num_layers
        self.include_diagonal_phases = include_diagonal_phases
        super(MeshLayer_taps, self).__init__(self.units, activation=activation, **kwargs)
        theta_init, phi_init, gamma_init = self.mesh.model.init()
        self.theta, self.phi, self.gamma = theta_init.to_tf("theta"), phi_init.to_tf("phi"), gamma_init.to_tf("gamma")
        # self.theta = tf.ones([1,1])*np.pi/3
        # self.phi = tf.ones([1,1])*np.pi/6
        # self.gamma = tf.ones([1,1])*0.0
        
    @tf.function
    def transform(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the operation (where :math:`U` represents the matrix for this layer):

        .. math::
            V_{\mathrm{out}} = V_{\mathrm{in}} U,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            inputs: :code:`inputs` batch represented by the matrix :math:`V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Transformed :code:`inputs`, :math:`V_{\mathrm{out}}`
        """
        mesh_phases, mesh_layers = self.phases_and_layers
        outputs = inputs * mesh_phases.input_phase_shift_layer if self.include_diagonal_phases else inputs
        fields = tf.expand_dims(outputs, axis=2)
        # pdb.set_trace()
        for layer in range(len(mesh_layers)):
            field, outputs = mesh_layers[layer].transform(outputs)
            fields = tf.concat([fields, tf.expand_dims(field, axis=2)], axis = 2)
        # return fields[...,1:]
        return fields
    
    @tf.function
    def inverse_transform(self, outputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the operation (where :math:`U` represents the matrix for this layer):

        .. math::
            V_{\mathrm{in}} = V_{\mathrm{out}} U^\dagger,

        where :math:`U \in \mathrm{U}(N)` and :math:`V_{\mathrm{out}}, V_{\mathrm{in}} \in \mathbb{C}^{M \\times N}`.

        Args:
            outputs: :code:`outputs` batch represented by the matrix :math:`V_{\mathrm{out}} \in \mathbb{C}^{M \\times N}`

        Returns:
            Inverse transformed :code:`outputs`, :math:`V_{\mathrm{in}}`
        """
        mesh_phases, mesh_layers = self.phases_and_layers
        inputs = outputs
        fields = tf.expand_dims(inputs, axis=2)
        for layer in reversed(range(len(mesh_layers))):
            field, inputs = mesh_layers[layer].inverse_transform(inputs)
            fields = tf.concat([fields, tf.expand_dims(field, axis=2)], axis = 2)
        if self.include_diagonal_phases:
            inputs = inputs * tf.math.conj(mesh_phases.input_phase_shift_layer)
        fields = tf.concat([fields, tf.expand_dims(inputs, axis=2)], axis = 2)
        # return fields[...,1:]
        return fields[...,1:]

    @property
    def phases_and_layers(self) -> Tuple[MeshPhasesTensorflow, List[MeshVerticalLayer]]:
        """

        Returns:
            Phases and layers for this mesh layer
        """
        mesh_phases = MeshPhasesTensorflow(
            theta=self.theta,
            phi=self.phi,
            mask=self.mesh.model.mask,
            gamma=self.gamma,
            hadamard=self.mesh.model.hadamard,
            units=self.units,
            basis=self.mesh.model.basis
        )
        mesh_layers = self.mesh.mesh_layers(mesh_phases)
        return mesh_phases, mesh_layers

    @property
    def phases(self) -> MeshPhases:
        """

        Returns:
            The :code:`MeshPhases` object for this layer
        """
        return MeshPhases(
            theta=self.theta.numpy() * self.mesh.model.mask,
            phi=self.phi.numpy() * self.mesh.model.mask,
            mask=self.mesh.model.mask,
            gamma=self.gamma.numpy()
        )


def roll_tensor(tensor: tf.Tensor, up=False):
    # a complex number-friendly roll that works on gpu
    if up:
        return tf.concat([tensor[1:], tensor[tf.newaxis, 0]], axis=0)
    return tf.concat([tensor[tf.newaxis, -1], tensor[:-1]], axis=0)