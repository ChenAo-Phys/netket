# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

from .abstract_density_matrix import AbstractDensityMatrix
from ..jax import Jax as JaxPure
from functools import partial


class Jax(JaxPure, AbstractDensityMatrix):
    def __init__(self, hilbert, module, dtype=complex):
        """
        Wraps a stax network (which is a tuple of `init_fn` and `predict_fn`)
        so that it can be used as a NetKet density matrix.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            module: A pair `(init_fn, predict_fn)`. See the documentation of
                `jax.experimental.stax` for more info.
            dtype: either complex or float, is the type used for the weights.
                In both cases the module must have a single output.
        """
        JaxPure.__init__(self, hilbert, module, dtype)
        AbstractDensityMatrix.__init__(self, hilbert)
        assert self.input_size == self.hilbert.size * 2

    @staticmethod
    @jax.jit
    def _dminput(xr, xc):
        if xc is None:
            x = xr
        else:
            x = jnp.hstack((xr, xc))
        return x

    def log_val(self, xr, xc=None, out=None):
        x = self._dminput(xr, xc)

        if out is None:
            out = self._forward_fn(self._params, x).reshape(x.shape[0],)
        else:
            out[:] = self._forward_fn(self._params, x).reshape(x.shape[0],)

        return out

    def der_log(self, xr, xc=None, out=None):
        x = self._dminput(xr, xc)

        out = self._perex_grads(self._params, x)

        return out

    def diagonal(self):
        from .diagonal import Diagonal

        diag = Diagonal(self)

        def diag_jax_forward(params, x):
            return self.jax_forward(params, self._dminput(x, x))

        diag.jax_forward = diag_jax_forward

        return diag


from jax.experimental import stax
from jax.experimental.stax import Dense
from jax.nn.initializers import glorot_normal, normal


def DensePurificationComplex(
    out_pure, out_mix, use_hidden_bias=True, W_init=glorot_normal(), b_init=normal()
):
    """Layer constructor function for a complex purification layer."""

    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        output_shape = input_shape[:-1] + (2 * out_pure + out_mix,)

        k = jax.random.split(rng, 7)

        input_size = input_shape[-1] // 2

        # Weights for the pure part
        Wr, Wi = (
            W_init(k[0], (input_size, out_pure)),
            W_init(k[1], (input_size, out_pure)),
        )

        # Weights for the mixing part
        Vr, Vi = (
            W_init(k[2], (input_size, out_mix)),
            W_init(k[3], (input_size, out_mix)),
        )

        if use_hidden_bias:
            br, bi = (b_init(k[4], (out_pure,)), b_init(k[5], (out_pure,)))
            cr = b_init(k[6], (out_mix,))

            return output_shape, (Wr, Wi, Vr, Vi, br, bi, cr)
        else:
            return output_shape, (Wr, Wi, Vr, Vi)

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        if use_hidden_bias:
            Wr, Wi, Vr, Vi, br, bi, cr = params
        else:
            Wr, Wi, Vr, Vi = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        thetar = jax.numpy.dot(xr[:,], (Wr + 1.0j * Wi))
        thetac = jax.numpy.dot(xc[:,], (Wr - 1.0j * Wi))

        thetam = jax.numpy.dot(xr[:,], (Vr + 1.0j * Vi))
        thetam += jax.numpy.dot(xc[:,], (Vr - 1.0j * Vi))

        if use_hidden_bias:
            thetar += br + 1.0j * bi
            thetac += br - 1.0j * bi
            thetam += 2 * cr

        return jax.numpy.hstack((thetar, thetam, thetac))

    return init_fun, apply_fun


from ..jax import LogCoshLayer, SumLayer


def JaxNdmSpin(hilbert, alpha, beta, use_hidden_bias=True):
    r"""
    A fully connected Neural Density Matrix (DBM). This type density matrix is
    obtained purifying a RBM with spin 1/2 hidden units.

    The number of purification hidden units can be chosen arbitrarily.

    The weights are taken to be complex-valued. A complete definition of this
    machine can be found in Eq. 2 of Hartmann, M. J. & Carleo, G.,
    Phys. Rev. Lett. 122, 250502 (2019).

    Args:
        hilbert: Hilbert space of the system.
        alpha: `alpha * hilbert.size` is the number of hidden spins used for
                the pure state part of the density-matrix.
        beta: `beta * hilbert.size` is the number of hidden spins used for the purification.
            beta=0 for example corresponds to a pure state.
        use_hidden_bias: If ``True`` bias on the hidden units is taken.
                         Default ``True``.
    """
    return Jax(
        hilbert,
        stax.serial(
            DensePurificationComplex(
                alpha * hilbert.size, beta * hilbert.size, use_hidden_bias
            ),
            LogCoshLayer,
            SumLayer(),
        ),
        dtype=float,
    )
