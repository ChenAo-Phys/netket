from .abstract_machine import AbstractMachine
import torch as _torch
import numpy as _np
import warnings


def _get_number_parameters(m):
    r"""Returns total number of variational parameters in a torch.nn.Module."""
    return sum(map(lambda p: p.numel(), _get_differentiable_parameters(m)))


def _get_differentiable_parameters(m):
    r"""Returns total number of variational parameters in a torch.nn.Module."""
    return filter(lambda p: p.requires_grad, m.parameters())


class Torch(AbstractMachine):
    def __init__(self, module, hilbert):
        self._module = _torch.jit.load(module) if isinstance(module, str) else module
        self._module.double()

        extend(self._module, (hilbert.size,))
        
        self._n_par = _get_number_parameters(self._module)
        self._parameters = list(_get_differentiable_parameters(self._module))
        self.n_visible = hilbert.size
        # TODO check that module has input shape compatible with hilbert size
        super().__init__(hilbert)

        

    @property
    def parameters(self):
        return (
            _torch.cat(
                tuple(p.view(-1) for p in _get_differentiable_parameters(self._module))
            )
            .detach()
            .numpy()
            .astype(_np.complex128)
        )

    def assign_beta(self, beta):
        self._module.beta = beta
        return

    def save(self, filename):
        _torch.save(self._module.state_dict(), filename)
        return

    def load(self, filename):
        self._module.load_state_dict(_torch.load(filename))
        return

    @parameters.setter
    def parameters(self, p):
        if not _np.all(p.imag == 0.0):
            warnings.warn(
                "PyTorch machines have real parameters, imaginary part will be discarded"
            )
        torch_pars = _torch.from_numpy(p.real)
        if torch_pars.numel() != self._n_par:
            raise ValueError(
                "p has wrong shape: {}; expected [{}]".format(
                    torch_pars.size(), self._n_par
                )
            )
        i = 0
        for x in map(
            lambda x: x.view(-1), _get_differentiable_parameters(self._module)
        ):
            x.data.copy_(torch_pars[i : i + len(x)].data)
            i += len(x)

    @property
    def n_par(self):
        r"""Returns the total number of trainable parameters in the machine.
        """
        return self._n_par

    def log_val(self, x, out=None):
        if len(x.shape) == 1:
            x = x[_np.newaxis, :]

        batch_shape = x.shape[:-1]

        with _torch.no_grad():
            t_out = self._module(_torch.from_numpy(x)).numpy().view(_np.complex128)

        if out is None:
            return t_out.reshape(batch_shape)

        _np.copyto(out, t_out.reshape(-1))

        return out

    def der_log(self, x, out=None):
        
        if len(x.shape) == 1:
            x = x[_np.newaxis, :]
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        
        if out is None:
            out = _np.empty([x.shape[0], self._n_par], dtype=_np.complex128)

        x = _torch.tensor(x, dtype=_torch.float64)

        with JacobianMode(self._module):
            m = self._module(x)
            m_sum_real = m[:,0].sum()
            m_sum_imag = m[:,1].sum()
            m_sum_real.backward(retain_graph=True)
            out.real = self._module.jacobian().numpy()
            m_sum_imag.backward()
            out.imag = self._module.jacobian().numpy()
        self._module.zero_grad()

        return out.reshape(
            tuple(list(batch_shape) + list(out.shape[-1:]))
        )

        
    def vector_jacobian_prod(self, x, vec, out=None):

        if out is None:
            out = _np.empty(self._n_par, dtype=_np.complex128)

        def write_to(dst):
            dst = _torch.from_numpy(dst)
            i = 0
            for g in (
                p.grad.flatten() for p in self._module.parameters() if p.requires_grad
            ):
                dst[i : i + g.numel()].copy_(g)
                i += g.numel()

        def zero_grad():
            for g in (p.grad for p in self._module.parameters() if p.requires_grad):
                if g is not None:
                    g.zero_()

        vecj = _torch.empty(x.shape[0], 2, dtype=_torch.float64)

        def get_vec(is_real):
            if is_real:
                vecj[:, 0] = _torch.from_numpy(vec.real)
                vecj[:, 1] = _torch.from_numpy(vec.imag)
            else:
                vecj[:, 0] = _torch.from_numpy(vec.imag)
                vecj[:, 1] = _torch.from_numpy(-vec.real)
            return vecj

        y = self._module(_torch.from_numpy(x))
        zero_grad()
        y.backward(get_vec(True), retain_graph=True)
        write_to(out.real)
        zero_grad()
        y.backward(get_vec(False))
        write_to(out.imag)

        return out

    @property
    def is_holomorphic(self):
        r"""PyTorch models are real-valued only, thus non holomorphic.
        """
        return False

    @property
    def state_dict(self):
        from collections import OrderedDict

        return OrderedDict(
            [(k, v.detach().numpy()) for k, v in self._module.state_dict().items()]
        )


class TorchLogCosh(_torch.nn.Module):
    """
    Log(cosh) activation function for PyTorch modules
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return -input + _torch.nn.functional.softplus(2.0 * input)


class TorchView(_torch.nn.Module):
    """
    Reshaping layer for PyTorch modules
    """

    def __init__(self, shape):
        """
        Init method.
        """
        super().__init__()  # init the base class
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)




import types
from functools import partial

def extend(model, input_shape):
    if not isinstance(model, _torch.nn.Module):
        raise TypeError('model should be a nn.Module')
    if not isinstance(input_shape, tuple):
        raise TypeError('input_shape should be a tuple')

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    weight_input_list = []
    weight_output_list = []
    weight_repeat_list = []
    bias_output_list = []
    bias_repeat_list = []

    x = _torch.zeros((1,) + input_shape, dtype=dtype, device=device)
    with _torch.no_grad():
        for module in model.children():
            y = module(x)
            if sum(p.numel() for p in module.parameters()):
                # for all layers with parameters

                # store parameters and clear bias for future calculation
                if module.weight is not None:
                    initial_weight = module.weight.data.clone()
                if module.bias is not None:
                    initial_bias = module.bias.data.clone()
                    module.bias.data = _torch.zeros_like(module.bias)

                if module.weight is not None:
                    Nweight = module.weight.numel()
                    weight_input = []
                    weight_output = []
                    weight_repeat = _torch.zeros(Nweight, dtype=_torch.long, device=device)
                    Xeye = _torch.eye(x.numel(), dtype=dtype, device=device).reshape((-1,)+x.shape[1:])
                    for i in range(Nweight):
                        weight = _torch.zeros(Nweight, dtype=dtype, device=device)
                        weight[i] = 1.
                        module.weight.data = weight.reshape(module.weight.shape)
                        # output of module is of dimension (j,k)
                        out = module(Xeye).reshape(x.numel(), y.numel())
                        if (out[out.abs()>1e-5] - 1.).abs().max() > 1e-5:
                            raise RuntimeError('the network is not written in the standard form, see https://github.com/ChenAo-Phys/pytorch-Jacobian')
                        nonzero = _torch.nonzero(out > 0.5, as_tuple=False).type(_torch.int16)
                        weight_input.append(nonzero[:,0])
                        weight_output.append(nonzero[:,1])
                        weight_repeat[i] = nonzero.shape[0]
                    weight_input_list.append(_torch.cat(weight_input, dim=0))
                    weight_output_list.append(_torch.cat(weight_output, dim=0))
                    weight_repeat_list.append(weight_repeat)
                    module.weight.data = initial_weight
                else:
                    weight_input_list.append(None)
                    weight_output_list.append(None)
                    weight_repeat_list.append(None)
                
                if module.bias is not None:
                    Nbias = module.bias.numel()
                    bias_output = []
                    bias_repeat = _torch.zeros(Nbias, dtype=_torch.long, device=device)
                    for i in range(Nbias):
                        bias = _torch.zeros(Nbias, dtype=dtype, device=device)
                        bias[i] = 1.
                        module.bias.data = bias.reshape(module.bias.shape)
                        out = module(x).reshape(-1)
                        if (out[out.abs()>1e-5] - 1.).abs().max() > 1e-5:
                            raise RuntimeError('the network is not written in the standard form, see https://github.com/ChenAo-Phys/pytorch-Jacobian')
                        nonzero = _torch.nonzero(out > 0.5, as_tuple=False).type(_torch.int16)
                        bias_output.append(nonzero[:,0])
                        bias_repeat[i] = nonzero.shape[0]
                    bias_output_list.append(_torch.cat(bias_output, dim=0))
                    bias_repeat_list.append(bias_repeat)
                    module.bias.data = initial_bias
                else:
                    bias_output_list.append(None)
                    bias_repeat_list.append(None)
                    
            x = _torch.zeros_like(y)
        
    if not hasattr(model, '_Jacobian_shape_dict'):
        model._Jacobian_shape_dict = {}
    model._Jacobian_shape_dict[input_shape] = (
        weight_input_list, weight_output_list, weight_repeat_list, bias_output_list, bias_repeat_list)


    # assign jacobian method to model
    def jacobian(self, as_tuple=False):
        shape = self.input_shape
        if hasattr(self, '_Jacobian_shape_dict') and shape in self._Jacobian_shape_dict:
            weight_input_list, weight_output_list, weight_repeat_list, bias_output_list, bias_repeat_list \
            = self._Jacobian_shape_dict[shape]
        else:
            raise RuntimeError('model or specific input shape is not extended for jacobian calculation')

        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        jac = []
        layer = 0
        for module in self.children():
            if sum(p.numel() for p in module.parameters()):
                weight_input = weight_input_list[layer]
                weight_output = weight_output_list[layer]
                weight_repeat = weight_repeat_list[layer]
                bias_output = bias_output_list[layer]
                bias_repeat = bias_repeat_list[layer]
                x = self.x_in[layer]
                N = x.shape[0]
                dz_dy = self.gradient[layer].reshape(N,-1)

                if weight_repeat is not None:
                    Nweight = weight_repeat.shape[0]
                    dz_dy_select = dz_dy[:, weight_output.long()]
                    x_select = x.reshape(N,-1)[:, weight_input.long()]
                    repeat = _torch.repeat_interleave(weight_repeat)
                    dz_dW = _torch.zeros(N,Nweight, dtype=dtype, device=device).index_add_(1, repeat, dz_dy_select * x_select)
                    if as_tuple:
                        dz_dW = dz_dW.reshape((N,) + module.weight.shape)
                    jac.append(dz_dW)
                if bias_repeat is not None:
                    Nbias = bias_repeat.shape[0]
                    dz_dy_select = dz_dy[:, bias_output.long()]
                    repeat = _torch.repeat_interleave(bias_repeat)
                    dz_db = _torch.zeros(N,Nbias, dtype=dtype, device=device).index_add_(1, repeat, dz_dy_select)
                    if as_tuple:
                        dz_db = dz_db.reshape((N,) + module.bias.shape)
                    jac.append(dz_db)
                layer += 1

        if as_tuple:
            return tuple(jac)
        else:
            return _torch.cat(jac, dim=1)
    
    if not hasattr(model, 'jacobian'):
        model.jacobian = types.MethodType(jacobian, model)


    
class JacobianMode():
    
    def __init__(self, model):
        self.model = model
        if not isinstance(model, _torch.nn.Module):
            raise TypeError('model should be a nn.Module')


    def __enter__(self):
        model = self.model
        model.x_in = []
        model.gradient = []
        self.forward_pre_hook = []
        self.backward_hook = []
        
        def record_input_shape(self, input):
            model.input_shape = input[0].shape[1:]

        def record_forward(self, input, layer):
            model.x_in[layer] = input[0].detach()

        def record_backward(self, grad_input, grad_output, layer):
            model.gradient[layer] = grad_output[0]

        module0 = next(model.children())
        self.first_forward_hook = module0.register_forward_pre_hook(record_input_shape)

        layer = 0
        for module in model.children():
            if sum(p.numel() for p in module.parameters()):
                model.x_in.append(None)
                model.gradient.append(None)
                self.forward_pre_hook.append(module.register_forward_pre_hook(partial(record_forward, layer=layer)))
                self.backward_hook.append(module.register_backward_hook(partial(record_backward, layer=layer)))
                layer += 1


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.first_forward_hook.remove()
        for hook in self.forward_pre_hook:
            hook.remove()
        for hook in self.backward_hook:
            hook.remove()
        
        del self.model.input_shape
        del self.model.x_in
        del self.model.gradient
