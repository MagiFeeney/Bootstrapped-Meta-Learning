"""
modified from torch.optim._functional.py
"""
from typing import Union

import torch.distributed
from torch import nn
from torch import distributed
from torch.nn.modules.batchnorm import _NormBase
from torch.optim._functional import *


def cosine_similarity(net: nn.Module, inner_prod: float = 0., norm_0: float = 0., norm_1: float = 0.):
    with torch.no_grad():
        for parameter_name, parameter in net._parameters.items():
            if parameter is not None:
                cpu_parameter = parameter.cpu()
                if distributed.get_rank() == 0:
                    remote_parameter = torch.empty_like(cpu_parameter)
                    distributed.recv(remote_parameter)
                    inner_prod += float((cpu_parameter * remote_parameter).sum())
                    norm_0 += float((cpu_parameter ** 2).sum())
                    norm_1 += float((remote_parameter ** 2).sum())
                if distributed.get_rank() == 1:
                    distributed.send(cpu_parameter, 0)
    for module_name, module in net._modules.items():
        inner_prod, norm_0, norm_1 = cosine_similarity(module, inner_prod, norm_0, norm_1)
    return inner_prod, norm_0, norm_1


def replace_nan_hook(grad: torch.Tensor):
    return torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)


def build_parameter_list_no_overlap(net: nn.Module,
                                    existing_set: set,
                                    insert_point: List[str],
                                    net_name: str):
    for parameter_name, parameters in net._parameters.items():
        if parameters not in existing_set:
            insert_point.append(net_name + "." + parameter_name)
            existing_set.add(parameters)

    for module_name, module in net._modules.items():
        build_parameter_list_no_overlap(module, existing_set, insert_point, net_name + '.' + module_name)


def clip_grad_norm_(
        grads: List[torch.Tensor], max_norm: Union[float, torch.Tensor], norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    if norm_type == math.inf:
        # norms = [p.abs().max() for p in grads if p is not None]
        norms = [p.detach().abs_().max() for p in grads if p is not None]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p, norm_type) for p in grads if p is not None]), norm_type)
        total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type) for p in grads if p is not None]),
                                norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for i, p in enumerate(grads):
        if p is not None:
            grads[i] = p.mul(clip_coef_clamped)
    return total_norm


def broadcast_net(net: nn.Module):
    for parameter_name, parameter in net._parameters.items():
        if parameter is not None:
            distributed.broadcast(parameter, 0)
    for module_name, module in net._modules.items():
        broadcast_net(module)
    return net


def detach_recursively(net: nn.Module):
    for parameter_name, parameter in net._parameters.items():
        if parameter is not None:
            parameter.detach_().requires_grad_(True)
    for module_name, module in net._modules.items():
        detach_recursively(module)
    return net


def clone_network(net_to: nn.Module, net_from: nn.Module):
    for parameter_name in net_from._parameters:
        net_to._parameters[parameter_name] = net_from._parameters[parameter_name].clone()
    for module_name in net_from._modules:
        net_to._modules[module_name] = clone_network(net_to._modules[module_name], net_from._modules[module_name])
    return net_to


def slide_buffer_window(net: nn.Module):
    if isinstance(net, _NormBase):
        for buffer_name, buffer in net._buffers.items():
            net._buffers[buffer_name] = buffer.detach().clone()


def recover_net_from_plain_list(net: nn.Module, parameter_list: List[nn.Parameter], parameter_idx: int):
    for parameter_name, parameter in net._parameters.items():
        if parameter is not None and not hasattr(parameter, '_recovered'):
            net._parameters[parameter_name] = parameter_list[parameter_idx].detach_().requires_grad_(True)
            net._parameters[parameter_name]._recovered = True
            parameter_idx += 1

    if hasattr(net, 'reset_running_stats'):
        net.reset_running_stats()
    for module_name, module in net._modules.items():
        parameter_idx = recover_net_from_plain_list(module, parameter_list, parameter_idx)
    return parameter_idx


def recover_net_from_plain_list_online(net: nn.Module, parameter_list: List[nn.Parameter], parameter_idx: int = 0):
    for parameter_name, parameter in net._parameters.items():
        if parameter is not None and not hasattr(parameter, '_recovered'):
            net._parameters[parameter_name] = parameter_list[parameter_idx]
            net._parameters[parameter_name]._recovered = True
            parameter_idx += 1

    for module_name, module in net._modules.items():
        parameter_idx = recover_net_from_plain_list_online(module, parameter_list, parameter_idx)
    return parameter_idx


def average_model(net: nn.Module):
    for parameter_name, parameter in net._parameters.items():
        if parameter is not None and not hasattr(parameter, '_recovered'):
            parameter.requires_grad_(False)
            if parameter.device != torch.device("cpu"):
                torch.cuda.synchronize(parameter.device)
            distributed.all_reduce(parameter)
            parameter /= distributed.get_world_size()
            parameter.requires_grad_(True)
            parameter._recovered = True
    for module_name, module in net._modules.items():
        average_model(module)


def reset_window(optimizer):
    for group in optimizer.param_groups:
        detach_recursively(group["params"])
    for state_name in optimizer.state:
        state = optimizer.state[state_name]
        for k in state:
            if isinstance(state[k], torch.Tensor):
                if state[k].grad_fn is not None:
                    state[k].detach_().requires_grad_(True)


def build_parameter_list(insert_point: List[str], net: nn.Module, net_name: str):
    for parameter_name, parameter in net._parameters.items():
        if parameter is not None:
            insert_point.append(net_name + "." + parameter_name)
    for module_name, module in net._modules.items():
        build_parameter_list(insert_point, module, net_name + '.' + module_name)


def meta_sgd(net: nn.Module,
             momentum_buffer_list: List[Optional[Tensor]],
             *,
             weight_decay: float or torch.Tensor,
             momentum: float or torch.Tensor,
             lr: float or torch.Tensor,
             dampening: float or torch.Tensor,
             nesterov: bool,
             ddp_descendent: bool = False,
             i: int = 0):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    ddp_descendent = ddp_descendent | isinstance(net, nn.parallel.DistributedDataParallel)

    for param_name, param in net._parameters.items():
        if param is None or param._fast_grad is None:
            continue
        d_p = param._fast_grad
        param._fast_grad = None
        if weight_decay != 0:
            if isinstance(weight_decay, float):
                d_p = d_p.add(param, alpha=weight_decay)
            else:
                d_p = d_p.add(param.mul(weight_decay))

        if momentum != 0:
            if momentum_buffer_list[i] is None:
                momentum_buffer_list[i] = torch.clone(d_p)
            else:
                if isinstance(dampening, float):
                    momentum_buffer_list[i] = momentum_buffer_list[i].mul(momentum).add(d_p, alpha=1 - dampening)
                else:
                    momentum_buffer_list[i] = momentum_buffer_list[i].mul(momentum).add(d_p.mul(1 - dampening))

            if nesterov:
                if isinstance(momentum, float):
                    d_p = d_p.add(momentum_buffer_list[i], alpha=momentum)
                else:
                    d_p = d_p.add(momentum_buffer_list[i].mul(momentum))
            else:
                d_p = momentum_buffer_list[i]

        if isinstance(lr, float):
            net._parameters[param_name] = param.add(d_p, alpha=-lr)
        else:
            net._parameters[param_name] = param.add(d_p.mul(-lr))
        net._parameters[param_name]._fast_grad = None
        i += 1

    for module_name, module in net._modules.items():
        if ddp_descendent:
            # DDP does not accept inplace buffer
            slide_buffer_window(module)

        i = meta_sgd(module,
                     momentum_buffer_list,
                     weight_decay=weight_decay, momentum=momentum,
                     lr=lr,
                     dampening=dampening,
                     nesterov=nesterov,
                     ddp_descendent=ddp_descendent,
                     i=i)
    return i


def meta_adam(net: nn.Module,
              params_with_grad: List[str],
              state: dict,
              state_steps: List[int],
              *,
              amsgrad: bool,
              beta1: float or Tensor,
              beta2: float or Tensor,
              lr: float or Tensor,
              weight_decay: float or Tensor,
              eps: float or Tensor,
              ddp_descendent: bool = False,
              i: int = 0):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    ddp_descendent = ddp_descendent | isinstance(net, nn.parallel.DistributedDataParallel)

    for state_id, (param_name, param) in zip(params_with_grad[i:], net._parameters.items()):
        if param is None or param._fast_grad is None:
            continue
        state_ = state[state_id]
        grad = param._fast_grad
        param._fast_grad = None
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            if isinstance(weight_decay, float):
                grad = grad.add(param, alpha=weight_decay)
            else:
                grad = grad.add(param.mul(weight_decay))

        # Decay the first and second moment running average coefficient
        if isinstance(beta1, float):
            state_['exp_avg'] = state_['exp_avg'].mul(beta1).add(grad, alpha=1 - beta1)
        else:
            state_['exp_avg'] = state_['exp_avg'].mul(beta1).add(grad.mul(1 - beta1))

        if isinstance(beta2, float):
            state_['exp_avg_sq'] = state_['exp_avg_sq'].mul(beta2).addcmul(grad, grad.conj(), value=1 - beta2)
        else:
            state_['exp_avg_sq'] = state_['exp_avg_sq'].mul(beta2).addcmul(grad.mul(1 - beta2), grad.conj(), value=1.)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(state_['max_exp_avg_sq'], state_['exp_avg_sq'], out=state_['max_exp_avg_sq'])
            # Use the max. for normalizing running avg. of gradient
            denom = (state_['max_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (state_['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        if isinstance(step_size, float):
            net._parameters[param_name] = param.addcdiv(state_['exp_avg'], denom, value=-step_size)
        else:
            net._parameters[param_name] = param.addcdiv(state_['exp_avg'].mul(step_size), denom, value=-1.)
        net._parameters[param_name]._fast_grad = None

        i += 1
    for module_name, module in net._modules.items():
        if ddp_descendent:
            # DDP does not accept inplace buffer
            slide_buffer_window(module)

        i = meta_adam(module,
                      params_with_grad,
                      state,
                      state_steps,
                      amsgrad=amsgrad,
                      beta1=beta1,
                      beta2=beta2,
                      lr=lr,
                      weight_decay=weight_decay,
                      eps=eps,
                      ddp_descendent=ddp_descendent,
                      i=i)
    return i


def meta_rmsprop(net: nn.Module,
                 params_with_grad: List[str],
                 state: dict,
                 *,
                 lr: float,
                 alpha: float,
                 eps: float,
                 weight_decay: float,
                 momentum: float,
                 centered: bool,
                 ddp_descendent: bool = False,
                 i: int = 0):
    ddp_descendent = ddp_descendent | isinstance(net, nn.parallel.DistributedDataParallel)

    for state_id, (param_name, param) in zip(params_with_grad[i:], net._parameters.items()):
        if param is None or param._fast_grad is None:
            continue
        state_ = state[state_id]
        grad = param._fast_grad
        param._fast_grad = None

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        state_["square_avg"] = state_["square_avg"].mul(alpha).addcmul(grad, grad, value=1 - alpha)

        if centered:
            state_["grad_avg"] = state_["grad_avg"].mul(alpha).add(grad, alpha=1 - alpha)
            avg = state_["square_avg"].addcmul(state_["grad_avg"], state_["grad_avg"], value=-1).sqrt().add(eps)
        else:
            avg = state_["square_avg"].sqrt().add(eps)

        if momentum > 0:
            buffer = state_["momentum_buffer"].mul(momentum).addcdiv(grad, avg)
            net._parameters[param_name] = param.add(buffer, alpha=-lr)
        else:
            net._parameters[param_name] = param.addcdiv(grad, avg, value=-lr)

        net._parameters[param_name]._fast_grad = None

        i += 1
    for module_name, module in net._modules.items():
        if ddp_descendent:
            # DDP does not accept inplace buffer
            slide_buffer_window(module)

        i = meta_rmsprop(module,
                         params_with_grad,
                         state,
                         lr=lr,
                         alpha=alpha,
                         eps=eps,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         centered=centered,
                         ddp_descendent=ddp_descendent,
                         i=i)
    return i
