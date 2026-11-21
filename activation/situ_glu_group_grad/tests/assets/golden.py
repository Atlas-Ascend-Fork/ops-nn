# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""situ_glu_group_grad TestSpec — kernel (numpy) + aclnn (torch) golden.

Outputs (order matches def.cpp / aclnn*.h):
  grad_x      (always) — gradient w.r.t. x,      shape = x,      dtype = grad_y.dtype
  grad_weight (always) — gradient w.r.t. weight, shape = weight if provided else empty [0], dtype = float32

Attributes: dim, beta, linear_beta, activate_left, group_list_type

Forward:  y = w * situ_a(gate) * up'
Backward (fp32 recompute):
  t = tanh(gate / beta), s = sigmoid(gate), A = beta * t * s
  U = linear_beta * tanh(up / linear_beta) if linear_beta > 0 else up
  grad_gate = grad_y * U * s * [(1 - t^2) + beta * t * (1 - s)] * w
  grad_up   = grad_y * A * (1 - t_u^2)  (linear_beta > 0) else grad_y * A   (* w)
  grad_w[n] = sum_h grad_y * A * U      (unweighted, over the h axis)

group_index is row truncation: realBs = min(groupRows, rows), where groupRows is group_index[-1] in
cumsum mode (group_list_type 0, the default) and sum(group_index) in count mode (group_list_type 1);
only the first realBs rows are produced by the kernel and masked rows are left untouched (not written,
not zeroed). The golden computes all rows, so customize_inputs normalizes group_index to cover every
row to keep the comparison well defined (mirrors TTK behaviour).
"""

import numpy

__spec__ = {
    "situ_glu_group_grad": "SituGluGroupGradTestSpec",
    "aclnnSituGluGroupGrad": "AclnnSituGluGroupGradTestSpec",
    "torch.ops.cann_ops_nn.situ_glu_group_grad": "E2eSituGluGroupGradTestSpec",
}

_GROUP_LIST_TYPE_CUMSUM = 0
_GROUP_LIST_TYPE_COUNT = 1


def _stable_sigmoid(x):
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        y = numpy.empty_like(x, dtype=numpy.float32)
        pos = x >= 0
        y[pos] = 1.0 / (1.0 + numpy.exp(-x[pos]))
        exp_x = numpy.exp(x[~pos])
        y[~pos] = exp_x / (1.0 + exp_x)
    return y


def _rows_of(x, dim):
    d = dim if dim >= 0 else dim + x.ndim
    rows = 1
    for s in x.shape[:d]:
        rows *= int(s)
    return d, rows


def _compute_situ_glu_group_grad(
    grad_y, x, weight, dim, beta, linear_beta, activate_left
):
    """Core numpy computation (fp32). Returns (grad_x, grad_weight).

    grad_weight is a flat float32 array of `rows` elements when weight is provided, else empty [0].
    """
    if x.shape == grad_y.shape:
        raise Exception(
            "SituGluGroupGrad grad_y and x must differ on the dim axis, got the same shape: {}".format(
                x.shape
            )
        )
    if x.dtype != grad_y.dtype:
        raise Exception(
            "SituGluGroupGrad grad_y and x dtype must match, got {} vs {}".format(
                grad_y.dtype, x.dtype
            )
        )

    d, rows = _rows_of(x, dim)
    axis = d if d >= 0 else d + x.ndim
    cut = x.shape[axis]
    if cut % 2 != 0:
        raise Exception(
            "SituGluGroupGrad x dim axis must be even, but got shape: {}".format(
                x.shape
            )
        )
    h = cut // 2
    if grad_y.shape[axis] != h:
        raise Exception(
            "SituGluGroupGrad grad_y dim axis must be half of x, got grad_y={}, x={}".format(
                grad_y.shape, x.shape
            )
        )

    x_m = numpy.moveaxis(x.astype(numpy.float32), axis, -1)
    gy_m = numpy.moveaxis(grad_y.astype(numpy.float32), axis, -1)

    first = x_m[..., :h]
    second = x_m[..., h:]
    gate, up = (first, second) if activate_left else (second, first)

    beta = float(beta)
    linear_beta = float(linear_beta)
    t = numpy.tanh(gate / beta)
    s = _stable_sigmoid(gate)
    a = beta * t * s
    if linear_beta > 0.0:
        t_u = numpy.tanh(up / linear_beta)
        up_prime = linear_beta * t_u
        up_grad = 1.0 - t_u * t_u
    else:
        up_prime = up
        up_grad = numpy.ones_like(up, dtype=numpy.float32)

    grad_gate = gy_m * up_prime * s * ((1.0 - t * t) + beta * t * (1.0 - s))
    grad_up = gy_m * a * up_grad

    grad_weight = numpy.zeros((0,), dtype=numpy.float32)
    if weight is not None:
        weight_np = weight.astype(numpy.float32)
        if weight_np.size != rows:
            raise Exception(
                "SituGluGroupGrad weight numel must equal prod(x.shape[:dim]) = {}, got {}".format(
                    rows, weight_np.size
                )
            )
        # weight is per row over x.shape[:axis]; broadcast over the moved layout
        # x_m shape = x.shape[:axis] + x.shape[axis+1:] + (h,), so the trailing
        # (x.ndim - axis) axes (post-split dims + the h axis) are broadcast dims.
        w = weight_np.reshape(x.shape[:axis] + (1,) * (x.ndim - axis))
        grad_gate = grad_gate * w
        grad_up = grad_up * w
        integrand = gy_m * a * up_prime
        # grad_w[n] sums every element sharing row n: the h axis plus all dims after
        # the split axis (in the moved layout these are axes axis..ndim-1).
        reduce_axes = tuple(range(axis, integrand.ndim))
        grad_weight = (
            numpy.sum(integrand, axis=reduce_axes, dtype=numpy.float32)
            .reshape(weight_np.shape)
            .astype(numpy.float32)
        )

    if activate_left:
        # first half is gate, second half is up
        grad_x_m = numpy.concatenate([grad_gate, grad_up], axis=-1)
    else:
        # first half is up, second half is gate
        grad_x_m = numpy.concatenate([grad_up, grad_gate], axis=-1)
    grad_x = numpy.moveaxis(grad_x_m, -1, axis).astype(grad_y.dtype, copy=False)
    return grad_x, grad_weight


def _group_index_values(n, rows, group_list_type):
    """group_index values covering all `rows` rows, so the golden comparison is well defined.

    count mode: sum(values) == rows; cumsum mode: values[-1] == rows (non-decreasing).
    """
    values = numpy.zeros((n,), dtype=numpy.int64)
    if group_list_type == _GROUP_LIST_TYPE_COUNT:
        base = rows // n
        values[:] = base
        values[-1] = rows - base * (n - 1)
    else:
        for i in range(n):
            values[i] = rows * (i + 1) // n
    return values


def _fill_group_index(group_index, rows, group_list_type=_GROUP_LIST_TYPE_CUMSUM):
    """Overwrite group_index in place so that every row is produced by the kernel.

    masked rows (>= realBs) are not written by the kernel, so the golden cannot define them;
    normalizing the group sizes keeps the comparison well defined.
    """
    if group_index is None:
        return
    try:
        import torch

        is_torch = isinstance(group_index, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        n = group_index.numel()
        if n == 0:
            return
        values = _group_index_values(n, rows, group_list_type)
        group_index.copy_(torch.from_numpy(values.reshape(group_index.shape)))
        return

    gi = numpy.asarray(group_index)
    n = gi.size
    if n == 0:
        return
    values = _group_index_values(n, rows, group_list_type)
    if isinstance(group_index, numpy.ndarray):
        numpy.copyto(group_index, values.reshape(group_index.shape))


def _rows_from_shape(shape, dim):
    ndim = len(shape)
    axis = dim if dim >= 0 else dim + ndim
    rows = 1
    for s in shape[:axis]:
        rows *= int(s)
    return rows


class _TorchSituGluGroupGrad:
    """torch 三方标杆（--xpu-perf / cross_check）：全程 torch 算子，可在 XPU 设备执行。

    与 _compute_situ_glu_group_grad 同口径（fp32 重算）；group_index 仅做行截断，
    customize_inputs 已归一化为满行，故此处忽略。Kernel/GEIR 传入前 numpy 已转
    torch.Tensor，参数名与 def.cpp 一致。
    """

    def __init__(
        self, *, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True, **kwargs
    ):
        self.dim = dim
        self.beta = float(beta)
        self.linear_beta = float(linear_beta)
        self.activate_left = activate_left

    def __call__(self, grad_y, x, weight=None, group_index=None, **kwargs):
        import torch

        axis = self.dim if self.dim >= 0 else self.dim + x.dim()
        h = x.shape[axis] // 2
        gy = grad_y.to(torch.float32)
        xf = x.to(torch.float32)
        first, second = xf.narrow(axis, 0, h), xf.narrow(axis, h, h)
        gate, up = (first, second) if self.activate_left else (second, first)

        t = torch.tanh(gate / self.beta)
        s = torch.sigmoid(gate)
        a = self.beta * t * s
        if self.linear_beta > 0.0:
            t_u = torch.tanh(up / self.linear_beta)
            up_prime = self.linear_beta * t_u
            up_grad = 1.0 - t_u * t_u
        else:
            up_prime = up
            up_grad = torch.ones_like(up)

        grad_gate = gy * up_prime * s * ((1.0 - t * t) + self.beta * t * (1.0 - s))
        grad_up = gy * a * up_grad

        grad_weight = torch.zeros((0,), dtype=torch.float32, device=x.device)
        if weight is not None:
            w = weight.to(torch.float32).reshape(
                list(x.shape[:axis]) + [1] * (x.dim() - axis)
            )
            grad_gate, grad_up = grad_gate * w, grad_up * w
            integrand = gy * a * up_prime  # grad_weight 不带 weight（与 golden 同口径）
            grad_weight = (
                integrand.sum(
                    dim=tuple(range(axis, integrand.dim())), dtype=torch.float32
                )
                .reshape(weight.shape)
                .to(torch.float32)
            )

        parts = [grad_gate, grad_up] if self.activate_left else [grad_up, grad_gate]
        return [torch.cat(parts, dim=axis).to(grad_y.dtype), grad_weight]


class SituGluGroupGradTestSpec:
    """situ_glu_group_grad kernel golden (numpy in; params match def.cpp).
    def.cpp inputs : grad_y, x, weight(optional), group_index(optional)
    def.cpp attr   : dim, beta, linear_beta, activate_left, group_list_type
    def.cpp outputs: grad_x, grad_weight(required)
    """

    third_party = {"torch": _TorchSituGluGroupGrad}

    def golden(
        grad_y,
        x,
        weight=None,
        group_index=None,
        dim=-1,
        beta=1.0,
        linear_beta=0.0,
        activate_left=True,
        group_list_type=_GROUP_LIST_TYPE_CUMSUM,
        **kwargs,
    ):
        # group_index only truncates rows; customize_inputs normalizes it to full coverage
        del group_index, group_list_type
        grad_x, grad_weight = _compute_situ_glu_group_grad(
            grad_y, x, weight, dim, beta, linear_beta, activate_left
        )
        return [grad_x, grad_weight]

    @staticmethod
    def customize_inputs(
        grad_y,
        x,
        weight=None,
        group_index=None,
        dim=-1,
        beta=1.0,
        linear_beta=0.0,
        activate_left=True,
        group_list_type=_GROUP_LIST_TYPE_CUMSUM,
        **kwargs,
    ):
        if group_index is not None:
            _, rows = _rows_of(x, dim)
            _fill_group_index(group_index, rows, group_list_type)
        return (grad_y, x, weight, group_index)


class AclnnSituGluGroupGradTestSpec:
    """aclnnSituGluGroupGrad golden (torch.Tensor in; params match aclnn*.h).
    header tensors : gradY, x, weightOptional, groupIndexOptional
    header scalar  : dim, beta, linearBeta, activateLeft, groupListType
    header outputs : gradXOut, gradWeightOut
    """

    def golden(
        gradY,
        x,
        weightOptional=None,
        groupIndexOptional=None,
        dim=-1,
        beta=1.0,
        linearBeta=0.0,
        activateLeft=True,
        groupListType=_GROUP_LIST_TYPE_CUMSUM,
        gradXOut=None,
        gradWeightOut=None,
        **kwargs,
    ):
        import torch

        def _to_numpy(t):
            if t is None:
                return None
            if hasattr(t, "detach"):
                t_cpu = t.detach().cpu()
                if t_cpu.dtype == torch.bfloat16:
                    # numpy() has no bfloat16 support; widen to float32 for the fp32 golden
                    return t_cpu.float().numpy()
                return t_cpu.numpy()
            return t

        grad_y_np = _to_numpy(gradY)
        x_np = _to_numpy(x)
        weight_np = _to_numpy(weightOptional)
        # groupIndexOptional only truncates rows; customize_inputs normalizes it, so it is unused here
        del groupIndexOptional, groupListType

        grad_x_np, grad_weight_np = _compute_situ_glu_group_grad(
            grad_y_np, x_np, weight_np, dim, beta, linearBeta, activateLeft
        )
        results = [
            torch.from_numpy(grad_x_np).to(gradY.dtype),
            torch.from_numpy(grad_weight_np),
        ]
        return results

    @staticmethod
    def customize_inputs(
        gradY,
        x,
        weightOptional=None,
        groupIndexOptional=None,
        dim=-1,
        beta=1.0,
        linearBeta=0.0,
        activateLeft=True,
        groupListType=_GROUP_LIST_TYPE_CUMSUM,
        gradXOut=None,
        gradWeightOut=None,
        **kwargs,
    ):
        if groupIndexOptional is not None:
            # x may be bfloat16 (torch), which numpy() cannot convert; rows only needs the shape
            rows = _rows_from_shape(x.shape, dim)
            _fill_group_index(groupIndexOptional, rows, groupListType)
        return None


class E2eSituGluGroupGradTestSpec:
    """torch.ops.cann_ops_nn.situ_glu_group_grad e2e golden (torch.Tensor in;
    params match the torch op schema)."""

    third_party = {"torch": _TorchSituGluGroupGrad}

    def golden(
        grad_y,
        x,
        weight=None,
        group_index=None,
        dim=-1,
        beta=1.0,
        linear_beta=0.0,
        activate_left=True,
        group_list_type=_GROUP_LIST_TYPE_CUMSUM,
        **kwargs,
    ):
        import torch

        def _to_numpy(t):
            if t is None:
                return None
            t_cpu = t.detach().cpu()
            if t_cpu.dtype == torch.bfloat16:
                return t_cpu.float().numpy()
            return t_cpu.numpy()

        grad_x_np, grad_weight_np = _compute_situ_glu_group_grad(
            _to_numpy(grad_y),
            _to_numpy(x),
            _to_numpy(weight),
            dim,
            beta,
            linear_beta,
            activate_left,
        )
        return [
            torch.from_numpy(grad_x_np).to(grad_y.dtype),
            torch.from_numpy(grad_weight_np),
        ]

    @staticmethod
    def customize_inputs(
        grad_y,
        x,
        weight=None,
        group_index=None,
        dim=-1,
        beta=1.0,
        linear_beta=0.0,
        activate_left=True,
        group_list_type=_GROUP_LIST_TYPE_CUMSUM,
        **kwargs,
    ):
        if group_index is not None:
            _fill_group_index(
                group_index, _rows_from_shape(x.shape, dim), group_list_type
            )
        return None
