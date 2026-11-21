# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch
import cann_ops_nn

DTYPE_TOLERANCE = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
}


def compare_results(result_npu, result_cpu, dtype, name=""):
    rtol, atol = DTYPE_TOLERANCE.get(dtype, (1e-5, 1e-5))
    result_on_cpu = result_npu.cpu() if result_npu.device.type != "cpu" else result_npu
    max_diff = (result_on_cpu - result_cpu).abs().max().item()
    print(f"{name} - max error: {max_diff:.6f}")
    assert torch.allclose(
        result_on_cpu.float(), result_cpu.float(), rtol=rtol, atol=atol
    ), f"Precision check failed: {name}"


def situ_glu_cpu(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
    d = x.shape[dim] // 2
    left, right = x.split(d, dim=dim)
    if activate_left:
        gate, up = left, right
    else:
        gate, up = right, left

    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None and linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def situ_glu_group_grad_cpu(
    grad_y,
    x,
    dim=-1,
    beta=1.0,
    linear_beta=0.0,
    activate_left=True,
    weight=None,
    group_index=None,
    group_list_type=0,
):
    d = x.shape[dim] // 2
    left, right = x.split(d, dim=dim)
    if activate_left:
        gate, up = left, right
    else:
        gate, up = right, left

    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    grad_y = grad_y.to(torch.float32)

    t = torch.tanh(gate / beta)
    s = torch.sigmoid(gate)
    situ_a = beta * t * s
    if linear_beta > 0:
        up_prime = linear_beta * torch.tanh(up / linear_beta)
    else:
        up_prime = up

    grad_situ_a = grad_y * up_prime
    grad_up_prime = grad_y * situ_a

    grad_gate = grad_situ_a * s * ((1 - t * t) + beta * t * (1 - s))
    if linear_beta > 0:
        t_u = torch.tanh(up / linear_beta)
        grad_up = grad_up_prime * (1 - t_u * t_u)
    else:
        grad_up = grad_up_prime

    grad_w = None
    if weight is not None:
        w = weight.to(torch.float32).reshape(-1, 1)
        # unweighted integrand for grad_weight, before exit-style weight multiply
        grad_w = (grad_y * situ_a * up_prime).sum(dim=-1)
        grad_gate = grad_gate * w
        grad_up = grad_up * w

    if activate_left:
        grad_x = torch.cat([grad_gate, grad_up], dim=dim)
    else:
        grad_x = torch.cat([grad_up, grad_gate], dim=dim)

    real_bs = None
    if group_index is not None:
        # group_list_type 0 (cumsum): the last element is the total row count;
        # group_list_type 1 (count): the sum of all elements is the total row count
        group_rows = (
            int(group_index[-1].item())
            if group_list_type == 0
            else int(group_index.sum().item())
        )
        real_bs = min(group_rows, grad_x.shape[0])
        # masked rows are not computed and not written by the kernel; mirror by zeroing
        # the golden rows and letting the caller compare only rows < real_bs
    grad_x = grad_x.to(x.dtype)
    return grad_x, grad_w, real_bs


NETWORK_SHAPES = [
    (2, 32),
    (1, 4624, 6144),
    (15908, 3072),
]

DTYPES = [torch.float32, torch.float16]

NET_DIM = -1
NET_BETA = 4.0
NET_LINEAR_BETA = 25.0
NET_ACTIVATE_LEFT = True


def _make_x(shape, dtype):
    if dtype == torch.float32:
        return torch.randn(*shape, dtype=torch.float32).npu()
    return torch.randn(*shape, dtype=torch.float32).to(dtype).npu()


def _make_grad_y_for(x, dim=-1):
    d = x.shape[dim] // 2
    y_shape = list(x.shape)
    y_shape[dim if dim >= 0 else dim + len(x.shape)] = d
    return torch.randn(*y_shape, dtype=torch.float32).to(x.dtype).npu()


@pytest.mark.parametrize("shape", NETWORK_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_situ_glu_group_grad_network(shape, dtype):
    x = _make_x(shape, dtype)
    grad_y = _make_grad_y_for(x, dim=NET_DIM)
    grad_x, grad_weight = cann_ops_nn.situ_glu_group_grad(
        grad_y,
        x,
        dim=NET_DIM,
        beta=NET_BETA,
        linear_beta=NET_LINEAR_BETA,
        activate_left=NET_ACTIVATE_LEFT,
    )
    # grad_weight is a required output: empty [0] when no weight is passed
    assert grad_weight.numel() == 0 and grad_weight.dtype == torch.float32
    grad_x_cpu, _, _ = situ_glu_group_grad_cpu(
        grad_y.cpu(),
        x.cpu(),
        dim=NET_DIM,
        beta=NET_BETA,
        linear_beta=NET_LINEAR_BETA,
        activate_left=NET_ACTIVATE_LEFT,
    )
    assert grad_x.dtype == x.dtype
    compare_results(
        grad_x, grad_x_cpu, dtype, name=f"situ_glu_group_grad_network_{shape}_{dtype}"
    )


OPTIONAL_DTYPES = [torch.float32, torch.bfloat16]


def _make_weight(rows):
    return (torch.rand(rows, dtype=torch.float32) + 0.5).npu()


@pytest.mark.parametrize("dtype", OPTIONAL_DTYPES)
@pytest.mark.parametrize("activate_left", [True, False])
@pytest.mark.parametrize("linear_beta", [0.0, 25.0])
def test_situ_glu_group_grad_weight(dtype, activate_left, linear_beta):
    # weight + grad_weight, no group_index
    rows, h = 8, 32
    x = _make_x((rows, 2 * h), dtype)
    grad_y = _make_grad_y_for(x)
    weight = _make_weight(rows)
    grad_x, grad_weight = cann_ops_nn.situ_glu_group_grad(
        grad_y,
        x,
        weight=weight,
        dim=-1,
        beta=1.3,
        linear_beta=linear_beta,
        activate_left=activate_left,
    )
    assert grad_weight is not None and grad_weight.dtype == torch.float32
    grad_x_cpu, grad_w_cpu, _ = situ_glu_group_grad_cpu(
        grad_y.cpu(),
        x.cpu(),
        beta=1.3,
        linear_beta=linear_beta,
        activate_left=activate_left,
        weight=weight.cpu(),
    )
    compare_results(
        grad_x,
        grad_x_cpu,
        dtype,
        name=f"grad_x_weight_{dtype}_{activate_left}_{linear_beta}",
    )
    compare_results(
        grad_weight,
        grad_w_cpu,
        torch.float32,
        name=f"grad_weight_{dtype}_{activate_left}_{linear_beta}",
    )


@pytest.mark.parametrize("dtype", OPTIONAL_DTYPES)
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_situ_glu_group_grad_weight_group(dtype, group_list_type):
    # weight + group_index (total row count < rows) + grad_weight: only the first real_bs rows are written
    rows, h = 6, 32
    x = _make_x((rows, 2 * h), dtype)
    grad_y = _make_grad_y_for(x)
    weight = _make_weight(rows)
    # count mode lists the per-group sizes (2 + 1 + 0), cumsum mode the running totals (ends at 3)
    group_sizes = [2, 1, 0] if group_list_type == 1 else [1, 2, 3]
    group_index = torch.tensor(group_sizes, dtype=torch.int64).npu()
    real_bs = 3
    grad_x, grad_weight = cann_ops_nn.situ_glu_group_grad(
        grad_y,
        x,
        weight=weight,
        group_index=group_index,
        dim=-1,
        beta=1.3,
        linear_beta=0.8,
        activate_left=True,
        group_list_type=group_list_type,
    )
    assert grad_weight is not None
    grad_x_cpu, grad_w_cpu, _ = situ_glu_group_grad_cpu(
        grad_y.cpu(),
        x.cpu(),
        beta=1.3,
        linear_beta=0.8,
        activate_left=True,
        weight=weight.cpu(),
        group_index=group_index.cpu(),
        group_list_type=group_list_type,
    )
    compare_results(
        grad_x[:real_bs],
        grad_x_cpu[:real_bs],
        dtype,
        name=f"grad_x_group_{dtype}_{group_list_type}",
    )
    compare_results(
        grad_weight[:real_bs],
        grad_w_cpu[:real_bs],
        torch.float32,
        name=f"grad_weight_group_{dtype}_{group_list_type}",
    )


@pytest.mark.parametrize("dtype", OPTIONAL_DTYPES)
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_situ_glu_group_grad_group_only(dtype, group_list_type):
    # group_index without weight: the total row count (sum 12 / last element 7) exceeds rows,
    # so it is clamped to rows and every row is valid in both group_list_type modes
    rows, h = 4, 32
    x = _make_x((rows, 2 * h), dtype)
    grad_y = _make_grad_y_for(x)
    group_index = torch.tensor([5, 7], dtype=torch.int64).npu()
    grad_x, grad_weight = cann_ops_nn.situ_glu_group_grad(
        grad_y,
        x,
        group_index=group_index,
        dim=-1,
        beta=4.0,
        linear_beta=25.0,
        activate_left=True,
        group_list_type=group_list_type,
    )
    # no weight -> grad_weight is an empty [0] tensor
    assert grad_weight.numel() == 0 and grad_weight.dtype == torch.float32
    grad_x_cpu, _, _ = situ_glu_group_grad_cpu(
        grad_y.cpu(),
        x.cpu(),
        beta=4.0,
        linear_beta=25.0,
        activate_left=True,
    )
    compare_results(
        grad_x,
        grad_x_cpu,
        dtype,
        name=f"grad_x_group_clamp_{dtype}_{group_list_type}",
    )
