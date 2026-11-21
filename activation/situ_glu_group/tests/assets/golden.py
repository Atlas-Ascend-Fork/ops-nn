#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""situ_glu_group 的 TTK 多通路 TestSpec golden。

覆盖通路：
  kernel / geir           -> SituGluGroupKernelSpec（输入 numpy.ndarray）
  aclnnSituGluGroup       -> SituGluGroupAclnnSpec（aclnn 接口）
  cann_ops_nn.situ_glu_group（单算子 / TorchAir） -> SituGluGroupE2ESpec（输入 torch.Tensor）

语义（见 docs/aclnnSituGluGroup.md / op_graph/situ_glu_group_proto.h）：
  x 沿 dim 合轴为 [pre, cut]，h = cut/2；activate_left=true 时 gate=前半、up=后半，否则对调。
  situ_a = beta * tanh(gate / beta) * sigmoid(gate)
  up'    = linear_beta > 0 ? linear_beta * tanh(up / linear_beta) : up
  y      = situ_a * up'
  可选 weight（逐行标量，元素数 = dim 之前各维乘积）：y[i] = y[i] * weight[i]
  可选 group_index（int64，1D 非空）：realBs 由 group_list_type 决定——
    count 模式（1）：realBs = min(sum(group_index), 行数)；
    cumsum 模式（0）：realBs = min(group_index[-1], 行数)（最后一个累计结尾偏移）。
  仅前 realBs 行写出。

group_index 的处理：
  - count 模式写每组行数、cumsum 模式写其前缀和，两种模式下"总量"的语义一致：
  - 默认（用例名不含 giclamp/trunc）：总量填成行数，realBs == 行数，
    golden 按全量行计算，整 tensor 比对。
  - giclamp：总量填成 2 倍行数，验证 min(总量, bs) 的截断上限。
  - trunc：总量填成小于行数的值（默认行数的一半，含 trunc1 时填 1），真实触发
    realBs < 行数的截断路径；golden 只返回前 realBs 行，pre_compare 把 NPU 输出裁到
    同样行数后再比对。
  - 模式来源：优先使用 `group_index_mode` 入参（sum_equals_rows / clamp / trunc /
    trunc_one），未提供时按用例名（giclamp / trunc1 / trunc）推断，保持旧 CSV 兼容。
  注意：截断用例只校验 [0, realBs) 前缀（realBs 算小会被发现）；realBs 之后的行是否被
  意外写入，需要在 kernel UT 里用 sentinel 校验（test_situ_glu_group.cpp 的
  group_index_tail_* / group_index_multichunk_tail 已覆盖）。截断支持任意 dim：
  [bs, splitD] 合并视图是行主序，"前 realBs 行"等价于扁平前缀 realBs * splitD 个元素。

GPU 三方标杆（third_party）：
  TTK 在 XPU/GPU 服务端执行 `_SituGluGroupTorchCompose`：小算子拼接
  (split + tanh + sigmoid + mul) 并外包 torch.compile（竞品最优形态——NPU 侧
  是单个融合内核，拿未融合 eager 逐算子 launch 比不是同一层次）；编译失败自动
  回落 eager。golden 是 float32 numpy 独立写法，cross_check 不会与 golden 同源恒 0。
  group_index 截断用例三方腿同样只返回前 realBs 行，只比前缀；尾行未定义的校验
  仍留在 kernel UT 的 sentinel 用例里。
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:  # pragma: no cover
    _bf16 = None

__spec__ = {
    "situ_glu_group": "SituGluGroupKernelSpec",
    "aclnnSituGluGroup": "SituGluGroupAclnnSpec",
    "cann_ops_nn.situ_glu_group": "SituGluGroupE2ESpec",
    "torch.ops.cann_ops_nn.situ_glu_group": "SituGluGroupE2ESpec",
}


def _to_np(t):
    """numpy / torch(CPU|NPU) -> numpy.ndarray（bf16/fp16 抬到 fp32 便于计算）。"""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    import torch

    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype in (torch.float16, torch.bfloat16):
            return t.float().numpy()
        return t.numpy()
    return np.asarray(t)


def _cast_out(y, target):
    s = str(target)
    if "bfloat16" in s:
        if _bf16 is None:
            # 静默退回 float32 会隐藏 bf16 舍入，导致 golden 比真实输出"更准"，
            # 掩盖精度问题；显式失败更安全。
            raise RuntimeError(
                "bfloat16 golden requires ml_dtypes; install it with `pip install ml_dtypes`"
            )
        return y.astype(_bf16)
    if "float16" in s:
        return y.astype(np.float16)
    if "float32" in s:
        return y.astype(np.float32)
    return y.astype(s)


def _row_count(shape, dim):
    n = 1
    for s in shape[:dim]:
        n *= int(s)
    return n


def _out_shape(shape, d):
    """infershape 语义：dim 轴折半，其余维度不变。"""
    out = list(shape)
    out[d] = out[d] // 2
    return tuple(out)


def _empty_like_shape(x_np, d, target):
    """空 tensor 直接给正确 shape/dtype 的空输出，不进入逐元素计算。"""
    return [_cast_out(np.empty(_out_shape(x_np.shape, d), dtype=np.float32), target)]


def _ref(x, weight, dim, beta, linear_beta, activate_left, target):
    """float32 CPU reference shared by all paths. Returns [y] as numpy."""
    x_np = _to_np(x)
    nd = x_np.ndim
    if nd == 0:
        raise ValueError("situ_glu_group golden expects an input with rank >= 1")
    d = dim % nd
    if x_np.size == 0:
        # 空输入（任一轴为 0，含最后一维为 0）：infershape 仍给出 last=cut//2 的空输出，
        # 这里必须显式返回该形状，不能依赖 np.split/reshape 的边界行为。
        return _empty_like_shape(x_np, d, target)
    # 参考计算统一用 float64：cross_check/mix_tolerance 会把 golden 走 Promote
    # （fp16/bf16→fp32、fp32→fp64），这里若降回 float32 就等于把框架抬上去的精度
    # 又砍回来（kernel/golden.py:793-799 明确点名的反模式）；最终由 _cast_out 降到
    # 目标 dtype。
    xf = x_np.astype(np.float64)
    h = xf.shape[d] // 2
    first, second = np.split(xf, [h], axis=d)
    gate, up = (first, second) if activate_left else (second, first)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        sig = 1.0 / (1.0 + np.exp(-gate))
        situ_a = beta * np.tanh(gate / beta) * sig
        up2 = linear_beta * np.tanh(up / linear_beta) if linear_beta > 0 else up
        y = situ_a * up2

    w = _to_np(weight)
    if w is not None:
        wshape = tuple(xf.shape[:d]) + (1,) * (nd - d)
        y = y * w.astype(np.float64).reshape(wshape)

    return [_cast_out(y, target)]


GROUP_LIST_TYPE_CUMSUM = 0  # group_index[i] is a cumulative row end offset
GROUP_LIST_TYPE_COUNT = 1  # group_index[i] is a per-group row count

GROUP_INDEX_MODE_SUM_EQUALS_ROWS = "sum_equals_rows"
GROUP_INDEX_MODE_CLAMP = "clamp"  # total == 2 * rows, kernel clamps realBs to rows
GROUP_INDEX_MODE_TRUNC = "trunc"  # total == rows // 2, real truncation
GROUP_INDEX_MODE_TRUNC_ONE = "trunc_one"  # total == 1, real truncation down to one row
_GROUP_INDEX_MODES = (
    GROUP_INDEX_MODE_SUM_EQUALS_ROWS,
    GROUP_INDEX_MODE_CLAMP,
    GROUP_INDEX_MODE_TRUNC,
    GROUP_INDEX_MODE_TRUNC_ONE,
)


def _resolve_group_list_type(value):
    """Normalize group_list_type; only 0 (cumsum) and 1 (count) are supported."""
    if value is None:
        return GROUP_LIST_TYPE_CUMSUM
    mode = int(value)
    if mode not in (GROUP_LIST_TYPE_CUMSUM, GROUP_LIST_TYPE_COUNT):
        raise ValueError(
            f"group_list_type must be 0 (cumsum) or 1 (count), but got {value}"
        )
    return mode


def _real_bs(group_index_np, group_list_type, rows):
    """Number of rows the operator writes, per group_index semantics.

    count  : total = sum(group_index)
    cumsum : total = group_index[-1] (the last cumulative end offset)
    The total is accumulated in Python integers (an int64 sum would wrap) and clamped to [0, rows];
    the kernel saturates at `rows`, so this mirrors its result for every non-negative input.
    """
    gi = np.asarray(group_index_np).reshape(-1)
    if gi.size == 0:
        return 0
    if group_list_type == GROUP_LIST_TYPE_CUMSUM:
        total = int(gi[-1])
    else:
        # Accumulate in Python integers: np.int64 wraps for a total >= 2^63, while the kernel
        # saturates at `rows`, so a wrapping sum would make the golden reject valid operator output.
        total = sum(int(v) for v in gi.reshape(-1))
    return max(0, min(total, rows))


# 三方标杆（GPU 小算子拼接 + torch.compile）的验收档。tanh/sigmoid 在不同实现里
# 的末位舍入不同，且 weight/group 截断只比前缀，取 L1（elementwise 有超越函数，
# 比 L0 更贴近真实跨平台差异）。
# 不覆盖 small_value/small_value_atol，直接用 TTK 默认档（fp32 small_value=2^-14、
# small_value_atol=2^-30）：本算子近零点由小因子乘积得到，绝对误差随 |y| 同比
# 缩小，实测最大 ~7.6e-10，低于 fp32 默认 small_value_atol=2^-30。small_value 同时
# 是三比值的安全分母下限，放宽它会让小值桶与比值判据一起失去区分度。若后续真
# 出现近零点计数失败，再只针对 fp32 配 small_value_atol=1e-6（≈1 fp32 ULP 量级）。
_TOL_CROSS_CHECK = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _group_index_mode(testcase_name="", explicit=None):
    """Resolve the group_index fill mode.

    `explicit` (e.g. a `group_index_mode` CSV column surfaced through kwargs) wins
    over the legacy name-based convention, so the semantics stop depending on how a
    case happens to be named. Unknown values fail loudly instead of silently
    degrading to "sum == rows".
    """
    if explicit is not None:
        mode = str(explicit).strip().lower()
        if mode not in _GROUP_INDEX_MODES:
            raise ValueError(
                f"unknown group_index_mode {explicit!r}, expected one of {_GROUP_INDEX_MODES}"
            )
        return mode
    name = str(testcase_name).lower()
    if "giclamp" in name:
        return GROUP_INDEX_MODE_CLAMP
    if "trunc1" in name:
        return GROUP_INDEX_MODE_TRUNC_ONE
    if "trunc" in name:
        return GROUP_INDEX_MODE_TRUNC
    return GROUP_INDEX_MODE_SUM_EQUALS_ROWS


def _fill_group_index(
    x,
    group_index,
    dim,
    testcase_name="",
    mode=None,
    group_list_type=GROUP_LIST_TYPE_COUNT,
):
    """按用例语义填 group_index 的值。

    count 模式（group_list_type=1）下写入每组行数：
    - sum_equals_rows：sum == 行数，所有行都产出，可整 tensor 比对；
    - clamp          ：sum == 2 * 行数，验证 min(sum, 行数) 的上钳制；
    - trunc          ：sum == max(1, 行数 // 2)，触发真实截断；
    - trunc_one      ：sum == 1，触发最小截断。
    cumsum 模式（group_list_type=0）下把相同分布写成前缀和，最后一个元素即目标行数。
    """
    if group_index is None:
        return
    x_np = _to_np(x)
    d = int(dim) % x_np.ndim
    rows = _row_count(x_np.shape, d)
    resolved = _group_index_mode(testcase_name, mode)
    if resolved == GROUP_INDEX_MODE_CLAMP:
        target = rows * 2
    elif resolved == GROUP_INDEX_MODE_TRUNC_ONE:
        target = 1
    elif resolved == GROUP_INDEX_MODE_TRUNC:
        target = max(1, rows // 2)
    else:
        target = rows
    gi = (
        group_index.detach().cpu().numpy().reshape(-1)
        if hasattr(group_index, "detach")
        else np.asarray(group_index).reshape(-1)
    )
    n = gi.size
    if n == 0:
        return
    base = target // n
    gi[:] = base
    gi[-1] = target - base * (n - 1)
    if _resolve_group_list_type(group_list_type) == GROUP_LIST_TYPE_CUMSUM:
        np.cumsum(gi, out=gi)
    if hasattr(group_index, "copy_"):
        import torch

        group_index.copy_(torch.from_numpy(gi.reshape(group_index.shape)))
    else:
        np.copyto(group_index, gi.reshape(group_index.shape))


def _apply_group_truncation(
    result, x, group_index, dim, group_list_type=GROUP_LIST_TYPE_COUNT
):
    """realBs < 行数时只保留 golden 的前 realBs 行；其余行算子不写出，不参与比对。

    kernel 的 [bs, splitD] 合并视图在 numpy 侧就是行主序，因此"前 realBs 行"等价于
    扁平前缀 realBs * splitD 个元素，对任意 cut 轴位置都成立（不要求 dim 是最后一维）。
    最后一维沿用二维形状，其余维度返回扁平前缀；两种形态都由 pre_compare 按扁平
    前缀裁剪 NPU 输出后再逐元素比对。
    """
    if group_index is None:
        return result
    x_np = _to_np(x)
    nd = x_np.ndim
    d = int(dim) % nd
    rows = _row_count(x_np.shape, d)
    gi = _to_np(group_index).reshape(-1)
    if gi.size == 0:
        return result
    real_bs = _real_bs(gi, group_list_type, rows)
    if rows <= 0 or real_bs >= rows:
        return result
    y = result[0]
    split_d = y.size // rows
    if split_d == 0:
        # cut 轴为 0（如最后一维为 0）时输出本身为空，没有可截断的前缀；
        # 返回原空张量，避免 reshape(-1, 0) 的歧义报错。
        return result
    # real_bs == 0（总量为 0）时前缀为空：算子不写出任何行，输出内容全部未定义，
    # pre_compare 会跳过比对（golden 侧同样给空张量，保持语义一致）。
    if d == nd - 1:
        return [y.reshape(rows, split_d)[:real_bs]]
    return [y.reshape(-1)[: real_bs * split_d]]


def _to_np_output(t):
    """pre_compare 入参可能是 numpy 或 torch tensor（ACLNN/E2E）；bf16 torch 无 numpy 视图。"""
    if t is None or isinstance(t, np.ndarray):
        return t
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    try:
        return t.numpy()
    except TypeError:
        arr = t.float().numpy()
        return arr.astype(_bf16) if _bf16 is not None else arr


def _group_trunc_pre_compare(npu_out, golden_out):
    """把 NPU 输出裁到 golden 的行数（仅截断用例需要），其余情况保持默认比对。"""
    a = _to_np_output(npu_out)
    b = _to_np_output(golden_out)
    if a is None or b is None:
        return None
    if b.size == 0:
        # 总量为 0：算子不写出任何行，golden 给空前缀，比较 0 个元素。
        return [a.reshape(-1)[:0].reshape(b.shape), b]
    if a.size == b.size or a.size < b.size:
        return None
    return [a.reshape(-1)[: b.size].reshape(b.shape), b]


class _SituGluGroupTorchCompose:
    """GPU 三方标杆：小算子拼接 + torch.compile（竞品最优形态）。

    TTK 在 GPU 服务端实例化并调用本类：张量参数由 ``__call__`` 传入，标量属性可能
    经 ``__init__``（属性式）或 ``__call__``（函数式）传入，两处都接住，参数名兼容
    kernel/aclnn/e2e 三套拼写（linear_beta/linearBeta 等）。

    计算统一 float32（fp16/bf16 抬精度，出口 cast 回输入 dtype），与 golden 的
    numpy 写法互相独立；torch.compile 失败时永久回落 eager，避免每次调用重复抛
    异常污染耗时。截断用例与 golden 一致只返回前 realBs 行。
    """

    _DEFAULTS = {
        "dim": -1,
        "beta": 1.0,
        "linear_beta": 0.0,
        "activate_left": True,
        "group_list_type": GROUP_LIST_TYPE_CUMSUM,
    }
    _ALIASES = {
        "weightOptional": "weight",
        "groupIndexOptional": "group_index",
        "linearBeta": "linear_beta",
        "activateLeft": "activate_left",
        "groupListType": "group_list_type",
    }

    def __init__(self, *args, **kwargs):
        del args
        self._attrs = dict(self._DEFAULTS)
        self._compiled = None
        self._merge(kwargs)

    def _merge(self, kw):
        for key, value in kw.items():
            name = self._ALIASES.get(key, key)
            if name in self._DEFAULTS and value is not None:
                self._attrs[name] = value

    @staticmethod
    def _forward(x, weight, dim, beta, linear_beta, activate_left):
        import torch

        xf = x.float()
        nd = xf.dim()
        d = dim % nd
        if xf.numel() == 0:
            out_shape = (
                tuple(xf.shape[:d]) + (xf.shape[d] // 2,) + tuple(xf.shape[d + 1 :])
            )
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)
        h = xf.shape[d] // 2
        first, second = torch.split(xf, h, dim=d)
        gate, up = (first, second) if activate_left else (second, first)
        situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        up2 = linear_beta * torch.tanh(up / linear_beta) if linear_beta > 0.0 else up
        y = situ_a * up2
        if weight is not None:
            wshape = tuple(xf.shape[:d]) + (1,) * (nd - d)
            y = y * weight.float().reshape(wshape)
        return y.to(x.dtype)

    @staticmethod
    def _truncate(y, x, group_index, dim, group_list_type):
        """行主序前缀裁剪：与 golden 的 _apply_group_truncation 语义一致。"""
        if group_index is None:
            return y
        gi = group_index.reshape(-1)
        if gi.numel() == 0:
            return y
        rows = 1
        for s in x.shape[: dim % x.dim()]:
            rows *= int(s)
        if rows <= 0:
            return y
        if group_list_type == GROUP_LIST_TYPE_CUMSUM:
            real_bs = int(gi[-1].item())
        else:
            # Same reason as _real_bs: sum in Python integers, not in int64 (which would wrap).
            real_bs = sum(int(v) for v in gi.tolist())
        real_bs = max(0, min(real_bs, rows))
        if real_bs >= rows:
            return y
        split_d = y.numel() // rows
        if split_d == 0:
            # cut 轴为 0：输出本身为空，无前缀可截断（同 golden 侧守卫）。
            return y
        # real_bs == 0（总量为 0）时返回空前缀：算子不写出任何行。
        if dim % x.dim() == x.dim() - 1:
            return y.reshape(rows, split_d)[:real_bs]
        return y.reshape(-1)[: real_bs * split_d]

    def _ensure_compiled(self):
        import torch

        if self._compiled is not None:
            return
        for options in ({"mode": "reduce-overhead"}, {}):
            try:
                self._compiled = torch.compile(self._forward, **options)
                return
            except Exception:  # pragma: no cover - 老版本 torch 无该 mode
                continue
        self._compiled = self._forward

    def __call__(
        self,
        x,
        weight=None,
        group_index=None,
        dim=None,
        beta=None,
        linear_beta=None,
        activate_left=None,
        group_list_type=None,
        out=None,
        *args,
        **kwargs,
    ):
        import torch

        del out, args
        weight = kwargs.pop("weightOptional", weight)
        group_index = kwargs.pop("groupIndexOptional", group_index)
        linear_beta = kwargs.pop("linearBeta", linear_beta)
        activate_left = kwargs.pop("activateLeft", activate_left)
        group_list_type = kwargs.pop("groupListType", group_list_type)
        kwargs.pop("out", None)
        if kwargs:
            self._merge(kwargs)
        attrs = dict(self._attrs)
        if dim is not None:
            attrs["dim"] = int(dim)
        if beta is not None:
            attrs["beta"] = float(beta)
        if linear_beta is not None:
            attrs["linear_beta"] = float(linear_beta)
        if activate_left is not None:
            attrs["activate_left"] = bool(activate_left)
        if group_list_type is not None:
            attrs["group_list_type"] = _resolve_group_list_type(group_list_type)
        else:
            attrs["group_list_type"] = _resolve_group_list_type(
                attrs["group_list_type"]
            )
        del kwargs

        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.as_tensor(weight, device=x.device)
            elif weight.device != x.device:
                weight = weight.to(x.device)
        if group_index is not None and not isinstance(group_index, torch.Tensor):
            group_index = torch.as_tensor(group_index, device=x.device)

        self._ensure_compiled()
        call_args = (
            x,
            weight,
            attrs["dim"],
            attrs["beta"],
            attrs["linear_beta"],
            attrs["activate_left"],
        )
        try:
            y = self._compiled(*call_args)
        except Exception:  # pragma: no cover - 编译失败永久回落 eager
            self._compiled = self._forward
            y = self._forward(*call_args)
        return [
            self._truncate(y, x, group_index, attrs["dim"], attrs["group_list_type"])
        ]


class SituGluGroupKernelSpec:
    """kernel + geir 共用（输入 numpy.ndarray）。参数名对齐 op_host/situ_glu_group_def.cpp。"""

    def golden(x, weight=None, group_index=None, **kwargs):
        dim = int(kwargs.get("dim", -1))
        beta = float(kwargs.get("beta", 1.0))
        linear_beta = float(kwargs.get("linear_beta", 0.0))
        activate_left = bool(kwargs.get("activate_left", True))
        group_list_type = _resolve_group_list_type(
            kwargs.get("group_list_type", GROUP_LIST_TYPE_CUMSUM)
        )
        od = kwargs.get("output_dtypes") or []
        target = od[0] if len(od) > 0 else str(np.asarray(x).dtype)
        if isinstance(target, (list, tuple)):
            target = target[0]
        y = _ref(x, weight, dim, beta, linear_beta, activate_left, target)
        return _apply_group_truncation(y, x, group_index, dim, group_list_type)

    def customize_inputs(x, weight=None, group_index=None, **kwargs):
        _fill_group_index(
            x,
            group_index,
            int(kwargs.get("dim", -1)),
            kwargs.get("testcase_name", ""),
            kwargs.get("group_index_mode"),
            _resolve_group_list_type(
                kwargs.get("group_list_type", GROUP_LIST_TYPE_CUMSUM)
            ),
        )
        return (x, weight, group_index)

    def pre_compare(npu_out, golden_out):
        return _group_trunc_pre_compare(npu_out, golden_out)

    third_party = {"torch": _SituGluGroupTorchCompose}
    tolerance = _TOL_CROSS_CHECK


class SituGluGroupAclnnSpec:
    """aclnnSituGluGroup（autogen 接口）: (x, weightOptional, groupIndexOptional, dim, beta, linearBeta, activateLeft, out)。"""

    def golden(
        x,
        weightOptional=None,
        groupIndexOptional=None,
        dim=-1,
        beta=1.0,
        linearBeta=0.0,
        activateLeft=True,
        groupListType=GROUP_LIST_TYPE_CUMSUM,
        out=None,
        **kwargs,
    ):
        del out  # 输出由框架分配/回填
        group_list_type = _resolve_group_list_type(groupListType)
        x_np = _to_np(x)
        y = _ref(
            x_np,
            weightOptional,
            int(dim),
            float(beta),
            float(linearBeta),
            bool(activateLeft),
            x_np.dtype,
        )
        return _apply_group_truncation(
            y, x_np, groupIndexOptional, int(dim), group_list_type
        )

    def customize_inputs(
        x,
        weightOptional=None,
        groupIndexOptional=None,
        dim=-1,
        beta=1.0,
        linearBeta=0.0,
        activateLeft=True,
        groupListType=GROUP_LIST_TYPE_CUMSUM,
        out=None,
        **kwargs,
    ):
        _fill_group_index(
            x,
            groupIndexOptional,
            int(dim),
            kwargs.get("testcase_name", ""),
            kwargs.get("group_index_mode"),
            _resolve_group_list_type(groupListType),
        )
        return (x, weightOptional, groupIndexOptional)

    def pre_compare(npu_out, golden_out):
        return _group_trunc_pre_compare(npu_out, golden_out)

    third_party = {"torch": _SituGluGroupTorchCompose}
    tolerance = _TOL_CROSS_CHECK


class SituGluGroupE2ESpec:
    """cann_ops_nn.situ_glu_group E2E（单算子 / TorchAir），输入为 torch.Tensor。"""

    def golden(
        x,
        weight=None,
        group_index=None,
        dim=-1,
        beta=1.0,
        linear_beta=0.0,
        activate_left=True,
        group_list_type=GROUP_LIST_TYPE_CUMSUM,
        **kwargs,
    ):
        target = x.dtype if hasattr(x, "dtype") else _to_np(x).dtype
        group_list_type = _resolve_group_list_type(group_list_type)
        y = _ref(
            x,
            weight,
            int(dim),
            float(beta),
            float(linear_beta),
            bool(activate_left),
            target,
        )
        return _apply_group_truncation(y, x, group_index, int(dim), group_list_type)

    def customize_inputs(
        x,
        weight=None,
        group_index=None,
        dim=-1,
        beta=1.0,
        linear_beta=0.0,
        activate_left=True,
        group_list_type=GROUP_LIST_TYPE_CUMSUM,
        **kwargs,
    ):
        del beta, linear_beta, activate_left
        _fill_group_index(
            x,
            group_index,
            int(dim),
            kwargs.get("testcase_name", ""),
            kwargs.get("group_index_mode"),
            _resolve_group_list_type(group_list_type),
        )
        return (x, weight, group_index)

    def pre_compare(npu_out, golden_out):
        return _group_trunc_pre_compare(npu_out, golden_out)

    third_party = {"torch": _SituGluGroupTorchCompose}
    tolerance = _TOL_CROSS_CHECK


# TTK 3.0.0 的 kernel/aclnn 通路通过 __golden__/__input__ 解析标杆（与 swiglu_group 等兄弟算子
# 一致）；上面的 __spec__ 保留给较新的 spec 通路。缺少这两个注册时 TTK 会把 golden 判为
# UNSUPPORTED，且不会调用 customize_inputs 填充 group_index（cumsum 用随机数据会直接触发
# 非法输入 trap）。
def situ_glu_group_kernel_golden(
    x,
    weight=None,
    group_index=None,
    dim=-1,
    beta=1.0,
    linear_beta=0.0,
    activate_left=True,
    group_list_type=GROUP_LIST_TYPE_CUMSUM,
    **kwargs,
):
    kwargs.update(
        dim=dim,
        beta=beta,
        linear_beta=linear_beta,
        activate_left=activate_left,
        group_list_type=group_list_type,
    )
    return SituGluGroupKernelSpec.golden(x, weight, group_index, **kwargs)[0]


def situ_glu_group_kernel_input(
    x,
    weight=None,
    group_index=None,
    dim=-1,
    beta=1.0,
    linear_beta=0.0,
    activate_left=True,
    group_list_type=GROUP_LIST_TYPE_CUMSUM,
    **kwargs,
):
    kwargs.update(
        dim=dim,
        beta=beta,
        linear_beta=linear_beta,
        activate_left=activate_left,
        group_list_type=group_list_type,
    )
    return list(
        SituGluGroupKernelSpec.customize_inputs(x, weight, group_index, **kwargs)
    )


def aclnn_situ_glu_group_golden(
    x,
    weightOptional=None,
    groupIndexOptional=None,
    dim=-1,
    beta=1.0,
    linearBeta=0.0,
    activateLeft=True,
    groupListType=GROUP_LIST_TYPE_CUMSUM,
    out=None,
    yOut=None,
    **kwargs,
):
    del yOut
    return SituGluGroupAclnnSpec.golden(
        x,
        weightOptional,
        groupIndexOptional,
        dim,
        beta,
        linearBeta,
        activateLeft,
        groupListType,
        out,
        **kwargs,
    )[0]


def aclnn_situ_glu_group_input(
    x,
    weightOptional=None,
    groupIndexOptional=None,
    dim=-1,
    beta=1.0,
    linearBeta=0.0,
    activateLeft=True,
    groupListType=GROUP_LIST_TYPE_CUMSUM,
    out=None,
    yOut=None,
    **kwargs,
):
    del yOut, out
    return list(
        SituGluGroupAclnnSpec.customize_inputs(
            x,
            weightOptional,
            groupIndexOptional,
            dim,
            beta,
            linearBeta,
            activateLeft,
            groupListType,
            None,
            **kwargs,
        )
    )


__golden__ = {
    "kernel": {"situ_glu_group": "situ_glu_group_kernel_golden"},
    "aclnn": {"aclnnSituGluGroup": "aclnn_situ_glu_group_golden"},
}
__input__ = {
    "kernel": {"situ_glu_group": "situ_glu_group_kernel_input"},
    "aclnn": {"aclnnSituGluGroup": "aclnn_situ_glu_group_input"},
}
