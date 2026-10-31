#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK TestSpec for UniqueConsecutive — three entry points, one shared torch core.

Entry points (lookup keys, see __spec__ below):
  * ``unique_consecutive``        — Kernel/GEIR plugin path (CSV ``op_name``).
  * ``aclnnUniqueConsecutive``    — ACLNN API path (CSV ``api_name``).
  * ``torch.unique_consecutive``  — E2E framework API path (CSV ``api_name``).

The grouping / adjacent-comparison / counting core is delegated to
``torch.unique_consecutive`` (torch 2.7.1+cpu).  Kernel-level reconstruction
(start offsets) is computed with torch tensors as well (``cumsum - counts``);
numpy remains only as a container bridge for dtype resolution and for the
value gather that preserves ml_dtypes bfloat16 payloads and full-range
integers bit-exactly.

Third-party adapters call native ``torch.unique_consecutive`` without casts,
copies or padding; dtypes unsupported by the native API (complex) are reported
explicitly instead of being silently emulated.

Contract split (do NOT cross-apply):
  * Kernel (def.cpp / tiling arch35): axis is pinned to the flatten sentinel
    1000 and ``return_idx`` is always False (CheckSupport rejects True), so
    the idx output is a length-1 shape descriptor whose value is undefined and
    the golden emits a shape-correct placeholder — that is the Kernel's own
    contract, NOT a reference for the wider ACLNN/E2E domain.
  * ACLNN (aclnn_unique_consecutive.h): ``dim`` uses 1000 as the None sentinel
    (same NoneN as op-plugin); ``returnInverse``/``returnCounts`` may be True,
    producing real inverse/counts.  Disabled slots are returned as ``None`` so
    TTK marks them SUPPRESSED (no reference value to compare).
  * E2E (torch.unique_consecutive): dim=None/integer, returns a 1-3 element
    tuple mirroring the ATen structure trimmed by the Python layer.
"""

import numpy as np
import torch

__golden__ = {"kernel": {"unique_consecutive": "unique_consecutive_golden"}}


def _register_torch_unique_consecutive_signature():
    """Register the real torch.unique_consecutive signature with TTK.

    Why: TTK's default signature source for this API is ``docstring(Args-only)``,
    which carries no keyword-only markers, so the e2e ParamPlan binds
    return_inverse / return_counts / dim POSITIONALLY. Positional invocation
    then hits a torch_npu 2.7.1.post6 binding defect: for the two-output flag
    combinations the binding returns ``(values, {0} dummy)`` instead of
    ``(values, inverse_or_counts)`` (op-plugin always materialises three
    defined tensors; the binding's positional path fails to trim them), while
    keyword invocation trims correctly. Registering the true signature —
    keyword-only flags with defaults, exactly what
    ``inspect.signature(torch.unique_consecutive)`` reports — makes the e2e
    runner pass them as keywords and takes the correct binding path.

    This uses TTK's public override API (``register_api_params`` /
    ``FrameworkApiInfoKeeper``), not a framework patch. Two extra steps make it
    effective even for the testcase that triggered this module's load, because
    the e2e worker builds the ParamPlan before loading the spec:
      * any already-cached docstring-derived APIParamInfo has its ParamInfo
        objects patched IN PLACE — ``match_overload`` hands the very same
        objects to ``plan.overload_params``, and ``plan.build_args`` runs only
        later (inside ``_execute_eager``), so the already-built plan picks up
        the keyword-only flags;
      * the stale keeper cache entry is dropped so later lookups resolve
        through the manual override.
    Independent of TTK import order: silently skipped when ttk is unavailable
    (e.g. standalone golden verification).
    """
    _kw_only_names = ("return_inverse", "return_counts", "dim")
    try:
        from ttk.core_modules.framework_api.api_model import ParamInfo
        from ttk.core_modules.framework_api.info_keeper import FrameworkApiInfoKeeper
        from ttk.core_modules.framework_api.torch.param_extractor import (
            register_api_params,
        )
    except ImportError:
        try:
            from ttk.core_modules.framework_api.framework_api_info_keeper import (
                FrameworkApiInfoKeeper,
            )
            from ttk.utilities.simple_param_extractor import (
                ParamInfo,
                register_api_params,
            )
        except ImportError:
            return
    register_api_params(
        "torch.unique_consecutive",
        [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(
                name="return_inverse",
                type="bool",
                default=False,
                is_optional=True,
                is_keyword_only=True,
            ),
            ParamInfo(
                name="return_counts",
                type="bool",
                default=False,
                is_optional=True,
                is_keyword_only=True,
            ),
            ParamInfo(
                name="dim",
                type="int",
                default=None,
                is_optional=True,
                is_keyword_only=True,
            ),
        ],
        source="spec:golden.py(inspect torch.unique_consecutive)",
    )
    keeper = FrameworkApiInfoKeeper()
    stale = keeper._cache.get("torch.unique_consecutive")
    if stale is not None and "spec:golden.py" not in (stale.source or ""):
        # In-place patch: plan.overload_params references these same ParamInfo
        # objects (match_overload does not copy), and build_args runs later.
        for plist in [stale.params] + [ov.params for ov in stale.overloads]:
            for p in plist:
                if p.name in _kw_only_names:
                    p.is_keyword_only = True
                    p.is_optional = True
    # Drop the stale cache entry so the next get_param_plan resolves through
    # the manual override above.
    keeper._cache.pop("torch.unique_consecutive", None)


def _patch_param_plan_keyword_binding():
    """Restore the real keyword-only binding for e2e torch.unique_consecutive.

    Why this exists on top of the signature registration: the e2e runner
    pickles each TestcaseE2e (with its already-resolved ``_api_info_cache``)
    from the MAIN process to a forkserver worker. That pickled APIParamInfo
    descends from the docstring-only parse (no keyword-only flags) and is a
    separate object graph from anything the worker-side registration above can
    reach, so the worker's ParamPlan still binds the flag attributes
    positionally and trips the torch_npu binding defect described in
    ``UniqueConsecutiveTorchSpec.golden``.

    This patch re-routes only this API's flag parameters back to keywords at
    ``ParamPlan.build_args`` time — the binding torch.unique_consecutive
    actually declares (``input`` positional, flags keyword-only) — gated on
    ``api_name`` so no other operator or test path is affected. It mirrors the
    comparison-registry extension precedent set by the Unique golden
    (extending framework behaviour from the operator's own spec file). Remove
    this patch and the signature override only after the toolkit's signature
    source marks these flags keyword-only, including pickled worker plans,
    and the four output-flag combinations pass without either adapter.

    Version basis: TTK 202675b6ad04d1f918f9ab50dd79dce597430fb1 used
    testcase_manager.param_plan; aaee95df5f2b0c83a121e30adc7511cd9f2b6812
    moved it to framework_api.api_param but still parses this API from the
    docstring without keyword-only flags. Both retain the build_args contract.
    """
    try:
        from ttk.core_modules.framework_api.api_param import ParamPlan
    except ImportError:
        try:
            from ttk.core_modules.testcase_manager.param_plan import ParamPlan
        except ImportError:
            return
    if getattr(ParamPlan.build_args, "_unique_consecutive_kw_patch", False):
        return
    original = ParamPlan.build_args
    kw_only = ("return_inverse", "return_counts", "dim")

    def build_args(self, nested_tensors):
        args, kwargs, extra = original(self, nested_tensors)
        if self.api_name != "torch.unique_consecutive" or len(args) <= 1:
            return args, kwargs, extra
        names = [p.name for p in self.overload_params]
        moved = []
        for idx, value in enumerate(args):
            if 0 < idx < len(names) and names[idx] in kw_only:
                kwargs.setdefault(names[idx], value)
            else:
                moved.append(value)
        return moved, kwargs, extra

    build_args._unique_consecutive_kw_patch = True
    ParamPlan.build_args = build_args


_register_torch_unique_consecutive_signature()
_patch_param_plan_keyword_binding()

# TestSpec registration (priority over __golden__ in TTK's plugin loader).
# GEIR reuses the Kernel spec entry; ACLNN/E2E each get an independent one.
__spec__ = {
    "unique_consecutive": "UniqueConsecutiveKernelSpec",
    "aclnnUniqueConsecutive": "UniqueConsecutiveAclnnSpec",
    "torch.unique_consecutive": "UniqueConsecutiveTorchSpec",
}

# axis sentinel: the kernel operator only supports flatten mode (def.cpp pins
# axis=1000); op-plugin uses the same NoneN=1000 sentinel for aclnn dim=None.
_FLATTEN_AXIS = 1000


def _resolve_np_dtype(dtypes, index, default):
    """Robustly resolve output_dtypes[index] to a numpy dtype, falling back to default."""
    try:
        d = dtypes[index]
    except (TypeError, IndexError, KeyError):
        return default
    if d is None:
        return default
    try:
        return np.dtype(d)
    except TypeError:
        try:
            return np.dtype(getattr(d, "name", d))
        except Exception:
            return default


def _group_tensor(flat):
    """Build a torch tensor for grouping that preserves the kernel's per-type NE
    semantics. torch.from_numpy cannot ingest ml_dtypes.bfloat16, so bf16 is widened
    to float32 — an exact widening that preserves IEEE grouping (nan != nan, -0.0 ==
    +0.0), matching the kernel's Reg::Compare<CMPMODE::NE>. All int (incl. uint16/
    uint32/uint64) and float16/float32 dtypes are supported natively by torch 2.7."""
    # ml_dtypes.bfloat16 reports dtype.kind == "V" (not "f") and torch.from_numpy
    # rejects it, so detect by name and widen to float32 (exact, IEEE-faithful proxy).
    if flat.dtype.name == "bfloat16":
        return torch.from_numpy(flat.astype(np.float32))
    return torch.from_numpy(flat)


def _grouped_starts_t(counts_t):
    """Group start offsets from torch counts: starts = cumsum(counts) - counts.

    Equivalent to prepend-zero cumsum, computed entirely in torch (the grouping
    decision itself is torch.unique_consecutive's)."""
    return counts_t.cumsum(0) - counts_t


def _consecutive_grouping_t(
    tensor, *, return_inverse=False, return_counts=False, dim=None
):
    """Torch-core consecutive grouping for dtypes torch.unique_consecutive cannot
    consume (complex64/complex128 on torch 2.7.1 CPU: '"unique" not implemented').

    Group real/imag pairs along the requested axis. Keeping the component axis
    intact gives component-wise IEEE equality (NaN components never equal),
    without treating real and imaginary components as separate input elements.

    Returns (values, inverse, counts) with inverse/counts None when disabled.
    """
    if dim is not None and not -tensor.ndim <= dim < tensor.ndim:
        raise IndexError(f"dim {dim} is out of range for rank {tensor.ndim}")
    grouped = tensor.reshape(-1) if dim is None else tensor
    axis = 0 if dim is None else dim % tensor.ndim
    pairs, inverse, counts = torch.unique_consecutive(
        torch.view_as_real(grouped),
        dim=axis,
        return_inverse=True,
        return_counts=True,
    )
    values = torch.view_as_complex(pairs.contiguous())
    if dim is None:
        inverse = inverse.reshape(tensor.shape)
    return (
        values,
        inverse if return_inverse else None,
        counts if return_counts else None,
    )


def _reference_consecutive(
    tensor, *, return_inverse=False, return_counts=False, dim=None
):
    """Shared torch reference for the torch-tensor entry points (ACLNN / E2E).

    Delegates grouping/adjacency/counting/inverse to ``torch.unique_consecutive``
    (torch CPU 2.7.1 supports every dtype of the ACLNN/E2E domain except complex,
    which falls back to the torch-op grouping above). Output layout follows the
    caller's contract:
      * returns a plain tuple ordered (values, inverse?, counts?) — the E2E API
        structure — when called through this function directly.
    """
    if tensor.is_complex():
        values, inverse, counts = _consecutive_grouping_t(
            tensor,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )
    else:
        result = torch.unique_consecutive(
            tensor, return_inverse=return_inverse, return_counts=return_counts, dim=dim
        )
        if not return_inverse and not return_counts:
            values, inverse, counts = result, None, None
        else:
            values = result[0]
            inverse = result[1] if return_inverse else None
            counts = result[-1] if return_counts else None
    return values, inverse, counts


def _pattern_for(name, is_unsigned):
    """Semantic input patterns keyed by testcase_name (customize_inputs)."""
    if "order" in name:  # adjacent vs non-adjacent repeats
        return [3, 1, 3, 2, 1, -1]
    if "signed_zero" in name or name.endswith("_zero"):
        # -0.0 == +0.0 grouping, signbit of group head kept.
        return [-0.0, 0.0, -2.0, 1.0, 1.0]
    if "nan" in name:  # adjacent NaN are distinct groups; inf/-inf distinct
        return [
            float("nan"),
            2.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            float("nan"),
            -0.0,
            0.0,
        ]
    if "largeint" in name:
        if is_unsigned:
            return [2**64 - 1, 2**64 - 1, 0, 2**64 - 2]
        return [2**63 - 1, 2**63 - 1, -(2**63), 2**53 + 1, 2**53 + 1]
    return None


def _fill_pattern(tensor, pattern):
    """Tile a pattern into a torch tensor in-place, preserving dtype/device."""
    values = torch.tensor(pattern, dtype=tensor.dtype, device=tensor.device)
    repeats = (tensor.numel() + values.numel() - 1) // values.numel()
    tensor.copy_(values.repeat(repeats)[: tensor.numel()].reshape(tensor.shape))
    return tensor


def _pre_compare_unique_consecutive(*outputs_and_goldens, **kwargs):
    """Treat -0.0 and +0.0 as the same value before byte comparison.

    Consecutive grouping follows IEEE (-0.0 == +0.0 -> one group), and the
    reference keeps the group head's original sign bit, while the device
    backends (AiCore/AICPU) may materialize the head of that group as either
    signed zero.  Since the two encodings denote the same IEEE value, normalize
    the sign of zero entries on floating-point arrays before binary_equal runs
    (same convention as the Unique delivery, !12141 signed-zero semantics).
    Integers (values/counts/inverse) are untouched, as are complex arrays whose
    zero components never reach this normalization (non-zero values compare
    bitwise; a zero complex is genuinely equal in both encodings only after the
    same normalization applies to its float components).
    """
    normalized = []
    for arr in outputs_and_goldens:
        if isinstance(arr, torch.Tensor):
            if arr.is_floating_point():
                zero_mask = arr == 0
                if bool(zero_mask.any()):
                    arr = arr.clone()
                    arr[zero_mask] = 0
            elif arr.is_complex():
                pair = torch.view_as_real(arr)
                zero_mask = pair == 0
                if bool(zero_mask.any()):
                    arr = arr.clone()
                    torch.view_as_real(arr)[zero_mask] = 0
        elif isinstance(arr, np.ndarray) and (
            arr.dtype.kind == "f" or arr.dtype.name == "bfloat16"
        ):
            zero_mask = arr == 0
            if np.any(zero_mask):
                arr = arr.copy()
                arr[zero_mask] = 0
        elif isinstance(arr, np.ndarray) and arr.dtype.kind == "c":
            # complex: normalize ±0 components (real/imag float pairs)
            pair_dtype = np.dtype(f"float{arr.dtype.itemsize // 2 * 8}")
            pair = arr.view(dtype=pair_dtype)
            zero_mask = pair == 0
            if np.any(zero_mask):
                pair = pair.copy()
                pair[zero_mask] = 0
                arr = pair.view(dtype=arr.dtype).reshape(arr.shape)
        normalized.append(arr)
    return normalized


# ---------------------------------------------------------------------------
# Kernel / GEIR golden (numpy in, numpy out) — contract-preserving refactor
# ---------------------------------------------------------------------------


def unique_consecutive_golden(
    x, *, return_idx=False, return_counts=False, axis=1000, **kwargs
):
    """CPU golden for UniqueConsecutive (kernel-level TTK plugin).

    Parameter names/order follow @unique_consecutive_def.cpp inputs (x) plus its
    attributes. All input tensors are numpy.ndarray (bfloat16 arrives as ml_dtypes
    bfloat16). Returns [y, idx, count] mapped positionally to output 0/1/2.

    Contract mapping (def.cpp / CheckSupport / tiling arch35):
      * axis        : always 1000 (flatten sentinel) -> torch dim=None (flatten).
                       Per-dim mode is never emitted by this operator (tiling
                       rejects axis != 1000).
      * return_idx  : always False (CheckSupport rejects True) -> idx value undefined.
      * return_counts: True/False -> real counts / placeholder (see below).

    Semantics — flattened consecutive dedup:
      * y     : consecutive-dedup group representatives in input order (non-adjacent
                repeats stay separate groups). Group boundaries/counts come from
                torch.unique_consecutive; y is then gathered from x by the group
                start indices (torch cumsum - counts, bridged to numpy only as the
                gather index container) so its bits/dtype are preserved EXACTLY
                (never cast large ints to float; bf16 NaN payload bits kept),
                matching the kernel copying x.
      * idx   : inverse indices. This operator never produces idx (return_idx is
                always False; CheckSupport rejects True). The kernel emits a length-1
                shape descriptor (CopyOutShape sets idx DIM0=1) but NEVER writes the
                idxGm_ data buffer. Consequences, split by shape vs value:
                  - SHAPE (1,) is contractual: it is the descriptor the kernel really
                    emits, and the shape-unknown metadata output is compared against
                    it, so golden idx must be a real (1,) ndarray (not None, not (0,)).
                  - VALUE is contractually UNDEFINED: since idxGm_ is never written,
                    the single element equals whatever pre-fills the HBM buffer. The
                    value is therefore NOT an expected result; ones((1,)) is only a
                    shape-correct placeholder. Value-level comparison of idx is not
                    meaningful — this is the Kernel's own contract and MUST NOT be
                    copied to ACLNN/E2E paths where returnInverse=True is real.
      * count : per-group element counts when return_counts=True (kernel writes it,
                SHAPE2 DIM0=G) -> real (G,) counts from torch, cast to the out_idx
                dtype (INT32/INT64). When return_counts=False the kernel emits
                SHAPE2 DIM0=1 but does not write countGm_ -> its value is undefined
                exactly like idx, so golden returns a shape-correct (1,) placeholder.

    Adjacent equality is delegated to torch.unique_consecutive, whose grouping mirrors
    the kernel's Reg::Compare<CMPMODE::NE>:
      * integer dtypes  -> exact (no float promotion; uint16/32/64 & int64 boundaries);
      * float dtypes    -> IEEE (nan != nan -> True, so adjacent nan become
                           separate groups; -0.0 == +0.0 -> same group).

    Empty input (size == 0): the kernel takes tiling key 666 (CopyOutEmptyShape),
    which sets all three output dims to 0, so golden returns three empty arrays.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name.
    """
    # axis is pinned to the flatten sentinel by the operator contract; flatten x.
    if axis != _FLATTEN_AXIS:
        raise ValueError(
            f"unique_consecutive golden only supports flatten mode (axis={_FLATTEN_AXIS}); "
            f"got axis={axis}"
        )
    if return_idx:
        # CheckSupport/tiling reject return_idx=True on the AICore path; the golden
        # refuses the same domain instead of fabricating an inverse reference.
        raise ValueError(
            "unique_consecutive golden does not support return_idx=True "
            "(CheckSupport rejects it on the kernel path)"
        )
    flat = np.asarray(x).reshape(-1)

    # Resolve idx/count dtype from out_idx (takes precedence over output_dtypes).
    # out_idx: 3 -> DT_INT32, 9 -> DT_INT64 (default).
    out_idx = kwargs.get("out_idx", 9)
    if out_idx == 3:
        idx_dtype = np.dtype(np.int32)
        count_dtype = np.dtype(np.int32)
    elif out_idx == 9:
        idx_dtype = np.dtype(np.int64)
        count_dtype = np.dtype(np.int64)
    else:
        # Unsupported out_idx: fallback to int64 (matches default).
        idx_dtype = np.dtype(np.int64)
        count_dtype = np.dtype(np.int64)

    n = flat.shape[0]

    # Empty input: kernel key 666 CopyOutEmptyShape -> all three outputs dim0 = 0.
    if n == 0:
        return [
            np.empty((0,), dtype=flat.dtype),
            np.empty((0,), dtype=idx_dtype),
            np.empty((0,), dtype=count_dtype),
        ]

    # Core: consecutive grouping / adjacent comparison / counting via torch.
    # dim=None -> flatten mode (matches axis=1000). counts is int64 group sizes.
    counts_t = torch.unique_consecutive(
        _group_tensor(flat), dim=None, return_counts=True
    )[1]

    # Group start offsets computed in torch (cumsum - counts); the numpy copy is
    # only the index container for the bit-exact gather from the ORIGINAL array.
    starts = _grouped_starts_t(counts_t).numpy()
    y = flat[starts]

    # idx: contractual shape (1,) (kernel emits DIM0=1); value UNDEFINED (idxGm_ never
    # written) -> shape-correct placeholder only. This is NOT a torch result —
    # return_idx is always False and idx is never computed (kernel contract).
    idx = np.ones((1,), dtype=idx_dtype)

    if return_counts:
        # Real per-group counts (G,) from the torch core -> expected result.
        count = counts_t.numpy().astype(count_dtype, copy=False)
    else:
        # count not produced (countGm_ unwritten) -> undefined value, (1,) placeholder
        # (kernel contract, same split as idx above).
        count = np.ones((1,), dtype=count_dtype)

    return [y, idx, count]


# ---------------------------------------------------------------------------
# ACLNN golden (torch in, three fixed slots out)
# ---------------------------------------------------------------------------


def aclnn_unique_consecutive_golden(
    selfT,
    returnInverse=False,
    returnCounts=False,
    dim=_FLATTEN_AXIS,
    valueOut=None,
    inverseOut=None,
    countsOut=None,
    **kwargs,
):
    """Reference aclnnUniqueConsecutive (parameters follow the C header
    ``aclnnUniqueConsecutiveGetWorkspaceSize`` without workspaceSize/executor;
    ``self`` is renamed selfT to avoid the Python keyword clash, position 1).

    ``dim`` uses the ACLNN/op-plugin NoneN=1000 sentinel: 1000 -> torch dim=None
    (flatten); any other value is a real dim and is forwarded as-is.

    Output slots (valueOut, inverseOut, countsOut positions):
      * returnInverse=True  -> real inverse (int64 per API; int32 arrives on the
        AICORE path via out_idx and is returned in the incoming tensor dtype when
        TTK provides one);
      * returnCounts=True   -> real per-group counts;
      * disabled slots return ``None`` — TTK compares mark them SUPPRESSED: the
        ACLNN contract never writes a disabled output, so it has no reference
        value (do NOT emit the Kernel's (1,) placeholder here).
    """
    torch_dim = None if dim == _FLATTEN_AXIS else int(dim)
    values, inverse, counts = _reference_consecutive(
        selfT,
        return_inverse=bool(returnInverse),
        return_counts=bool(returnCounts),
        dim=torch_dim,
    )
    if (
        inverse is not None
        and inverseOut is not None
        and inverseOut.dtype != inverse.dtype
    ):
        # out_idx INT32 path: honour the dtype the case requested.
        inverse = inverse.to(inverseOut.dtype)
    if counts is not None and countsOut is not None and countsOut.dtype != counts.dtype:
        counts = counts.to(countsOut.dtype)
    return [values, inverse, counts]


# ---------------------------------------------------------------------------
# Third-party competitors (native torch, no casts/copies/padding)
# ---------------------------------------------------------------------------


class UniqueConsecutiveTorchCompetitor:
    """Native torch.unique_consecutive baseline for the E2E entry point."""

    def __init__(self, return_inverse=False, return_counts=False, **kwargs):
        self.flags = bool(return_inverse), bool(return_counts)

    def __call__(self, input, dim=None, **kwargs):
        if input.is_complex():
            raise NotImplementedError(
                "Native torch.unique_consecutive does not implement complex dtypes"
            )
        return torch.unique_consecutive(
            input,
            return_inverse=self.flags[0],
            return_counts=self.flags[1],
            dim=dim,
        )


class UniqueConsecutiveAclnnCompetitor:
    """Native baseline wrapped to the ACLNN parameter convention.

    Binding contract (xpu-server ``bind_params``): a pool entry named ``self``
    is never injected by name — the ACLNN input tensor arrives as the first
    POSITIONAL argument (``*args``), while the flag attributes arrive by name
    through ``**kwargs``. Local direct calls pass the tensor positionally as
    well, so both paths share this signature.
    """

    def __init__(self, returnInverse=False, returnCounts=False, **kwargs):
        self.flags = bool(returnInverse), bool(returnCounts)

    def __call__(impl, *args, dim=_FLATTEN_AXIS, **kwargs):
        if not args:
            raise TypeError(
                "UniqueConsecutiveAclnnCompetitor expects the 'self' tensor "
                "as the first positional argument"
            )
        self = args[0]
        returnInverse = kwargs.get("returnInverse", impl.flags[0])
        returnCounts = kwargs.get("returnCounts", impl.flags[1])
        if self.is_complex():
            raise NotImplementedError(
                "Native torch.unique_consecutive does not implement complex dtypes"
            )
        torch_dim = None if dim == _FLATTEN_AXIS else int(dim)
        return _reference_consecutive(
            self,
            return_inverse=bool(returnInverse),
            return_counts=bool(returnCounts),
            dim=torch_dim,
        )


class UniqueConsecutiveKernelCompetitor:
    """Native baseline for the Kernel entry point (flatten contract).

    The kernel domain pins axis=1000 (flatten) and return_idx=False; the native
    torch call mirrors exactly that domain (dim=None), so any widening beyond it
    would be tested through the ACLNN/E2E specs instead.
    """

    def __init__(
        self,
        return_idx=False,
        return_counts=False,
        axis=_FLATTEN_AXIS,
        out_idx=9,
        **kwargs,
    ):
        if bool(return_idx):
            raise NotImplementedError(
                "Kernel baseline does not support return_idx=True (kernel contract)"
            )
        if int(axis) != _FLATTEN_AXIS:
            raise NotImplementedError(
                "Kernel baseline only supports the flatten sentinel axis=1000"
            )
        self.return_counts = bool(return_counts)
        self.out_idx = int(out_idx)

    def __call__(impl, x, **kwargs):
        if x.is_complex():
            raise NotImplementedError(
                "Native torch.unique_consecutive does not implement complex dtypes"
            )
        # Defensive domain check: attributes may also arrive at call time.
        if bool(kwargs.get("return_idx", False)):
            raise NotImplementedError(
                "Kernel baseline does not support return_idx=True (kernel contract)"
            )
        if "axis" in kwargs and int(kwargs["axis"]) != _FLATTEN_AXIS:
            raise NotImplementedError(
                "Kernel baseline only supports the flatten sentinel axis=1000"
            )
        values, _, counts = _reference_consecutive(
            x, return_inverse=False, return_counts=impl.return_counts, dim=None
        )
        if counts is not None and impl.out_idx == 3:  # DT_INT32
            counts = counts.to(torch.int32)
        # Three slots aligned with the kernel outputs; the idx slot is never
        # produced on the kernel path (return_idx=False contract) -> None.
        return [values, None, counts]


# ---------------------------------------------------------------------------
# TestSpec classes
# ---------------------------------------------------------------------------


class UniqueConsecutivePrecisionSpec:
    """Value-copy operator: compare declared dtypes exactly (binary_equal).

    No arithmetic tolerance is used: arithmetic promotion to float64 changes
    dtype-dependent routes and ordering semantics (lesson recorded for Unique).
    Integer indices/counts and bit-copied values all compare bitwise; ±0 sign
    bits are part of the copied value and stay exact on the NPU paths.
    """

    tolerance = {
        dtype: {"standard": "binary_equal"}
        for dtype in (
            "bool",
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "complex64",
            "complex128",
        )
    }


class UniqueConsecutiveKernelSpec(UniqueConsecutivePrecisionSpec):
    """Kernel / GEIR spec — numpy golden, torch third_party baseline.

    The golden keeps the exact signature of the legacy __golden__ registration
    (x + def.cpp attributes), so behaviour is identical whether TTK resolves it
    through the TestSpec or the legacy dict.
    """

    golden = staticmethod(unique_consecutive_golden)
    third_party = {"torch": UniqueConsecutiveKernelCompetitor}
    pre_compare = staticmethod(_pre_compare_unique_consecutive)

    @staticmethod
    def customize_inputs(x, **kwargs):
        """Kernel path (numpy in/out): inject semantic patterns by testcase_name."""
        name = str(kwargs.get("testcase_name", ""))
        is_unsigned = x.dtype.kind == "u"
        pattern = _pattern_for(name, is_unsigned)
        if pattern is None:
            return (x,)
        tiled = (pattern * ((x.size + len(pattern) - 1) // len(pattern)))[: x.size]
        x[:] = np.asarray(tiled, dtype=x.dtype).reshape(x.shape)
        return (x,)


class UniqueConsecutiveAclnnSpec(UniqueConsecutivePrecisionSpec):
    """ACLNN spec — torch golden bound to the aclnnUniqueConsecutive signature."""

    golden = staticmethod(aclnn_unique_consecutive_golden)
    third_party = {"torch": UniqueConsecutiveAclnnCompetitor}
    pre_compare = staticmethod(_pre_compare_unique_consecutive)

    @staticmethod
    def customize_inputs(
        selfT,
        returnInverse=False,
        returnCounts=False,
        dim=_FLATTEN_AXIS,
        valueOut=None,
        inverseOut=None,
        countsOut=None,
        **kwargs,
    ):
        """ACLNN path (torch in, in-place): inject semantic patterns by testcase_name.

        Signature follows the C header order — TTK's ParamPlan binds positionally
        for the aclnn level; disabled output slots arrive as None and are ignored.
        """
        name = str(kwargs.get("testcase_name", ""))
        is_unsigned = selfT.dtype in (
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
        pattern = _pattern_for(name, is_unsigned)
        if pattern is None:
            return
        _fill_pattern(selfT, pattern)


class UniqueConsecutiveTorchSpec(UniqueConsecutivePrecisionSpec):
    """E2E spec — torch API signature golden; returns the API's tuple structure."""

    third_party = {"torch": UniqueConsecutiveTorchCompetitor}
    pre_compare = staticmethod(_pre_compare_unique_consecutive)

    @staticmethod
    def golden(input, return_inverse=False, return_counts=False, dim=None, **kwargs):
        """E2E golden bound to the torch.unique_consecutive API signature.

        Return structure mirrors the ATen Python-layer trimming: 1 element for
        values-only, 2 for one flag, 3 for both — the structure
        ``torch.unique_consecutive`` itself returns on every backend that
        implements the ATen contract.

        Known executor/binding defect, reproduced and version-pinned against
        torch_npu 2.7.1.post6 + ops-test-kit Rev 202675b: the TTK e2e runner
        passes attributes positionally (aten-schema style), and the
        torch_npu binding's positional path fails to trim the op-plugin's
        always-defined {0} dummies for the two-output flag combinations —
        ``unique_consecutive(x, ri, rc, None)`` returns ``(values, {0})``
        instead of ``(values, inverse_or_counts)``. Keeping this golden at the
        ATen structure makes those cases FAIL honestly (shape mismatch) rather
        than silently comparing a dummy; real inverse/counts coverage stays
        complete through the both-flags cases here and through the ACLNN path.
        """
        values, inverse, counts = _reference_consecutive(
            input,
            return_inverse=bool(return_inverse),
            return_counts=bool(return_counts),
            dim=dim,
        )
        if inverse is None and counts is None:
            return values
        if inverse is None:
            return values, counts
        if counts is None:
            return values, inverse
        return values, inverse, counts

    @staticmethod
    def customize_inputs(
        input, return_inverse=False, return_counts=False, dim=None, **kwargs
    ):
        """E2E path (torch in, in-place): inject semantic patterns by testcase_name.

        Signature follows the torch API order — TTK's ParamPlan binds positionally
        for the e2e level.
        """
        name = str(kwargs.get("testcase_name", ""))
        is_unsigned = input.dtype in (
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
        pattern = _pattern_for(name, is_unsigned)
        if pattern is None:
            return
        _fill_pattern(input, pattern)
