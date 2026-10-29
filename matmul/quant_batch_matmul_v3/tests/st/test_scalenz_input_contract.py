# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU system test: WeightNZ + ScaleNZ through eager, ACLNN and native ACLNN Graph.

Run one path per process, e.g. python test_scalenz_input_contract.py --mode aclnn_graph.
QBMM_CUSTOM_OPAPI selects the built custom ACLNN library for direct ACLNN execution and native graph capture/replay.
QBMM_DESCRIPTOR_PROBE selects scalenz_descriptor_snapshot.cpp built against that package's opdev headers/libraries.
QBMM_TEST_OUTPUT_DTYPE selects fp32 (default), fp16 or bf16 for the same interface matrix.
"""

import argparse
import ctypes as ct
import itertools
import json
import math
import os
from pathlib import Path

import torch
import torch_npu

from scalenz_test_utils import AclnnInputRejected, is_input_rejection

OUTPUT_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
    os.environ.get("QBMM_TEST_OUTPUT_DTYPE", "fp32")
]


def fp4_pack(values):
    # E2M1 encodings for the exact test values {-2, -1, 0, 1, 2}.
    table = torch.tensor([12, 10, 0, 2, 4], dtype=torch.uint8)
    bits = table[(values + 2).long()]
    return (bits[..., 0::2] | (bits[..., 1::2] << 4)).contiguous()


def make_case(
    fp4,
    ta,
    tb,
    weight_nz,
    batch="none",
    equal_axes=False,
    bad_scale=False,
    scale_nz=True,
    byte_scale=False,
    mkn=None,
):
    m, k, n = mkn if mkn is not None else ((6, 194, 4) if equal_axes else (6, 130, 46))
    kg = (k + 63) // 64
    ba = (2,) if batch in ("a", "ab") else ()
    bb = (2,) if batch in ("b", "ab") else ()

    def values(shape, step):
        index = torch.arange(math.prod(shape)).reshape(shape)
        batch_phase = index // math.prod(shape[-2:])
        return ((index * step + batch_phase) % 5 - 2).float()

    def scales(shape, row_weight, column_weight, lane_weight):
        row = torch.arange(shape[-3]).view(-1, 1, 1)
        column = torch.arange(shape[-2]).view(1, -1, 1)
        lane = torch.arange(2).view(1, 1, 2)
        exponent = row * row_weight + column * column_weight + lane * lane_weight
        if len(shape) > 3:
            exponent = exponent + torch.arange(math.prod(shape[:-3])).reshape(
                shape[:-3] + (1, 1, 1)
            )
        return (126 + exponent % 3).to(torch.uint8)

    a, b = values(ba + (m, k), 3), values(bb + (k, n), 7)
    sa = scales(ba + (m, kg, 2), 1, 2, 1)
    # Distinct axis coefficients also detect an accidental K/N swap when Kg == N.
    sb = scales(bb + (kg, n, 2), 2, 1, 2)
    sa_f = (
        torch.pow(2.0, sa.float() - 127).flatten(-2).repeat_interleave(32, -1)[..., :k]
    )
    sb_f = (
        torch.pow(2.0, sb.float() - 127)
        .transpose(-1, -2)
        .flatten(-3, -2)
        .repeat_interleave(32, -2)[..., :k, :]
    )
    expected = ((a * sa_f) @ (b * sb_f)).to(OUTPUT_DTYPE)
    a_base = a.transpose(-1, -2).contiguous() if ta else a.contiguous()
    b_base = b.transpose(-1, -2).contiguous() if tb else b.contiguous()
    if fp4:
        x, w = fp4_pack(a_base).npu(), fp4_pack(b_base).npu()
    else:
        x, w = (
            a_base.to(torch.float8_e4m3fn).npu(),
            b_base.to(torch.float8_e4m3fn).npu(),
        )
    if weight_nz:
        # Packed FP4 uses the existing uint8 producer; the matmul call declares
        # the real FP4 dtype. No explicit FP4 format-cast extension is required.
        w = torch_npu.npu_format_cast(w, 29)
    if ta:
        x = x.transpose(-1, -2)
    if tb:
        w = w.transpose(-1, -2)
    if ta:
        sa = sa.transpose(-3, -2).contiguous()
    sa = (sa if byte_scale else sa.view(torch.float8_e8m0fnu)).npu()
    if ta:
        sa = sa.transpose(-3, -2)
    if bad_scale:
        sb = sb.transpose(-3, -2).contiguous()
    if not scale_nz and tb:
        sb = sb.transpose(-3, -2).contiguous()
    sb = (sb if byte_scale else sb.view(torch.float8_e8m0fnu)).npu()
    if scale_nz:
        cast = (
            {
                "customize_dtype": torch_npu.float8_e8m0fnu,
                "input_dtype": torch_npu.float8_e8m0fnu,
            }
            if byte_scale
            else {}
        )
        sb = torch_npu.npu_format_cast(sb, 29, **cast)
    elif tb:
        sb = sb.transpose(-3, -2)
    return (x, w, sb, sa), expected


def public_call(fp4):
    def call(x, w, sb, sa):
        kwargs = (
            {
                "x1_dtype": torch_npu.float4_e2m1fn_x2,
                "x2_dtype": torch_npu.float4_e2m1fn_x2,
            }
            if fp4
            else {}
        )
        return torch_npu.npu_quant_matmul(
            x,
            w,
            sb,
            pertoken_scale=sa,
            output_dtype=OUTPUT_DTYPE,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
            **kwargs,
        )

    return call


class DirectAclnn:
    def __init__(self):
        self.base = ct.CDLL("libopapi.so")
        self.api = ct.CDLL(os.environ["QBMM_CUSTOM_OPAPI"])
        self.base.aclCreateTensor.restype = ct.c_void_p
        self.base.aclCreateTensor.argtypes = [
            ct.POINTER(ct.c_int64),
            ct.c_uint64,
            ct.c_int,
            ct.POINTER(ct.c_int64),
            ct.c_int64,
            ct.c_int,
            ct.POINTER(ct.c_int64),
            ct.c_uint64,
            ct.c_void_p,
        ]
        self.base.aclDestroyTensor.argtypes = [ct.c_void_p]
        self.base.aclDestroyAclOpExecutor.argtypes = [ct.c_void_p]
        self.snapshot = None
        if os.environ.get("QBMM_DESCRIPTOR_PROBE"):
            self.probe = ct.CDLL(os.environ["QBMM_DESCRIPTOR_PROBE"])
            self.snapshot = self.probe.QbmmScaleDescriptorSnapshot
            self.snapshot.restype = ct.c_uint64
            self.snapshot.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int64), ct.c_uint64]

    def snapshot_descriptor(self, descriptor):
        if self.snapshot is None:
            raise RuntimeError(
                "QBMM_DESCRIPTOR_PROBE is required for descriptor immutability checks"
            )
        values = (ct.c_int64 * 128)()
        count = self.snapshot(descriptor, values, len(values))
        assert 0 < count <= len(values), count
        return tuple(values[:count])

    def descriptor(self, t, dtype, private=False, fp4=False, fault=None):
        shape, stride = list(t.shape), list(t.stride())
        fmt = 29 if private else 2
        if fault == "base_format":
            fmt = int(torch_npu.get_npu_format(t))
        if private:
            # These tests format-cast dense producer inputs. Check the actual allocation
            # against the producer's physical layout; no TorchAir dependency is needed.
            if dtype == 37:  # E8M0 ScaleNZ uses three logical trailing axes.
                storage = shape[:-3] + [(shape[-2] + 15) // 16, shape[-3], 16, 2]
            else:  # FP8 / packed-uint8 WeightNZ producer has C0=32 bytes.
                storage = shape[:-2] + [
                    (shape[-1] + 31) // 32,
                    (shape[-2] + 15) // 16,
                    16,
                    32,
                ]
            assert math.prod(storage) == torch_npu.get_storage_size(t)
        else:
            storage = shape.copy()
        if fp4:
            shape[-1] *= 2
            stride = [s * 2 for s in stride[:-1]] + [1]
            storage[-1] *= 2

        def arr(vals):
            return (ct.c_int64 * len(vals))(*vals)

        offset = 1 if fault == "offset" else 0
        if fault == "stride":
            stride[-2] += 1
        if fault == "storage":
            storage[-1] = 1
        if fault == "dtype":
            dtype = 0
        result = self.base.aclCreateTensor(
            arr(shape),
            len(shape),
            dtype,
            arr(stride),
            offset,
            fmt,
            arr(storage),
            len(storage),
            t.data_ptr(),
        )
        if not result:
            raise RuntimeError("aclCreateTensor failed")
        return result

    def call(
        self,
        args,
        expected,
        fp4,
        ta,
        tb,
        weight_nz,
        capture=False,
        fault=None,
        scale_nz=True,
        check_descriptors=False,
    ):
        torch.npu.synchronize()  # Flush Torch producers before submitting raw ACLNN work.
        x, w, sb, sa = args
        x = x.transpose(-1, -2) if ta else x
        w = w.transpose(-1, -2) if tb else w
        sa = sa.transpose(-3, -2) if ta else sa
        sb = sb.transpose(-3, -2) if tb and not scale_nz else sb
        y = torch.empty_like(expected, device="npu")
        tensors = []
        try:
            a_type = 35 if fault in ("a_e5m2", "ab_e5m2") else (40 if fp4 else 36)
            b_type = 35 if fault in ("b_e5m2", "ab_e5m2") else (40 if fp4 else 36)
            tensors = [
                self.descriptor(x, a_type, fp4=fp4),
                self.descriptor(w, b_type, private=weight_nz, fp4=fp4),
                self.descriptor(
                    sa,
                    37,
                    fault="base_format"
                    if fault in ("scale_a_format", "group_ncl")
                    else None,
                ),
                self.descriptor(sb, 37, private=scale_nz, fault=fault),
                self.descriptor(
                    y,
                    {torch.float32: 0, torch.float16: 1, torch.bfloat16: 27}[
                        expected.dtype
                    ],
                ),
            ]
            prefix = "aclnnQuantMatmulWeightNz" if weight_nz else "aclnnQuantMatmulV5"
            query, execute = (
                getattr(self.api, prefix + "GetWorkspaceSize"),
                getattr(self.api, prefix),
            )
            query.argtypes = [ct.c_void_p] * 9 + [
                ct.c_bool,
                ct.c_bool,
                ct.c_int64,
                ct.c_void_p,
                ct.POINTER(ct.c_uint64),
                ct.POINTER(ct.c_void_p),
            ]
            query.restype = execute.restype = ct.c_int
            execute.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_void_p]
            workspace, executor = ct.c_uint64(), ct.c_void_p()
            before = (
                [self.snapshot_descriptor(t) for t in tensors[2:4]]
                if check_descriptors
                else None
            )
            status = query(
                *tensors[:4],
                None,
                None,
                None,
                None,
                None,
                ta,
                tb,
                (1 << 32) | (1 << 16) | (64 if fault in ("group", "group_ncl") else 32),
                tensors[-1],
                ct.byref(workspace),
                ct.byref(executor),
            )
            if check_descriptors:
                assert before == [self.snapshot_descriptor(t) for t in tensors[2:4]], (
                    "Query modified input scales"
                )
                first_status, first_executor = status, executor
                workspace, executor = ct.c_uint64(), ct.c_void_p()
                status = query(
                    *tensors[:4],
                    None,
                    None,
                    None,
                    None,
                    None,
                    ta,
                    tb,
                    (1 << 32)
                    | (1 << 16)
                    | (64 if fault in ("group", "group_ncl") else 32),
                    tensors[-1],
                    ct.byref(workspace),
                    ct.byref(executor),
                )
                assert before == [self.snapshot_descriptor(t) for t in tensors[2:4]], (
                    "Repeated query modified input scales"
                )
                assert status == first_status, "Repeated query changed status"
                if first_status == 0:
                    assert self.base.aclDestroyAclOpExecutor(first_executor) == 0
            if status:
                raise AclnnInputRejected(prefix, status)
            memory = torch.empty(workspace.value, dtype=torch.uint8, device="npu")
            stream = ct.c_void_p(torch.npu.current_stream().npu_stream)

            def run():
                status = execute(
                    memory.data_ptr() if workspace.value else None,
                    workspace.value,
                    executor,
                    stream,
                )
                if status:
                    raise RuntimeError(f"{prefix} failed: {status}")

            if not capture:
                run()
                torch.npu.synchronize()
                return y
            runtime = ct.CDLL("libascendcl.so")
            runtime.aclmdlRICaptureBegin.argtypes = [ct.c_void_p, ct.c_int]
            runtime.aclmdlRICaptureEnd.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
            runtime.aclmdlRIExecuteAsync.argtypes = [ct.c_void_p, ct.c_void_p]
            runtime.aclmdlRIDestroy.argtypes = [ct.c_void_p]

            def check(status):
                if status:
                    raise RuntimeError(f"ACLNN Graph runtime failed: {status}")

            model = ct.c_void_p()
            # Query, allocation, format cast and Torch producers have all completed.
            check(runtime.aclmdlRICaptureBegin(stream, 0))
            try:
                run()
            finally:
                check(runtime.aclmdlRICaptureEnd(stream, ct.byref(model)))
            try:
                for _ in range(2):
                    # Clearing the destination proves that each replay performs computation.
                    y.zero_()
                    torch.npu.synchronize()
                    check(runtime.aclmdlRIExecuteAsync(model, stream))
                    torch.npu.synchronize()
                    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=1e-3)
                return y
            finally:
                check(runtime.aclmdlRIDestroy(model))
        finally:
            for t in tensors:
                self.base.aclDestroyTensor(t)


def resolved_api_library(name):
    """Record the real op-plugin symbol target so tests cannot silently use the stock API."""
    lib = ct.CDLL(str(Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"))
    getter = getattr(lib, "_Z16GetOpApiFuncAddrPKc")
    getter.restype, getter.argtypes = ct.c_void_p, [ct.c_char_p]
    pointer = getter((name + "GetWorkspaceSize").encode())

    class DlInfo(ct.Structure):
        _fields_ = [
            ("file", ct.c_char_p),
            ("base", ct.c_void_p),
            ("symbol", ct.c_char_p),
            ("address", ct.c_void_p),
        ]

    info = DlInfo()
    dladdr = ct.CDLL(None).dladdr
    dladdr.argtypes = [ct.c_void_p, ct.POINTER(DlInfo)]
    if not pointer or not dladdr(pointer, ct.byref(info)):
        raise RuntimeError(f"Cannot resolve {name}")
    return info.file.decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["eager", "direct", "aclnn_graph"], required=True
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--byte-scale",
        action="store_true",
        help="Exercise the existing uint8 + E8M0 dtype arguments.",
    )
    options = parser.parse_args()
    if os.environ.get("QBMM_TEST_OP_PLUGIN"):
        torch.ops.load_library(os.environ["QBMM_TEST_OP_PLUGIN"])
    torch.npu.set_device(0)
    libraries = {
        name: resolved_api_library(name)
        for name in ("aclnnQuantMatmulV5", "aclnnQuantMatmulWeightNz")
    }
    if os.environ.get("QBMM_CUSTOM_OPAPI"):
        expected_library = Path(os.environ["QBMM_CUSTOM_OPAPI"]).resolve()
        assert all(
            Path(name).resolve() == expected_library for name in libraries.values()
        ), libraries
    print(
        json.dumps({"api_libraries": libraries, "torch_npu_path": torch_npu.__file__}),
        flush=True,
    )
    records = []
    direct = DirectAclnn() if options.mode in ("direct", "aclnn_graph") else None
    specs = [
        (fp4, ta, tb, True, batch)
        for fp4, ta, tb, batch in itertools.product(
            (False, True), (False, True), (False, True), ("none", "a", "b", "ab")
        )
    ]
    if options.smoke:
        specs = [(False, False, True, True, "none"), (True, False, True, True, "none")]
    for fp4, ta, tb, weight_nz, batch in specs:
        weight_nz_limit = (
            fp4 and ta and weight_nz
        )  # Existing MXFP4 WeightNZ API constraint.
        if weight_nz_limit:
            entry = dict(
                fp4=fp4,
                ta=ta,
                tb=tb,
                weight_nz=weight_nz,
                batch=batch,
                mode=options.mode,
                byte_scale=options.byte_scale,
                status="skipped_existing_limit",
                reason="Existing MXFP4 WeightNZ requires transposeX1=False.",
            )
            records.append(entry)
            print(json.dumps(entry), flush=True)
            continue
        # Both N/K tails and a shape crossing multiple N tiles/K blocks. Batch
        # data has distinct scale exponents, independently of either transpose.
        cases = [
            (False, False, None),
            (True, False, None),
            (False, True, None),
            (False, False, (66, 2050, 590)),
        ]
        for equal_axes, bad_scale, mkn in cases:
            name = dict(
                fp4=fp4,
                ta=ta,
                tb=tb,
                weight_nz=weight_nz,
                batch=batch,
                equal_axes=equal_axes,
                bad_scale=bad_scale,
                mode=options.mode,
                byte_scale=options.byte_scale,
                mkn=mkn,
            )
            stage = "prepare"
            try:
                args, expected = make_case(
                    fp4,
                    ta,
                    tb,
                    weight_nz,
                    batch,
                    equal_axes,
                    bad_scale,
                    byte_scale=options.byte_scale,
                    mkn=mkn,
                )
                stage = "operator"
                if direct:
                    result = direct.call(
                        args,
                        expected,
                        fp4,
                        ta,
                        tb,
                        weight_nz,
                        capture=options.mode == "aclnn_graph",
                        check_descriptors=True,
                    )
                else:
                    result = public_call(fp4)(*args)
                torch.npu.synchronize()
                error = float((result.cpu() - expected).abs().max())
                name.update(
                    status="unexpected_accept"
                    if bad_scale
                    else ("pass" if error < 1e-3 else "wrong_result"),
                    max_error=error,
                )
            except Exception as exc:
                status = "error"
                if stage == "operator" and bad_scale and is_input_rejection(exc):
                    status = "expected_reject"
                name.update(status=status, stage=stage, error=str(exc)[:2500])
            records.append(name)
            print(json.dumps(name), flush=True)
    for fp4, equal_axes in itertools.product((False, True), repeat=2):
        entry = dict(
            mode=options.mode,
            fp4=fp4,
            equal_axes=equal_axes,
            regression="ND B + ScaleNZ must be rejected",
            byte_scale=options.byte_scale,
        )
        try:
            args, expected = make_case(
                fp4,
                False,
                False,
                False,
                equal_axes=equal_axes,
                byte_scale=options.byte_scale,
            )
            try:
                if direct:
                    direct.call(
                        args,
                        expected,
                        fp4,
                        False,
                        False,
                        False,
                        capture=options.mode == "aclnn_graph",
                        check_descriptors=True,
                    )
                else:
                    public_call(fp4)(*args)
                entry["status"] = "unexpected_accept"
            except RuntimeError as exc:
                if not is_input_rejection(exc, allow_weight_guard=True):
                    raise
                entry.update(status="expected_reject", error=str(exc)[:1000])
        except Exception as exc:
            entry.update(status="error", error=str(exc)[:2500])
        records.append(entry)
        print(json.dumps(entry), flush=True)
    if direct:
        for fault in (
            "stride",
            "offset",
            "storage",
            "dtype",
            "group",
            "group_ncl",
            "a_e5m2",
            "b_e5m2",
            "ab_e5m2",
        ):
            entry = dict(
                mode=options.mode,
                invalid_descriptor=fault,
                byte_scale=options.byte_scale,
            )
            args, expected = make_case(
                False, False, False, True, byte_scale=options.byte_scale
            )
            try:
                direct.call(
                    args,
                    expected,
                    False,
                    False,
                    False,
                    True,
                    capture=options.mode == "aclnn_graph",
                    fault=fault,
                    check_descriptors=True,
                )
                entry["status"] = "unexpected_accept"
            except RuntimeError as exc:
                entry.update(
                    status="expected_reject" if is_input_rejection(exc) else "error",
                    error=str(exc)[:1000],
                )
            records.append(entry)
            print(json.dumps(entry), flush=True)
        for fp4 in (False, True):
            entry = dict(
                mode=options.mode,
                fp4=fp4,
                regression="ScaleA base format query is immutable",
                byte_scale=options.byte_scale,
            )
            try:
                args, expected = make_case(
                    fp4, False, False, True, byte_scale=options.byte_scale
                )
                result = direct.call(
                    args,
                    expected,
                    fp4,
                    False,
                    False,
                    True,
                    capture=options.mode == "aclnn_graph",
                    fault="scale_a_format",
                    check_descriptors=True,
                )
                torch.npu.synchronize()
                error = float((result.cpu() - expected).abs().max())
                entry.update(
                    status="pass" if error < 1e-3 else "wrong_result", max_error=error
                )
            except Exception as exc:
                entry.update(status="error", error=str(exc)[:2500])
            records.append(entry)
            print(json.dumps(entry), flush=True)
    # Exercise the old WeightNZ + ND-scale path through the same changed host
    # and Tensor implementation. Cover all supported transposes and broadcasts.
    for fp4, ta, tb, batch in itertools.product(
        (False, True), (False, True), (False, True), ("none", "a", "b", "ab")
    ):
        if fp4 and ta:
            continue  # Existing FP4 WeightNZ API restriction.
        entry = dict(
            mode=options.mode,
            fp4=fp4,
            ta=ta,
            tb=tb,
            batch=batch,
            byte_scale=options.byte_scale,
            regression="WeightNZ + ND scale",
        )
        try:
            args, expected = make_case(
                fp4, ta, tb, True, batch, scale_nz=False, byte_scale=options.byte_scale
            )
            if direct:
                result = direct.call(
                    args,
                    expected,
                    fp4,
                    ta,
                    tb,
                    True,
                    capture=options.mode == "aclnn_graph",
                    scale_nz=False,
                    check_descriptors=True,
                )
            else:
                result = public_call(fp4)(*args)
            torch.npu.synchronize()
            error = float((result.cpu() - expected).abs().max())
            entry.update(
                status="pass" if error < 1e-3 else "wrong_result", max_error=error
            )
            if options.mode == "eager" and tb:
                mismatch, _ = make_case(
                    fp4,
                    False,
                    False,
                    True,
                    batch,
                    scale_nz=False,
                    byte_scale=options.byte_scale,
                )
                x, w, _, sa = args
                try:
                    public_call(fp4)(x, w, mismatch[2], sa)
                    torch.npu.synchronize()
                    raise AssertionError(
                        "WeightNZ + ND scale transpose mismatch was accepted"
                    )
                except RuntimeError as exc:
                    if (
                        "Input x2 tensor and scale tensor's transpose are not same"
                        not in str(exc)
                    ):
                        raise
                entry["nd_scale_transpose_check"] = "pass"
        except Exception as exc:
            entry.update(status="error", error=str(exc)[:2500])
        records.append(entry)
        print(json.dumps(entry), flush=True)
    if options.mode == "eager":
        for fp4 in (False, True):
            entry = dict(
                mode="eager", fp4=fp4, regression="ND scale keeps its transpose check"
            )
            try:
                args, expected = make_case(
                    fp4,
                    False,
                    True,
                    False,
                    scale_nz=False,
                    byte_scale=options.byte_scale,
                )
                call = public_call(fp4)
                y = call(*args)
                torch.npu.synchronize()
                assert float((y.cpu() - expected).abs().max()) < 1e-3
                x, w, sb, sa = args
                mismatch, _ = make_case(
                    fp4,
                    False,
                    False,
                    False,
                    scale_nz=False,
                    byte_scale=options.byte_scale,
                )
                try:
                    call(x, w, mismatch[2], sa)
                    raise AssertionError("ND B/scale transpose mismatch was accepted")
                except RuntimeError as exc:
                    assert "transpose are not same" in str(exc), str(exc)
                entry["status"] = "pass"
            except Exception as exc:
                entry.update(status="error", error=str(exc)[:2500])
            records.append(entry)
            print(json.dumps(entry), flush=True)
    summary = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        torch_npu_path=torch_npu.__file__,
        api_libraries=libraries,
        records=records,
    )
    if options.output:
        options.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if any(
        r["status"] not in ("pass", "expected_reject", "skipped_existing_limit")
        for r in records
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
