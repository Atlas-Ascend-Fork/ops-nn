/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Unit tests for lars_v2_update public Host tiling logic (DESIGN §5.3).
//
// TDD red-baseline (Task 19): the public tiling functions are stubs, so every
// case is expected to FAIL (0 passed). Task 20 implements the real formulas and
// these cases must then all pass.
//
// Independence contract: the Ora* functions below hand-implement the DESIGN §5.3
// formulas. They MUST NOT call any optiling:: function under test. The optiling::
// functions are only invoked inside the TEST_P bodies as the "actual" result.

#include <algorithm>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "lars_v2_update_tiling_data.h" // SplitResult, MultiCoreResult, PHYS_NODES
#include "lars_v2_update_tiling.h"      // optiling:: public tiling functions

namespace {
// DESIGN §5.3 constants (hand-copied; oracle must NOT call optiling:: functions).
constexpr int64_t PHYS_NODES_ORACLE = 3; // DESIGN §5.3 P conclusion (PHYS_NODES)
constexpr int64_t UB_192K = 196608;      // Ascend950DT typical UB = 192 KiB
constexpr int64_t UB_256K = 262144;      // 256 KiB variant

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ===== Independent oracle: PadAndSqueeze (DESIGN §5.3) =====
struct OraPadSqueeze {
    std::vector<int64_t> maximum_bro_shape;
    std::vector<std::vector<int64_t>> normal_input_shapes;
    std::vector<std::vector<int64_t>> normal_output_shapes;
};
OraPadSqueeze OraPadAndSqueeze(const std::vector<std::vector<int64_t>>& in,
                               const std::vector<std::vector<int64_t>>& out)
{
    OraPadSqueeze r;
    int64_t ni = (int64_t)in.size(), no = (int64_t)out.size(), mr = 0;
    for (auto& s : in)
        mr = std::max(mr, (int64_t)s.size());
    for (auto& s : out)
        mr = std::max(mr, (int64_t)s.size());
    auto pad = [&](const std::vector<int64_t>& s) {
        std::vector<int64_t> p;
        p.assign(mr - (int64_t)s.size(), 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    std::vector<std::vector<int64_t>> pin(ni), pout(no);
    for (int64_t i = 0; i < ni; i++)
        pin[i] = pad(in[i]);
    for (int64_t i = 0; i < no; i++)
        pout[i] = pad(out[i]);
    r.normal_input_shapes.assign(ni, {});
    r.normal_output_shapes.assign(no, {});
    for (int64_t d = 0; d < mr; d++) {
        // 独立按 numpy 广播语义推导该维结果，刻意不采用实现里的
        // "running max + has_zero 标志" 结构，避免 oracle 与实现同构而失去差分能力：
        //   1) 收集该维所有参与方（6 input + 1 output，已补 1）的取值
        //   2) 全为 1 -> 该维被 squeeze
        //   3) 否则取所有 != 1 的取值集合 S：
        //        0 ∈ S      -> 结果为 0（0 是广播湮灭元：0 与 1 广播得 0）
        //        S 只有一个值 v -> 结果为 v（1 可广播到 v）
        //        S 有多个值   -> 非法广播，交由 CheckBroadcastShape 拒绝，此处取 -1 作哨兵
        std::vector<int64_t> vals;
        for (int64_t i = 0; i < ni; i++)
            vals.push_back(pin[i][d]);
        for (int64_t i = 0; i < no; i++)
            vals.push_back(pout[i][d]);
        std::vector<int64_t> nonOne;
        for (int64_t v : vals) {
            if (v == 1)
                continue;
            if (std::find(nonOne.begin(), nonOne.end(), v) == nonOne.end())
                nonOne.push_back(v);
        }
        if (!nonOne.empty()) {
            int64_t resolved;
            if (std::find(nonOne.begin(), nonOne.end(), 0) != nonOne.end())
                resolved = 0;
            else if (nonOne.size() == 1)
                resolved = nonOne[0];
            else
                resolved = -1; // 非法广播哨兵
            r.maximum_bro_shape.push_back(resolved);
            for (int64_t i = 0; i < ni; i++)
                r.normal_input_shapes[i].push_back(pin[i][d]);
            for (int64_t i = 0; i < no; i++)
                r.normal_output_shapes[i].push_back(pout[i][d]);
        }
    }
    if (r.maximum_bro_shape.empty()) {
        r.maximum_bro_shape.push_back(1);
        for (int64_t i = 0; i < ni; i++)
            r.normal_input_shapes[i].push_back(1);
        for (int64_t i = 0; i < no; i++)
            r.normal_output_shapes[i].push_back(1);
    }
    return r;
}

// ===== Independent oracle: CheckBroadcastShape (DESIGN §5.3) =====
struct OraBcast {
    bool ok;
    int64_t bad_dim;
};
OraBcast OraCheckBroadcast(const std::vector<std::vector<int64_t>>& pin, const std::vector<std::vector<int64_t>>& pout,
                           int64_t mr)
{
    OraBcast r{true, -1};
    for (int64_t d = 0; d < mr; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < pin.size(); i++) {
            if (pin[i][d] != 1) {
                if (ref == -1)
                    ref = pin[i][d];
                else if (pin[i][d] != ref) {
                    r.ok = false;
                    r.bad_dim = d;
                    return r;
                }
            }
        }
        for (size_t i = 0; i < pout.size(); i++) {
            if (pout[i][d] != 1) {
                if (ref == -1)
                    ref = pout[i][d];
                else if (pout[i][d] != ref) {
                    r.ok = false;
                    r.bad_dim = d;
                    return r;
                }
            }
        }
    }
    return r;
}

// ===== Independent oracle: per_buf (DESIGN §5.3, P=3) =====
int64_t OraPerBufBytes(int64_t ub, int64_t P) { return ((ub - SCALAR_BUF_BYTES) / P) & ~31LL; }
int64_t OraPerBufElems(int64_t bytes) { return bytes / 4; }

// ===== Independent oracle: FindSplitAxis (DESIGN §5.3) =====
struct OraSplit {
    int64_t axis, a_i, a_o, a_i_tail;
};
OraSplit OraFindSplit(const std::vector<int64_t>& shape, int64_t ub, int64_t P)
{
    OraSplit o{0, 0, 1, 0};
    int64_t per_buf_bytes = OraPerBufBytes(ub, P);
    int64_t per_buf_elems = OraPerBufElems(per_buf_bytes);
    int64_t rank = (int64_t)shape.size();
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (shape[k] * inner > per_buf_elems) {
            o.axis = k;
            o.a_i = per_buf_elems / inner;
            o.a_o = CeilDiv(shape[k], o.a_i);
            int64_t rem = shape[k] % o.a_i;
            o.a_i_tail = (rem == 0) ? o.a_i : rem;
            return o;
        }
        if (k == 0) {
            o.axis = 0;
            o.a_i = shape[0];
            o.a_o = 1;
            o.a_i_tail = shape[0];
            return o;
        }
        inner *= shape[k];
    }
    return o;
}

// ===== Independent oracle: MultiCoreSplit (DESIGN §5.3) =====
struct OraMc {
    int64_t num_cores, total_tiles, tiles_main, cores_tail;
};
OraMc OraMultiCore(const std::vector<int64_t>& shape, const OraSplit& sp, int64_t max_cores)
{
    OraMc o;
    int64_t k = sp.axis, outer_prod = 1;
    for (int64_t j = 0; j < k; j++)
        outer_prod *= shape[j];
    o.total_tiles = outer_prod * sp.a_o;
    o.num_cores = (o.total_tiles < max_cores) ? o.total_tiles : max_cores;
    o.tiles_main = o.total_tiles / o.num_cores;
    o.cores_tail = o.total_tiles % o.num_cores;
    return o;
}

// ===== Independent oracle: ChooseTilingKey (DESIGN §5.1/§5.3) =====
int64_t OraChooseTilingKey(int64_t r)
{
    if (r <= 4)
        return 4;
    if (r <= 8)
        return 8;
    return -1;
}

// ===== Independent oracle: ResolveAttrs (DESIGN §5.3) =====
struct OraAttrs {
    float hyperpara;
    float epsilon;
    int64_t use_clip;
};
OraAttrs OraResolveAttrs(const float* hp, const float* eps, const bool* uc)
{
    OraAttrs o;
    o.hyperpara = hp ? *hp : 0.001f;
    o.epsilon = eps ? *eps : 0.00001f;
    o.use_clip = (uc && *uc) ? 1LL : 0LL;
    return o;
}
} // namespace

// =====================================================================
// PadAndSqueeze (6 in / 1 out)
// =====================================================================
struct PadCase {
    const char* name;
    std::vector<int64_t> w;      // w and g share this shape
    std::vector<int64_t> scalar; // shape of the 4 scalar inputs ([] or [1])
};
class PadAndSqueezeTest : public testing::TestWithParam<PadCase> {};

TEST_P(PadAndSqueezeTest, Formula)
{
    const auto& p = GetParam();
    std::vector<std::vector<int64_t>> in(6);
    in[0] = p.w;
    in[1] = p.w; // w, g (same shape)
    in[2] = p.scalar;
    in[3] = p.scalar; // w_square_sum, g_square_sum
    in[4] = p.scalar;
    in[5] = p.scalar;                              // weight_decay, learning_rate
    std::vector<std::vector<int64_t>> out = {p.w}; // g_new.shape == w.shape

    OraPadSqueeze exp = OraPadAndSqueeze(in, out);

    std::vector<int64_t> mbs;
    std::vector<std::vector<int64_t>> nin, nout;
    bool ret = optiling::PadAndSqueeze(in, out, mbs, nin, nout);
    EXPECT_TRUE(ret);
    EXPECT_EQ(mbs, exp.maximum_bro_shape);
    EXPECT_EQ(nin, exp.normal_input_shapes);
    EXPECT_EQ(nout, exp.normal_output_shapes);
}

INSTANTIATE_TEST_SUITE_P(LarsV2Update, PadAndSqueezeTest,
                         testing::Values(PadCase{"scalar_w", {}, {}}, PadCase{"1d_256", {256}, {}},
                                         PadCase{"1d_256_s1", {256}, {1}}, PadCase{"1d_4096", {4096}, {}},
                                         PadCase{"2d_1024", {1024, 1024}, {}}, PadCase{"2d_1024_s1", {1024, 1024}, {1}},
                                         PadCase{"4d_conv", {64, 3, 7, 7}, {}},
                                         PadCase{"4d_conv2", {128, 128, 3, 3}, {}}, PadCase{"5d", {8, 8, 8, 8, 8}, {}},
                                         PadCase{"8d_max", {2, 2, 2, 2, 2, 2, 2, 2}, {}},
                                         PadCase{"empty_0_3", {0, 3}, {}}, PadCase{"nonalign_4_7", {4, 7}, {}},
                                         // 空 tensor：0 维必须传播，不能被标量补位的 1 吞掉
                                         PadCase{"empty_1d", {0}, {}}, PadCase{"empty_1d_s1", {0}, {1}},
                                         PadCase{"empty_0_3_s1", {0, 3}, {1}}, PadCase{"empty_3_0", {3, 0}, {}},
                                         PadCase{"empty_0_0", {0, 0}, {}}, PadCase{"empty_mid_dim", {2, 0, 3}, {}},
                                         PadCase{"empty_all_ones_squeezed", {1, 0, 1, 0, 1}, {}},
                                         PadCase{"empty_8d", {0, 2, 2, 2, 2, 2, 2, 2}, {}},
                                         PadCase{"empty_zero_off_axis", {10, 0, 10, 10, 10, 10, 10, 10}, {}},
                                         PadCase{"empty_large_inner", {0, 1024}, {}},
                                         PadCase{"empty_multi_zero", {5, 0, 7, 0, 3}, {}}));

// 空 tensor 端到端契约：PadAndSqueeze → FindSplitAxis → MultiCoreSplit 之后，
// kernel 侧必须一个元素都不写（否则向 0 元素输出越界写）。
// 复刻 kernel 的两道守卫：Process() 的 totalTiles==0 短路与 ProcessTile() 的 count<=0 跳过。
struct EmptyCase {
    const char* name;
    std::vector<int64_t> w;
};
class EmptyTensorNoWriteTest : public testing::TestWithParam<EmptyCase> {};

TEST_P(EmptyTensorNoWriteTest, KernelWritesNothing)
{
    const auto& p = GetParam();
    std::vector<std::vector<int64_t>> in(6);
    in[0] = p.w;
    in[1] = p.w;
    for (int64_t i = 2; i < 6; i++)
        in[i] = {1};
    std::vector<std::vector<int64_t>> out = {p.w};

    std::vector<int64_t> mbs;
    std::vector<std::vector<int64_t>> nin, nout;
    ASSERT_TRUE(optiling::PadAndSqueeze(in, out, mbs, nin, nout));
    // 前置断言：0 维必须保留在广播坐标系里，否则后续切分按非空规模进行
    ASSERT_TRUE(std::find(mbs.begin(), mbs.end(), 0) != mbs.end()) << "empty dim was swallowed";

    SplitResult split{};
    ASSERT_TRUE(optiling::FindSplitAxis(mbs, 4, UB_256K, PHYS_NODES_ORACLE, split));
    MultiCoreResult mc{};
    ASSERT_TRUE(optiling::MultiCoreSplit(mbs, split, 72, mc));
    ASSERT_GE(mc.num_cores, 1);
    ASSERT_GE(split.a_o, 1);

    // 复刻 kernel 写出量：totalTiles==0 整体短路，否则逐 tile 判 count<=0
    int64_t written = 0;
    if (mc.total_tiles != 0) {
        int64_t effRank = (int64_t)mbs.size();
        for (int64_t t = 0; t < mc.total_tiles; t++) {
            int64_t innerCount = 1;
            for (int64_t j = split.axis + 1; j < effRank; j++)
                innerCount *= mbs[j];
            int64_t aoOff = t % split.a_o;
            int64_t aiSeg = (aoOff == split.a_o - 1) ? split.a_i_tail : split.a_i;
            int64_t count = aiSeg * innerCount;
            if (count <= 0)
                continue;
            written += count;
        }
    }
    EXPECT_EQ(written, 0) << "kernel would write into a 0-element output buffer";
}

INSTANTIATE_TEST_SUITE_P(LarsV2Update, EmptyTensorNoWriteTest,
                         testing::Values(EmptyCase{"e1d", {0}}, EmptyCase{"e_0_3", {0, 3}}, EmptyCase{"e_3_0", {3, 0}},
                                         EmptyCase{"e_0_0", {0, 0}}, EmptyCase{"e_mid", {2, 0, 3}},
                                         EmptyCase{"e_squeeze", {1, 0, 1, 0, 1}},
                                         EmptyCase{"e_8d", {0, 2, 2, 2, 2, 2, 2, 2}},
                                         EmptyCase{"e_zero_off_axis", {10, 0, 10, 10, 10, 10, 10, 10}},
                                         EmptyCase{"e_large_inner", {0, 1024}},
                                         EmptyCase{"e_multi_zero", {5, 0, 7, 0, 3}}));

// =====================================================================
// CheckBroadcastShape (w.shape==g.shape validation)
// =====================================================================
struct BcastCase {
    const char* name;
    std::vector<std::vector<int64_t>> pin;  // 6 inputs, padded
    std::vector<std::vector<int64_t>> pout; // 1 output, padded
    int64_t mr;
    bool ok;
    int64_t bad;
};
class CheckBroadcastShapeTest : public testing::TestWithParam<BcastCase> {};

TEST_P(CheckBroadcastShapeTest, Formula)
{
    const auto& p = GetParam();
    OraBcast exp = OraCheckBroadcast(p.pin, p.pout, p.mr);
    // Self-check: oracle agrees with the hand-coded expectation in the case.
    EXPECT_EQ(exp.ok, p.ok);
    EXPECT_EQ(exp.bad_dim, p.bad);

    int64_t bad = INT64_MIN;
    bool ret = optiling::CheckBroadcastShape(p.pin, p.pout, p.mr, &bad);
    EXPECT_EQ(ret, exp.ok);
    EXPECT_EQ(bad, exp.bad_dim);
}

INSTANTIATE_TEST_SUITE_P(
    LarsV2Update, CheckBroadcastShapeTest,
    testing::Values(
        BcastCase{
            "valid_2d", {{1024, 1024}, {1024, 1024}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, {{1024, 1024}}, 2, true, -1},
        BcastCase{"valid_1d", {{256}, {256}, {1}, {1}, {1}, {1}}, {{256}}, 1, true, -1},
        BcastCase{"mismatch_dim1", {{2, 3}, {2, 4}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, {{2, 3}}, 2, false, 1},
        BcastCase{"mismatch_dim0", {{2, 3}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, {{2, 3}}, 2, false, 0},
        BcastCase{"mismatch_1d", {{256}, {512}, {1}, {1}, {1}, {1}}, {{256}}, 1, false, 0}));

// =====================================================================
// FindSplitAxis (P=3, per_buf_bytes=(ub/3)&~31, per_buf_elems=bytes/4)
// =====================================================================
struct SplitCase {
    const char* name;
    std::vector<int64_t> shape;
    int64_t ub;
};
class FindSplitAxisTest : public testing::TestWithParam<SplitCase> {};

TEST_P(FindSplitAxisTest, Formula)
{
    const auto& p = GetParam();
    OraSplit exp = OraFindSplit(p.shape, p.ub, PHYS_NODES_ORACLE);
    SplitResult act;
    act.axis = INT64_MIN;
    act.a_i = INT64_MIN;
    act.a_o = INT64_MIN;
    act.a_i_tail = INT64_MIN;
    bool ret = optiling::FindSplitAxis(p.shape, /*dtype_size=*/4, p.ub, PHYS_NODES_ORACLE, act);
    EXPECT_TRUE(ret);
    EXPECT_EQ(act.axis, exp.axis);
    EXPECT_EQ(act.a_i, exp.a_i);
    EXPECT_EQ(act.a_o, exp.a_o);
    EXPECT_EQ(act.a_i_tail, exp.a_i_tail);
}

INSTANTIATE_TEST_SUITE_P(
    LarsV2Update, FindSplitAxisTest,
    testing::Values(SplitCase{"1d_256_ub192", {256}, UB_192K}, SplitCase{"1d_4096_ub192", {4096}, UB_192K},
                    SplitCase{"1d_16385_ub192", {16385}, UB_192K}, SplitCase{"2d_1024_ub192", {1024, 1024}, UB_192K},
                    SplitCase{"4d_conv_ub192", {64, 3, 7, 7}, UB_192K},
                    SplitCase{"4d_conv2_ub192", {128, 128, 3, 3}, UB_192K}, SplitCase{"2d_4_7_ub192", {4, 7}, UB_192K},
                    SplitCase{"8d_max_ub192", {2, 2, 2, 2, 2, 2, 2, 2}, UB_192K},
                    SplitCase{"5d_ub192", {8, 8, 8, 8, 8}, UB_192K},
                    SplitCase{"5d_mix_ub192", {2, 4, 8, 16, 32}, UB_192K},
                    SplitCase{"2d_1024_ub256", {1024, 1024}, UB_256K}, SplitCase{"1d_big_ub256", {100000}, UB_256K}));

// =====================================================================
// MultiCoreSplit
// =====================================================================
struct McCase {
    const char* name;
    std::vector<int64_t> shape;
    int64_t axis;
    int64_t a_o;
    int64_t max_cores;
};
class MultiCoreSplitTest : public testing::TestWithParam<McCase> {};

TEST_P(MultiCoreSplitTest, Formula)
{
    const auto& p = GetParam();
    OraSplit sp;
    sp.axis = p.axis;
    sp.a_o = p.a_o;
    sp.a_i = 1;
    sp.a_i_tail = 1; // a_i unused by MultiCoreSplit
    OraMc exp = OraMultiCore(p.shape, sp, p.max_cores);

    SplitResult ub;
    ub.axis = p.axis;
    ub.a_i = 1;
    ub.a_o = p.a_o;
    ub.a_i_tail = 1;
    MultiCoreResult act;
    act.num_cores = INT64_MIN;
    act.total_tiles = INT64_MIN;
    act.tiles_main = INT64_MIN;
    act.cores_tail = INT64_MIN;
    bool ret = optiling::MultiCoreSplit(p.shape, ub, p.max_cores, act);
    EXPECT_TRUE(ret);
    EXPECT_EQ(act.num_cores, exp.num_cores);
    EXPECT_EQ(act.total_tiles, exp.total_tiles);
    EXPECT_EQ(act.tiles_main, exp.tiles_main);
    EXPECT_EQ(act.cores_tail, exp.cores_tail);
}

INSTANTIATE_TEST_SUITE_P(
    LarsV2Update, MultiCoreSplitTest,
    testing::Values(McCase{"lt_cores", {256}, 0, 1, 32},           // total=1  -> num=1, main=1, tail=0
                    McCase{"gt_exact", {1024, 1024}, 0, 64, 32},   // total=64 -> num=32,main=2, tail=0
                    McCase{"gt_tail", {65}, 0, 65, 32},            // total=65 -> num=32,main=2, tail=1
                    McCase{"outer_prod", {4, 1024}, 1, 9, 32},     // total=36 -> num=32,main=1, tail=4
                    McCase{"outer_exact", {4, 1024}, 1, 8, 32},    // total=32 -> num=32,main=1, tail=0
                    McCase{"small_cores8", {1024, 1024}, 0, 64, 8} // total=64 -> num=8, main=8, tail=0
                    ));

// =====================================================================
// ChooseTilingKey dispatch (effective rank -> 4 / 8 / -1)
// =====================================================================
struct KeyCase {
    const char* name;
    int64_t rank;
};
class ChooseTilingKeyTest : public testing::TestWithParam<KeyCase> {};

TEST_P(ChooseTilingKeyTest, Formula)
{
    const auto& p = GetParam();
    int64_t exp = OraChooseTilingKey(p.rank);
    int64_t act = optiling::ChooseTilingKey(p.rank);
    EXPECT_EQ(act, exp);
}

INSTANTIATE_TEST_SUITE_P(LarsV2Update, ChooseTilingKeyTest,
                         testing::Values(KeyCase{"r0", 0}, // -> 4
                                         KeyCase{"r1", 1}, // -> 4
                                         KeyCase{"r4", 4}, // -> 4
                                         KeyCase{"r5", 5}, // -> 8
                                         KeyCase{"r8", 8}, // -> 8
                                         KeyCase{"r9", 9}  // -> -1
                                         ));

// =====================================================================
// per_buf helpers (P=3)
// =====================================================================
struct PbCase {
    const char* name;
    int64_t ub;
    int64_t P;
};
class PerBufTest : public testing::TestWithParam<PbCase> {};

TEST_P(PerBufTest, Formula)
{
    const auto& p = GetParam();
    int64_t expBytes = OraPerBufBytes(p.ub, p.P);
    int64_t expElems = OraPerBufElems(expBytes);
    int64_t actBytes = optiling::ComputePerBufBytes(p.ub, p.P);
    int64_t actElems = optiling::ComputePerBufElems(actBytes);
    EXPECT_EQ(actBytes, expBytes);
    EXPECT_EQ(actElems, expElems);
    EXPECT_EQ(expElems, expBytes / 4); // cast.md §Tile: always /4
}

INSTANTIATE_TEST_SUITE_P(LarsV2Update, PerBufTest,
                         testing::Values(PbCase{"ub192_p3", 196608, 3},      // bytes=65344, elems=16336
                                         PbCase{"ub256_p3", 262144, 3},      // bytes=87200, elems=21800
                                         PbCase{"ub192off_p3", 196611, 3},   // (65366)&~31=65344 -> mask exercised
                                         PbCase{"ub200k_p3", 200000, 3},     // bytes=66656, elems=16664
                                         PbCase{"ub192_p3_align", 196704, 3} // bytes=65568, elems=16392
                                         ));

// =====================================================================
// ResolveAttrs (attrs acquisition with defaults)
// =====================================================================
struct AttrCase {
    const char* name;
    bool has_hp;
    float hp;
    bool has_eps;
    float eps;
    bool has_uc;
    bool uc;
};
class ResolveAttrsTest : public testing::TestWithParam<AttrCase> {};

TEST_P(ResolveAttrsTest, Formula)
{
    const auto& p = GetParam();
    const float* hp = p.has_hp ? &p.hp : nullptr;
    const float* eps = p.has_eps ? &p.eps : nullptr;
    const bool* uc = p.has_uc ? &p.uc : nullptr;

    OraAttrs exp = OraResolveAttrs(hp, eps, uc);

    optiling::LarsAttrs act;
    act.hyperpara = -3.3e30f;
    act.epsilon = -3.3e30f;
    act.use_clip = INT64_MIN;
    optiling::ResolveAttrs(hp, eps, uc, act);
    EXPECT_FLOAT_EQ(act.hyperpara, exp.hyperpara);
    EXPECT_FLOAT_EQ(act.epsilon, exp.epsilon);
    EXPECT_EQ(act.use_clip, exp.use_clip);
}

INSTANTIATE_TEST_SUITE_P(
    LarsV2Update, ResolveAttrsTest,
    testing::Values(AttrCase{"defaults", false, 0.0f, false, 0.0f, false, false},   // -> 0.001, 1e-5, 0
                    AttrCase{"explicit", true, 1.0f, true, 0.0f, true, true},       // -> 1.0,   0.0,  1
                    AttrCase{"hp_zero", true, 0.0f, true, 1e-5f, false, false},     // -> 0.0,   1e-5, 0
                    AttrCase{"uc_true", true, 0.001f, true, 1e-5f, true, true},     // -> 0.001, 1e-5, 1
                    AttrCase{"uc_nullptr", true, 0.001f, true, 1e-5f, false, false} // -> 0.001, 1e-5, 0
                    ));
