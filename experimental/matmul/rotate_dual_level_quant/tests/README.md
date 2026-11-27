# RotateDualLevelQuant 测试

本目录提供 TTK golden 资产和 A5 UT。输出顺序均为 `y、level0_scale、level1_scale`。

## 文件与入口

| 文件 | 用途与入口 |
|---|---|
| `assets/rotate_dual_level_quant_kernel_golden.py` | Kernel 直调；`__spec__["rotate_dual_level_quant"]` |
| `assets/rotate_dual_level_quant_aclnn_golden.py` | ACLNN 直调；`__spec__["aclnnRotateDualLevelQuant"]` |
| `assets/rotate_dual_level_quant_e2e_golden.py` | Torch 单算子参考；`__spec__["torch.ops.cann_ops_nn.rotate_dual_level_quant"]` |
| `st/arch35/ttk_kernel_rotate_dual_level_quant_st.csv` | Kernel 直调 TTK 用例表（`ttk kernel`） |
| `st/arch35/ttk_aclnn_rotate_dual_level_quant_st.csv` | ACLNN 直调 TTK 用例表（`ttk aclnn`） |
| `ut/op_host/test_aclnn_rotate_dual_level_quant.cpp` | ACLNN 参数、输出类型与 shape、空输入、非连续输出和异常路径 |
| `ut/op_host/test_rotate_dual_level_quant_infershape.cpp` | 三路输出的 shape/dtype 推导，动态维度、分组和非法块大小 |
| `ut/op_host/arch35/test_rotate_dual_level_quant_tiling.cpp` | 实际 Tiling 注册入口、两级块布局、类型分母、splitM、Fixpipe 单/双接收槽、UB 容量边界和 Ascend950 平台约束 |
| `ut/op_kernel/arch35/test_rotate_dual_level_quant.cpp` | BF16 → FP4 CPU 仿真直调，断言 y 数据、两级 scale、padding 和未覆盖行 |

Torch 文件提供参考计算和比较入口。实际 E2E 调用由算子目录下的 `torch_extension` 注册，
构建、安装和接口说明见 [Torch 接口文档](../docs/torchapi_rotate_dual_level_quant.md)。

## Golden 数值规则

1. FP16/BF16 输入先以 FP32 累加执行旋转矩阵乘，再将结果转换为 BF16。
   支持共享 `(K,K)`、按列块 `(N/K,K,K)`、分组 `(E,K,K)` 和分组按列块 `(E,N/K,K,K)` 旋转矩阵。
2. 可选 alpha 在 `(0,1)` 内时，按每 32 元素 clamp，界限转为 BF16，随后执行 level0。
3. 按原始行的 N 轴每 512 元素统计绝对值最大值，尾块仅统计有效元素。
   FP32 一级尺度使用 `amax * FP32(1/CMAX)`，CMAX 对 FP4/E4M3FN/E5M2 分别为 `6/448/57344`。
   原数据在 FP32 域除以尺度后转换为 BF16，作为 level1 输入。
4. level1 按每 32 元素执行 MX 量化，支持 FP4 的 scaleAlg=0/2、FP8 的 scaleAlg=0/1，
   FP4 的 rint/round/floor，以及 FP8 的 rint。尺度指数和 BF16 倒数按后处理的位运算规则生成。
5. `level0_scale` 为 FP32，shape 为 `[...,ceil(N/512)]`；`level1_scale` 保留 E8M0 dtype，
   shape 为 `[...,ceil(N/64),2]`。奇数个有效 MX 尺度后的 padding 为编码 0。
6. level0 排除 Inf/NaN 参与 amax，FP32 subnormal 尺度置零，并在归一化后恢复输入的有符号零。
   这些是参考实现规则，特殊值的完整设备行为仍需 A5 验证。

NumPy 下 FP4 使用每元素一字节的逻辑 carrier，E8M0 使用一字节编码；不把 E8M0 数值转换为 uint8。
Torch 比较入口支持 FP4 的打包 uint8 输出（低 nibble 为前一个元素），解包后按逻辑元素比较。

## TTK 参数与比对约定

Kernel 属性使用 snake_case，包含 `y_dtype、axis、round_mode、scale_alg、dst_type_max、trans、group_list_type、level0_block_size、level1_block_size`。
`y_dtype` 的 GE 编号为 FP4=40、E4M3FN=36、E5M2=35。两级块大小当前固定为 512、32。

ACLNN 适配器的参数顺序与公开接口一致：

```text
x, rotation, alpha, groupList, groupListType, axis, roundMode, scaleAlg,
dstTypeMax, trans, level0BlockSize, level1BlockSize,
yOut, level0ScaleOut, level1ScaleOut
```

Torch 适配器沿用 RotateQuant 的属性命名习惯，使用 `dst_dtype、transpose_y`，并增加两级块大小。
dst_dtype 省略或为 None 时采用 FP4；支持 FP4 的编号 296，以及 FP8 的 dtype 名或已有测试框架编号。

`compare` 的输出参数顺序为：

```text
actual_y, actual_level0_scale, actual_level1_scale,
golden_y, golden_level0_scale, golden_level1_scale
```

比较策略如下，作为当前测试资产的初始规则，待设备验证后确认适用性：

- y 允许一个相邻可表示值的差距；默认不允许超出该差距的元素。可从 TTK `precision_tolerances[0][1]` 指定允许失败比例。
- `level0_scale` 使用 `rtol=1e-3`、`atol=1e-8` 逐点比较。每条用例按该输出的有效元素数单独统计，超差比例 ≤ 0.1% 时通过。
- level1_scale 编码逐字节相等，包含 padding。
- 比较前检查每路输出的 shape/dtype。

分组允许空组和仅覆盖 M 的连续前缀。ACLNN golden 要求传入三路初始输出，计算时复制它们，保留未覆盖行。
Kernel golden 默认以数值 1 初始化三路参考输出；运行包含未覆盖行的 Kernel 用例时，设备输出也应初始化为数值 1，
或通过 `initial_y、initial_level0_scale、initial_level1_scale` 传入与设备一致的初始输出。
比较时 TTK `compare_context.input_tensors` 须提供四个 Kernel/Torch 输入槽，或七个 ACLNN Tensor 槽；
`attributes` 须提供对应的 group_list_type/groupListType。Kernel/ACLNN 对未覆盖行按原始字节精确比较。
Torch 输出由接口分配，未覆盖行的初值未定义，因此只比较已覆盖行。

## 运行

TTK golden 脚本依赖 NumPy、ml_dtypes；与 TTK 环境保持一致时可安装 en_dtypes。
未安装 en_dtypes 时使用 ml_dtypes 的 `float4_e2m1fn、float8_e8m0fnu` 等价编码，需使用提供这些类型的版本。
实际 Torch Tensor 路径还需安装匹配版本的 PyTorch。

在配置了 CANN SDK、配套的 Tensor 双级 kernel 及统一 epilogue 接口、Ascend950 CPU 仿真库的环境中，使用仓库 UT 入口：

```bash
bash build.sh -u --ophost --experimental --soc=ascend950 --ops=rotate_dual_level_quant
bash build.sh -u --opapi --experimental --soc=ascend950 --ops=rotate_dual_level_quant
bash build.sh -u --opkernel --experimental --soc=ascend950 --ops=rotate_dual_level_quant
```

Host/API UT 由算子根目录 `add_all_modules_sources` 自动收集；Kernel UT 由 `tests/ut/op_kernel/CMakeLists.txt`
以 `ascend950pr_9599` CPU 仿真目标接入，通过实际 C++ 模板接口检查兼容性。当前 Kernel UT 编译 BF16 → FP4；
FP16/BF16 × 三种输出 dtype 的参考计算和 ACLNN 参数检查均有覆盖，完整设备 dtype 矩阵需在 A5 执行验证。

## TTK ST 运行

```bash
P=/mnt/workspace/dual_level_quant/ops-nn_dual_level_rotate_quant/experimental/matmul/rotate_dual_level_quant
cd /mnt/workspace/lanqu_ttk/ops-test-kit && source /mnt/workspace/Ascend/cann/set_env.sh
python3 -m ttk kernel -i $P/tests/st/arch35/ttk_kernel_rotate_dual_level_quant_st.csv --plugin $P/tests/assets --backend npusim --sim-cores 0
python3 -m ttk aclnn -i $P/tests/st/arch35/ttk_aclnn_rotate_dual_level_quant_st.csv --plugin $P/tests/assets --backend npusim --sim-cores 0
```

当前 ST 覆盖矩阵（Ascend950，NPUSim）：

| 用例 | 通路 | x / y | K | 尺寸 (M,N) | 覆盖点 |
|---|---|---|---|---|---|
| ..._0001 | kernel | fp16 / fp4_e2m1 | 64 | 64,128 | 基础 |
| ..._0002 | kernel | bf16 / fp4_e2m1 | 128 | 32,256 | dtype 组合 |
| ..._0003 | kernel | fp16 / e4m3fn | 32 | 16,64 | dtype 组合 |
| ..._0004 | kernel | bf16 / e5m2 | 256 | 16,256 | dtype 组合 |
| ..._0005 | kernel | fp16 / fp4_e2m1 | 64 | 32,128 | alpha clamp |
| ..._0006 | kernel | bf16 / fp4_e2m1 | 64 | 32,128 | scaleAlg=2, dstTypeMax=8.0 |
| ..._0007 | kernel | fp16 / e4m3fn | 32 | 8,544 | N 尾块 + 奇数 scale padding |
| ..._0008 | kernel | fp16 / fp4_e2m1 | 64 | 8,128 | groupList count 分组 |
| ..._0001 | aclnn | bf16 / e4m3fn | 64 | 64,128 | 基础 |
| ..._0002 | aclnn | fp16 / fp4_e2m1 | 64 | 64,128 | dtype 组合 |
| ..._0003 | aclnn | bf16 / fp4_e2m1 | 64 | 32,128 | scaleAlg=2, dstTypeMax=8.0 |
| ..._0004 | aclnn | fp16 / fp4_e2m1 | 64 | 32,128 | roundMode=floor |
| ..._0005 | aclnn | bf16 / e5m2 | 64 | 32,128 | dtype 组合 |
| ..._0006 | aclnn | fp16 / e4m3fn | 64 | 32,128 | scaleAlg=1 |
| ..._0007 | aclnn | fp16 / fp4_e2m1 | 64 | 8,128 | groupList cumsum 分组 |
