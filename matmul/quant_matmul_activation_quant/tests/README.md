# QuantMatmulActivationQuant 测试

覆盖 GELU（tanh/erf）的 MXFP8、MXFP4 通路，以及 SwiGLU、SiLU 的 MXFP8 通路。

| 目录 | 内容 |
| --- | --- |
| `ut/op_api` | ND、WeightNZ 接口测试 |
| `ut/op_host` | InferShape、Host tiling 测试 |
| `ut/op_kernel` | CSV 驱动的 Kernel UT |
| `st/arch35` | TTK Kernel、ACLNN、E2E 用例 |
| `assets` | Golden 与比较器 |

## 环境

- Linux、Ascend950、配套 CANN 和 ops-tensor/Blaze；Kernel UT 需要 tikicpulib。
- 安装当前版本的 Host/API 和 Kernel 算子包，参见[编译安装说明](../../../docs/zh/install/build.md)。
- TTK 需要 `ml-dtypes`、`en-dtypes>=0.0.4`；ACLNN、E2E 和 NPU Golden 需要配套的 `torch`、`torch_npu`，E2E 还需要 TorchAir 和 `cann_ops_nn`。
- TTK 需支持 WeightNZ 完整存储传递，以及非原生 dtype（FP4）的自定义 NPU Golden。

## Kernel UT

在 ops-nn 根目录执行：

```bash
bash build.sh -u --opkernel --ops=quant_matmul_activation_quant --soc=ascend950
```

用例由 [CSV](ut/op_kernel/test_quant_matmul_activation_quant.csv) 驱动。修改输入或计算参数时，同步更新 `expectedY`、`expectedYScale`。

## TTK ST

| 通路 | 子命令 | 用例文件 | 融合算子设备耗时列 |
| --- | --- | --- | --- |
| Kernel | `kernel` | [Kernel CSV](st/arch35/ttk_kernel_quant_matmul_activation_quant_st.csv) | `dyn_perf_us`（默认动态编译） |
| ACLNN ND | `aclnn` | [ND CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv) | `op_perf_us` |
| ACLNN WeightNZ | `aclnn` | [WeightNZ CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_weight_nz_st.csv) | `op_perf_us` |
| E2E | `e2e` | [E2E CSV](st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv) | `eager_device_perf_us` |

在 ops-test-kit 目录运行，例如 ACLNN ND：

```bash
cd /path/to/ops-test-kit
OP_TESTS=/path/to/ops-nn/matmul/quant_matmul_activation_quant/tests
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" --seed 42
```

其余通路按表替换子命令和 CSV。默认使用 CPU Golden，检查结果 CSV 的 `precision_status` 是否为 `PASS`。

用例配置约定：

- Kernel NZ：`input_shapes` 填物理形状，`input_ori_shapes` 填逻辑形状，`input_ori_formats` 填 `ND`。
- ACLNN NZ：`tensor_view_shapes` 填逻辑形状，`tensor_storage_shapes` 显式填写完整物理形状。FP8 权重的逻辑形状为 `(*batch, R, C)` 时，物理形状为 `(*batch, ceil(C/32), ceil(R/16), 16, 32)`。
- E2E：API 为 `cann_ops_nn.ops.quant_matmul_activation_quant`，张量顺序为 `x1,x2,x2_scale,x1_scale,bias`；矩阵视图保持 `(...,M,K)`、`(...,K,N)`。
- 输出 dtype 属性：Kernel 使用 `y_dtype`，ACLNN 使用 `yDtype`，E2E 使用 `output_dtype`。E2E FP4 使用 `uint8` 打包，并设置 `output_dtype/x1_dtype/x2_dtype=296`。具体枚举及布局参考各通路 CSV。

## 拼接性能对比

Kernel、ACLNN（含 WeightNZ）和 E2E 均可通过 `--golden-mode NPU` 测量以下拼接标杆：

| 激活 | 量化类型 | 拼接标杆 |
| --- | --- | --- |
| SiLU | MXFP8 | QBMM(FP32) → SiLU(FP32) → Cast(BF16) → DynamicMxQuant |
| SwiGLU | MXFP8 | QBMM(BF16) → SwigluMxQuant（尾轴切分、左半激活、mode=0） |
| GELU（tanh/erf） | MXFP8、MXFP4 | QBMM(FP32) → GELU(FP32) → Cast(BF16) → DynamicMxQuant |

Cast 计入标杆耗时。SwiGLU 需环境提供 `torch_npu.npu_swiglu_mx_quant`。MXFP8 支持 `scale_alg=0/1`、`round_mode="rint"`；MXFP4 支持 `scale_alg=0/2`、`round_mode="rint"/"floor"/"round"`，并传递 `dst_type_max`。

在上述环境中运行，例如：

```bash
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" -t silu_nd_e4m3_n65 \
  --golden-mode NPU --devices 0 --process-count 1 \
  --warmup true --run 1 --seed 42 -o silu_perf.csv
```

测试 SwiGLU 或 GELU 时，可将 `-t` 改为 `swiglu_s0_y36`、`gelu_tanh_s0_y36` 或 `gelu_tanh_fp4_nd_rint`。其他通路按上表替换子命令、CSV，并选择对应激活的用例。
保持预热开启，使 NZ 权重格式准备在预热阶段完成。

拼接耗时为 `golden_device_perf_us`，加速比为 **拼接耗时 / 融合算子设备耗时**。
仅使用 `golden_source=npu` 且耗时为正的结果；`golden_kernel_details` 提供标杆各算子的耗时及调用次数。
这些列统计设备执行时间。当前 NPU Golden 每例采样一次，`--run` 不增加其采样次数；可多次串行运行，分别保存结果后取中位数。

## Golden 与精度

精度验收使用 `--golden-mode Enable`（默认）的 CPU Golden，参考实现和比较器见 [assets/spec.py](assets/spec.py)。`NPU` 模式的精度结果基于拼接输出；SwiGLU 标杆在激活前舍入为 BF16，与融合实现的舍入位置不同。

ST 对 `y` 和 `y_scale` 分别要求至少 99.9% 的元素误差不超过 1 ULP；形状和 dtype 必须一致，NaN 位置及 Inf 符号必须匹配，正负零视为相等。FP4 按解包后的元素比较。Kernel UT 对固定输入严格比较。
