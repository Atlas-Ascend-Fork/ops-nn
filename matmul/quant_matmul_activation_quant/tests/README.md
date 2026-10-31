# QuantMatmulActivationQuant 测试

覆盖GELU（tanh/erf）的MXFP8、MXFP4计算，以及SwiGLU、SiLU的MXFP8计算。算子规格见[README](../README.md)。

| 目录 | 测试内容 |
| --- | --- |
| [ut/op_api](ut/op_api) | ND、NZ接口的参数和布局校验 |
| [ut/op_host](ut/op_host) | 形状推导和Tiling |
| [ut/op_kernel](ut/op_kernel) | CSV驱动的Kernel UT |
| [st/arch35](st/arch35) | TTK Kernel、ACLNN、E2E测试 |
| [assets/spec.py](assets/spec.py) | 参考计算（Golden）和精度比较器 |

## 环境准备

- 使用Linux和Ascend 950，配置配套的CANN、ops-tensor/Blaze，并安装当前版本的Host/API及Kernel算子包，参见[编译安装说明](../../../docs/zh/install/build.md)。Kernel UT还需要tikicpulib。
- TTK需要`ml-dtypes`、`en-dtypes>=0.0.4`。ACLNN、E2E和NPU Golden需要`torch`、`torch_npu`；E2E还需要`cann_ops_nn`。
- 测试NZ或FP4时，TTK须支持完整NZ存储的传递，以及FP4自定义NPU Golden。

## Kernel UT

在ops-nn根目录执行：

```bash
bash build.sh -u --opkernel --ops=quant_matmul_activation_quant --soc=ascend950
```

用例见[Kernel UT CSV](ut/op_kernel/test_quant_matmul_activation_quant.csv)。修改输入或计算参数时，须同步更新期望值`expectedY`和`expectedYScale`。

## TTK ST

| 测试入口 | 子命令 | 用例文件 | 融合算子的设备耗时列 |
| --- | --- | --- | --- |
| Kernel | `kernel` | [Kernel CSV](st/arch35/ttk_kernel_quant_matmul_activation_quant_st.csv) | `dyn_perf_us`（默认动态编译） |
| ACLNN ND | `aclnn` | [ND CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv) | `op_perf_us` |
| ACLNN NZ | `aclnn` | [NZ CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_weight_nz_st.csv) | `op_perf_us` |
| E2E | `e2e` | [E2E CSV](st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv) | `eager_device_perf_us` |

以ACLNN ND为例，在ops-test-kit目录运行：

```bash
cd /path/to/ops-test-kit
OP_TESTS=/path/to/ops-nn/matmul/quant_matmul_activation_quant/tests
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" --seed 42
```

其他入口替换子命令和CSV即可。默认使用CPU Golden进行精度检查，结果CSV的`precision_status`应为`PASS`。

本算子的E2E交付和验收范围为PyTorch eager，仅验证单算子调用：

```bash
python3 -m ttk e2e \
  -i "$OP_TESTS/st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" --golden-mode Enable --seed 42 \
  --eager=true -c=false -d=false -o e2e_results.csv
```

检查`eager_precision`有精度结果，且`precision_status`为`PASS`。静态图和动态图不在交付范围内。

### 用例配置

- Kernel NZ：`input_shapes`填写物理形状，`input_ori_shapes`填写逻辑形状，`input_ori_formats`填写`ND`。
- ACLNN NZ：`tensor_view_shapes`填写逻辑形状，`tensor_storage_shapes`填写完整物理形状。例如FP8权重的逻辑形状为`(batch,R,C)`时，物理形状为`(batch,ceil(C/32),ceil(R/16),16,32)`。
- E2E：CSV中的API名称为`cann_ops_nn.ops.quant_matmul_activation_quant`，Tensor顺序为`x1,x2,x2_scale,x1_scale,bias`；矩阵的逻辑视图为`(...,M,K)`、`(...,K,N)`。
- 输出类型在Kernel CSV中使用`y_dtype`，ACLNN CSV中使用`yDtype`，E2E CSV中使用`output_dtype`。这些是测试配置字段；ACLNN接口本身通过输出Tensor描述数据类型。枚举值按对应CSV填写。
- E2E FP4使用uint8打包，并设置`x1_dtype=x2_dtype=output_dtype=296`，布局见[PyTorch接口文档](../docs/torchapi_quant_matmul_activation_quant.md#布局与存储)。

## 与独立算子组合的性能对比

使用`--golden-mode NPU`，可测量以下参考组合。QBMM表示量化矩阵乘，括号内为该步骤的输出类型。

| 激活 | 量化类型 | 参考组合 |
| --- | --- | --- |
| SiLU | MXFP8 | QBMM(FP32) → SiLU(FP32) → Cast(BF16) → DynamicMxQuant |
| SwiGLU | MXFP8 | QBMM(BF16) → SwigluMxQuant（沿尾轴等分，激活左半部分，mode=0） |
| GELU（tanh/erf） | MXFP8、MXFP4 | QBMM(FP32) → GELU(FP32) → Cast(BF16) → DynamicMxQuant |

Cast计入参考组合耗时。SwiGLU需要`torch_npu.npu_swiglu_mx_quant`。量化算法和舍入模式按用例参数执行。

例如，测试ND SiLU：

```bash
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" -t silu_nd_e4m3_n65 \
  --golden-mode NPU --devices 0 --process-count 1 \
  --warmup true --run 1 --seed 42 -o silu_perf.csv
```

测试其他激活时，`-t`可改为`swiglu_s0_y36`、`gelu_tanh_s0_y36`或`gelu_tanh_fp4_nd_rint`。保持预热开启，使NZ权重的格式准备在预热阶段完成。

结果中的`golden_device_perf_us`为参考组合的设备耗时；融合算子的耗时列见上表。加速比按“参考组合耗时 / 融合算子耗时”计算，仅使用`golden_source=npu`且耗时为正的结果。`golden_kernel_details`记录参考组合中各算子的耗时和调用次数。

当前NPU Golden每例采样一次，`--run`不增加其采样次数。需要多次采样时，串行重复运行、分别保存结果并取中位数。

## 精度判定

精度验收使用默认的`--golden-mode Enable`，由CPU Golden计算期望值。NPU模式以独立算子的组合输出为参考；其中SwiGLU在激活前舍入为BF16，与融合算子的舍入位置不同，因此该模式用于性能对比。

ST分别检查`y`和`y_scale`：形状和dtype一致，至少99.9%的元素误差不超过1 ULP，NaN位置及Inf符号匹配，正负零视为相等。FP4按解包后的元素比较。Kernel UT使用固定输入并严格比较期望结果。
