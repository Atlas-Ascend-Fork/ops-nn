# rotate\_dual\_level\_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

对输入沿尾轴分块做旋转矩阵乘法，按需对旋转结果执行 clamp，再依次执行一级归一化和二级 MX 量化。接口返回量化结果 `y`、FP32 一级尺度 `level0_scale` 和 E8M0 二级尺度 `level1_scale`。

令 `N=x.shape[-1]`，`M` 为前导维合并后的行数，一维输入时 `M=1`。`K` 为旋转矩阵的阶数，`B=N/K` 为每行的旋转块数。记输入第 `m` 行第 `b` 块为 $X_{m,b}$，对应的旋转矩阵为 $R_{e(m),b}$。旋转结果为

$$
Z_{m,b}=\operatorname{cast}_{\mathrm{BF16}}\left(X_{m,b}R_{e(m),b}\right).
$$

普通模式使用共享旋转矩阵或按列块选择矩阵。分组模式下，`e(m)` 表示第 `m` 行所属的组，可在组内共享旋转矩阵，也可按列块选择矩阵。

clamp 启用时，每 32 个元素独立计算裁剪阈值。一级归一化沿原始尾轴每 512 个元素计算一个 FP32 尺度。二级 MX 量化对归一化结果每 32 个元素计算一个 E8M0 尺度。数学公式、舍入规则和尺度边界参见 [aclnnRotateDualLevelQuant](aclnnRotateDualLevelQuant.md)。

## 函数原型

`cann_ops_nn.rotate_dual_level_quant` 的函数签名如下。

```python
def rotate_dual_level_quant(
    x,
    rotation,
    *,
    alpha=None,
    dst_dtype=None,
    axis=-1,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    transpose_y=False,
    group_list=None,
    group_list_type=0,
    level0_block_size=512,
    level1_block_size=32,
):
    ...
```

按 `(y, level0_scale, level1_scale)` 的顺序返回三个 Tensor。导入扩展包后，也可用 `torch.ops.cann_ops_nn.rotate_dual_level_quant` 调用。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | 必选 | 待旋转量化的 NPU 张量。普通模式支持 1～7 维，分组模式仅支持二维。 | `torch.float16`、`torch.bfloat16` | 普通模式 `(*, N)`；分组模式 `(M, N)` |
| rotation | Tensor | 必选 | 旋转矩阵，与 `x` 的数据类型、设备一致。共享矩阵与按列块矩阵的选择见约束说明。 | `torch.float16`、`torch.bfloat16` | 普通模式 `(K, K)` 或 `(B, K, K)`；分组模式 `(E, K, K)` 或 `(E, B, K, K)` |
| alpha | Tensor | 可选 | clamp 系数，默认 `None`。传入时须与 `x` 同设备，值在 `(0.0, 1.0)` 内时才启用 clamp。 | `torch.bfloat16` | `(1,)` |
| dst_dtype | int 或 torch.dtype | 可选 | 量化目标类型，默认 `None`，按 FP4 处理。FP4 接受 `torch_npu.float4_e2m1fn_x2`（`296`）。FP8 接受 `torch.float8_e4m3fn`（`24` 或 `292`）和 `torch.float8_e5m2`（`23` 或 `291`）。 | - | - |
| axis | int | 可选 | 量化轴，仅支持 `-1` 或 `x.dim() - 1`。默认 `-1`，传入 `None` 时取默认值。 | - | - |
| round_mode | str | 可选 | 最终量化的舍入方式。默认 `"rint"`，传入 `None` 时取默认值。与 `dst_dtype` 的组合见约束说明。 | - | - |
| scale_alg | int | 可选 | 二级 MX 尺度的计算方法。默认 `0`，传入 `None` 时取默认值。与 `dst_dtype` 的组合见约束说明。 | - | - |
| dst_type_max | float | 可选 | 二级尺度计算使用的范围参数。默认 `0.0`，传入 `None` 时取默认值。与 `scale_alg` 的组合见约束说明。 | - | - |
| transpose_y | bool | 可选 | 是否转置量化结果。默认且仅支持 `False`，传入 `None` 时取默认值。 | - | - |
| group_list | Tensor | 可选 | M 轴连续分组的行数列表，与 `x` 同设备。默认 `None`，表示普通模式。传入 Tensor 时，组数 `E` 须在 `[1, 1024]` 内。 | `torch.int64` | `(E,)` |
| group_list_type | int | 可选 | 分组列表的表示方式。`0` 表示累计行数，`1` 表示各组行数，默认 `0`。普通模式也须传入合法取值。 | - | - |
| level0_block_size | int | 可选 | 一级量化块大小。默认且仅支持 `512`。 | - | - |
| level1_block_size | int | 可选 | 二级 MX 量化块大小。默认且仅支持 `32`。 | - | - |

## 返回值说明

三个输出均位于 `x.device`。下表中 `*` 表示与 `x` 相同的前导维。

| 输出名 | 输出类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| y | Tensor | 必选 | 量化结果。FP4 以打包字节返回，每字节的低 4 位对应前一个 E2M1 元素，高 4 位对应后一个元素。FP8 以指定类型返回。 | FP4 为 `torch.uint8`；FP8 为 `torch.float8_e4m3fn` 或 `torch.float8_e5m2` | FP4 为 `(*, N/2)`；FP8 与 `x.shape` 相同 |
| level0_scale | Tensor | 必选 | 一级尺度，沿每行尾轴每 512 个元素对应一个尺度。 | `torch.float32` | `(*, ceil(N/512))` |
| level1_scale | Tensor | 必选 | 二级 MX 尺度，沿每行尾轴每 32 个元素对应一个尺度。最后一维按两个 E8M0 编码成对存放。 | `torch.float8_e8m0fnu` | `(*, ceil(N/64), 2)` |

FP4 的打包 shape 是 PyTorch 侧的存储形状，传入 ACLNN 的 FP4 描述符使用与 `x` 一致的逻辑 shape。每行有效二级尺度数为 `ceil(N/32)`，为奇数时末字节填入 E8M0 编码 0。编码 0 对应 `2^-127`，该填充位置应从有效尺度中排除。

## 约束说明

### 输入与旋转矩阵

- 所有输入 Tensor 均使用基础存储格式。绑定层转换为 ND 描述符，保留逻辑 shape、stride 和 storage offset，非连续输入由 ACLNN 连续化。
- `rotation` 的最后两维相等。`K` 仅支持 `32`、`64`、`128`、`256`，且 `N` 须为 `K` 的整数倍。`x` 的各维及合并后的 `M` 均不得超过 `INT32_MAX`。
- 普通模式下，`(K, K)` 表示共享旋转矩阵，`(B, K, K)` 表示第 `b` 个列块使用 `rotation[b]`。支持满足 shape 约束的空输入，接口返回对应的空输出。
- 分组模式下，`x` 为 `(M, N)`，允许 `M=0`，且 `N>0`。`rotation` 首维须等于 `group_list` 的长度 `E`。`(E, K, K)` 表示组内共享 `rotation[e]`，`(E, B, K, K)` 表示按组及列块使用 `rotation[e, b]`。

### 分组与量化参数

- `group_list_type=0` 时，`group_list` 须非负且单调不减，最后一个值不超过 `M`。`group_list_type=1` 时，各组行数须非负，总和不超过 `M`。`M=0` 时，各组行数均须为 0。
- 分组从第 0 行起连续划分，允许空组和仅覆盖前缀。空组仍占用对应的旋转矩阵下标。调用方负责保证设备侧列表的值满足约束。接口新分配三个输出，未覆盖行的值未定义，调用方应只使用已覆盖行。
- FP4 支持 `round_mode` 为 `"rint"`、`"round"` 或 `"floor"`，`scale_alg` 为 `0` 或 `2`。FP8 仅支持 `"rint"`，`scale_alg` 为 `0` 或 `1`。
- `scale_alg=2` 时，`dst_type_max` 须为 `[6.0, 12.0]` 内的有限值，校验后转换为 FP32 再选择二级尺度算法。其他场景须传入 `0.0`。

### 执行模式

需要支持原生 `torch.float8_e8m0fnu` 的 PyTorch。接口支持 eager 执行和 Meta 推导。Meta 路径根据形状、类型等元数据推导输出。设备型号由 ACLNN 检查，分组列表的取值由调用方保证。当前未提供 TorchAir GE converter。

## 确定性计算

默认支持确定性计算。

## 调用示例

### 构建与安装

在 Linux 环境下使用相互匹配的 PyTorch 和 torch_npu、CANN Toolkit 及 Ascend 950 设备。先在仓库根目录构建自定义算子包，按安装包的说明安装并加载环境，再构建并安装 PyTorch 单算子 wheel。通用步骤见 [torch_extension 构建与安装](../../../../torch_extension/README.md) 和 [自定义算子包调用](../../../../docs/zh/invocation/quick_op_invocation.md)。

```bash
bash build.sh --pkg --experimental --soc=ascend950 --ops=rotate_dual_level_quant --vendor_name=custom

# 安装自定义算子包并加载其环境后执行以下命令。
bash build.sh --torch_extension --experimental --ops=rotate_dual_level_quant --vendor_name=custom
python3 -m pip install build_out/cann_ops_nn_custom-*.whl
```

以下示例从该单算子 wheel 导入 `cann_ops_nn_custom`。使用包含本算子的通用 wheel 时，可直接导入 `cann_ops_nn`。首次 eager 调用会即时编译 C++ 绑定并调用 `aclnnRotateDualLevelQuant`。

### 单算子模式调用（eager）

```python
import torch
import torch_npu
import cann_ops_nn_custom as cann_ops_nn

M, N, K = 4, 544, 32
x = torch.randn(M, N, dtype=torch.bfloat16, device="npu:0")
rotation = torch.eye(K, dtype=x.dtype, device=x.device)
y, level0_scale, level1_scale = cann_ops_nn.rotate_dual_level_quant(x, rotation)

# 默认 FP4。y 为 uint8，shape 为 (4, 272)。
# 两级尺度的 shape 分别为 (4, 2) 和 (4, 9, 2)。
```

### 带可选输入的分组调用

```python
import torch
import torch_npu
import cann_ops_nn_custom as cann_ops_nn

M, N, K, E = 4, 544, 32, 3
x = torch.randn(M, N, dtype=torch.bfloat16, device="npu:0")
rotation = torch.randn(E, K, K, dtype=x.dtype, device=x.device)
alpha = torch.tensor([0.95], dtype=torch.bfloat16, device=x.device)
group_list = torch.tensor([0, 2, 1], dtype=torch.int64, device=x.device)
group_cumsum = torch.tensor([0, 2, 3], dtype=torch.int64, device=x.device)

for groups, group_type in ((group_list, 1), (group_cumsum, 0)):
    y, level0_scale, level1_scale = cann_ops_nn.rotate_dual_level_quant(
        x,
        rotation,
        alpha=alpha,
        dst_dtype=torch.float8_e4m3fn,
        scale_alg=1,
        group_list=groups,
        group_list_type=group_type,
    )
    # 空组占用 rotation[0]，只有前三行的输出有效。
    y, level0_scale, level1_scale = y[:3], level0_scale[:3], level1_scale[:3]
```
