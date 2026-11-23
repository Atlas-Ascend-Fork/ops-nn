# situ_glu_group_quant

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

- 接口功能：稀疏SiTU门控激活与MXFP8（e4m3fn，blocksize=32）尾轴动态量化融合算子，面向MoE分组场景。输入`x`末维为拼接的gate/up两半（2\*EPR），输出量化结果`y`（float8_e4m3fn）与缩放因子`y_scale`（float8_e8m0）；激活结果驻留片上，与串联两个小算子`npu_situ_and_mul_sparse -> npu_dynamic_mx_quant(dst=e4m3fn)`在有效行内逐位一致。
- 计算公式：同[aclnnSituGluGroupQuant](./aclnnSituGluGroupQuant.md)功能说明（激活在fp32域单次退出舍入；每32元素一组按OCP MX语义量化；可选weight按行缩放）。

## 函数原型

```python
cann_ops_nn.situ_glu_group_quant(
    x,
    weight=None,
    group_index=None,
    beta=1.0,
    linear_beta=0.0,
    activate_left=False,
    axis=-1,
    quant_mode=0,
    dst_type=36,
    dst_type_max=0.0,
    round_mode='rint',
    group_list_type=0
) -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | 待激活量化的数据，末维为拼接的gate/up两半（2*EPR），末维必须为正偶数。 | float16、bfloat16 | 1-24维（y_scale比x多一维，框架shape上限25维）；EPR=末维/2 ∈ [128, 3072]且为32的倍数 |
| `weight` | Tensor | 可选 | 逐token（行）缩放因子，按全局行号索引；元素个数必须等于`x`展开后的总行数；不传则不缩放。 | float32 | 1维 |
| `group_index` | Tensor | 可选 | 每个专家分到的token行数；不传为全行模式，传入为分组语义（validRows = min(sum,总行数)，padding行不写出）。必须配合`group_list_type=1`。 | int64 | 1维 |
| `beta` | float | 可选 | gate支路的SiTU激活参数，必须大于0。建议取值在[2^-10, 2^10]范围内：极大beta配合小幅度输入时，激活链的浮点大数相消会使结果退化为0（指令序实现口径，与上游实现逐位一致）。默认`1.0`。 | - | - |
| `linear_beta` | float | 可选 | up支路软限幅参数：`<=0`禁用（up直通），`>0`启用（幅度钳位 ±linear_beta）。建议取值在[2^-10, 2^10]范围内（同beta的相消精度边界）。默认`0.0`（禁用）。 | - | - |
| `activate_left` | bool | 可选 | gate/up方向选择：`True`表示gate为前半、up为后半；`False`表示gate为后半、up为前半。默认`False`。 | - | - |
| `axis` | int | 可选 | 量化轴，当前仅支持`-1`。默认`-1`。 | - | - |
| `quant_mode` | int | 可选 | 量化模式，当前仅支持`0`（MX量化）。默认`0`。 | - | - |
| `dst_type` | int | 可选 | 目标量化数据类型，当前仅支持`36`（FLOAT8_E4M3FN）。默认`36`。 | - | - |
| `dst_type_max` | float | 可选 | 目标数据类型数值上限，取值 >= 0，当前不参与计算。默认`0.0`。 | - | - |
| `round_mode` | str | 可选 | 舍入模式，当前仅支持`'rint'`。默认`'rint'`。 | - | - |
| `group_list_type` | int | 可选 | **默认值为`0`（cumsum预留模式，当前会被拒收），不传即报错；必须显式传`1`（count模式）方可调用。** | - | - |

## 返回值说明

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | 量化结果。 | float8_e4m3fn | `x.shape[:-1] + [EPR]` |
| `y_scale` | Tensor | MX量化缩放因子，每32个元素一组对应一个scale。 | float8_e8m0 | `x.shape[:-1] + [ceil(ceil(EPR/32)/2), 2]` |

## 约束说明

- 该接口支持训练、推理场景下使用。
- 执行模式：支持eager（单算子直接调用）与TorchAir图模式调用（converter见`torch_extension/graph_convert_situ_glu_group_quant.py`）；
- `x`末维必须为正偶数，EPR ∈ [128, 3072]且为32的倍数（防呆拒收）。
- `weight`元素个数必须等于`x`展开后的总行数（防呆拒收）。
- `beta`必须大于0；`linear_beta`任意取值合法（≤0禁用软限幅，>0启用）；`axis`/`quant_mode`/`dst_type`/`round_mode`/`group_list_type`当前仅支持默认合法值，非法值防呆拒收。**注意：`group_list_type`默认值为0（预留模式），按默认签名调用（即使不传`group_index`）也会被拒收，必须显式传`group_list_type=1`。**
- 分组语义下padding行不写出，其内容为分配器残留，调用方不应读取。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, EPR = 128, 512
  x = torch.randn(M, 2 * EPR, dtype=torch.bfloat16).npu()
  group_index = torch.full((28,), M // 28, dtype=torch.int64).npu()

  # 分组量化（28 专家，MoE 场景）
  y, y_scale = cann_ops_nn.situ_glu_group_quant(x, group_index=group_index, group_list_type=1)
  print(y.shape, y_scale.shape)  # torch.Size([128, 512]) torch.Size([128, 8, 2])

  # 全行模式 + 逐token weight 缩放
  w = torch.rand(M, dtype=torch.float32).npu() + 0.5
  y, y_scale = cann_ops_nn.situ_glu_group_quant(x, weight=w, beta=1.0, linear_beta=2.0,
                                                activate_left=False, group_list_type=1)
  ```

- TorchAir图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, x, group_index):
          return cann_ops_nn.situ_glu_group_quant(x, group_index=group_index,
                                                  beta=1.0, linear_beta=2.0,
                                                  activate_left=True, group_list_type=1)

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  M, EPR = 128, 512
  x = torch.randn(M, 2 * EPR, dtype=torch.bfloat16).npu()
  group_index = torch.full((28,), M // 28, dtype=torch.int64).npu()
  y, y_scale = model(x, group_index)
  ```
