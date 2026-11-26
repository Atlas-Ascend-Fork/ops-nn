# DynamicDualLevelMxQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: performs MX quantization with a destination data type of the FLOAT4 class. Only the trailing axis is quantized, and all preceding axes are merged into one axis. The input is divided into multiple data blocks according to the given level0_block_size, each data block undergoes first-level quantization, and the quantization scale level0_scale is output. Then the first-level quantization result is used as the new input, divided into multiple data blocks according to the given level1_block_size, each data block undergoes second-level quantization, and the quantization scale level1_scale is output. The data type conversion is performed according to round_mode to obtain the quantization result y.
  - Optional function: fuses the smooth scale operation. Before quantizing the data input x, x=x*smooth_scale (broadcast element-wise multiplication) is performed first.

- Formula:
  - Groups the input x along the trailing axis by $k_0$ = level0_block_size values. A group of $k_0$ values $\{\{x_i\}_{i=1}^{k_0}\}$ is dynamically quantized into $\{level0_scale, \{temp_i\}_{i=1}^{k_0}\}$, where $k_0$ = level0_block_size. Then groups temp along the trailing axis by $k_1$ = level1_block_size values. A group of $k_1$ values $\{\{temp_i\}_{i=1}^{k_1}\}$ is dynamically quantized into $\{level1_scale, \{y_i\}_{i=1}^{k_1}\}$, where $k_1$ = level1_block_size

  $$
  input\_max_i = max_i(abs(x_i))
  $$

  $$
  level0\_scale_i = input\_max_i / (FP4\_E2M1\_MAX)
  $$

  $$
  temp_i = cast\_to\_x\_type(x_i / level0\_scale_i), \space i\space from\space 1\space to\space level0\_block\_size
  $$

  - The calculation algorithm of level1_scale_i is specified by scale_alg:
    - When scale_alg is 0 (OCP Microscaling Formats (Mx) Specification):
      $$
      shared\_exp_i = floor(log_2(max_i(|temp_i|))) - emax
      $$
      $$
      level1\_scale_i = 2^{shared\_exp_i}
      $$
    - When scale_alg is 1 (ceil rounding algorithm):
      $$
      S_{fp32} = \frac{max_i(|temp_i|)}{Amax(DType)}
      $$
      where $Amax(DType)$ is the maximum value 6 of FLOAT4_E2M1. The unbiased exponent of $S_{fp32}$ is rounded up (the exponent is incremented by 1 when $S_{fp32}$ is a normal number with a non-zero mantissa, or a subnormal number with a mantissa greater than 0.5), and $level1\_scale_i = 2^{E_{int}}$.
    - When scale_alg is 2 (Dynamic Dtype Range algorithm):
      $$
      S_{fp32} = \frac{max_i(|temp_i|)}{Amax(DType)}
      $$
      where $Amax(DType)$ is the value of dst_type_max. When dst_type_max is 0, the maximum value 6 of FLOAT4_E2M1 is used. When dst_type_max is 0, 6, or 7, the exponent is rounded up by the BF16 high-mantissa-bit carry method; for other values, the exponent is incremented by 1 only when $S_{fp32}$ is a normal number with a non-zero mantissa. $level1\_scale_i = 2^{E_{int}}$.

  $$
  y_i = cast\_to\_FP4\_E2M1(temp_i/level1\_scale_i, round\_mode), \space i\space from\space 1\space to\space level1\_block\_size
  $$

  - The quantized $y_{i}$ values form the output y at the positions corresponding to $x_{i}$. The $level0\_scale_i$ values form the output $level0\_scale$ according to the groups of the trailing axis, and the $level1\_scale_i$ values form the output $level1\_scale$ according to the groups of the trailing axis.

  - max_i denotes the maximum value in the i-th group.

  - emax: the exponent bits of the maximum regular number of the corresponding data type.

      |   DataType    | emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Indicates the input tensor of the operator, corresponding to <em>x</em><sub>i</sub> in the formula;</li><li>The shape supports 1-7 dimensions, and the last dimension must be an even number.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale</td>
      <td>Optional input</td>
      <td><ul><li>Indicates the input used for the smooth scale operation;</li><li>When the smooth scale operation does not need to be fused, pass nullptr for smooth_scale;</li><li>When smooth_scale is not nullptr, the dtype of smooth_scale must be the same as that of x, and the shape is 1-dimensional with a length equal to the last dimension of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16 (and the same as input x)</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the mode of data conversion, corresponding to round_mode in the formula;</li><li>Supports {"rint", "round", "floor"};</li><li>The default value is "rint".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level0_block_size</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the block_size of the first-level quantization, corresponding to level0_block_size in the formula;</li><li>The input range is {512}.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level1_block_size</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the block_size of the second-level quantization, corresponding to level1_block_size in the formula;</li><li>The input range is {32}.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the calculation method of level1_scale;</li><li>Values 0, 1, and 2 are supported, indicating the OCP Microscaling Formats (Mx) Specification, the ceil rounding algorithm, and the Dynamic Dtype Range algorithm, respectively;</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the maximum value of the destination data type, corresponding to Amax(DType) in the formula;</li><li>Effective only when scale_alg is 2. Values 0.0 and 6.0-12.0 are supported. When the value is 0.0, the maximum value 6 of FLOAT4_E2M1 is used;</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Indicates the result corresponding to the quantized input x, corresponding to <em>y</em><sub>i</sub> in the formula;</li><li>The shape is the same as that of input x.</li></ul></td>
      <td>FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>level0_scale</td>
      <td>Output</td>
      <td><ul><li>Indicates the scale of the first-level quantization, corresponding to level0_scale in the formula;</li><li>The value of the shape on the trailing axis is the value of the trailing axis of x divided by level0_block_size and rounded up</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>level1_scale</td>
      <td>Output</td>
      <td><ul><li>Indicates the scale of the second-level quantization, corresponding to level1_scale in the formula;</li><li>The size of the shape is the dim of x + 1;</li><li>The values of the shape on the last two axes are ((ceil(x.shape[-1] / level1_block_size) + 2 - 1) / 2, 2), with even padding applied, where the pad fill value is 0.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The shape constraints of x, level0_scale, and level1_scale are as follows:
    - rank(level1_scale) = rank(x) + 1.
    - level0_scale.shape[-1] = ceil(x.shape[-1] / level0_block_size).
    - level1_scale.shape[-2] = (ceil(x.shape[-1] / level1_block_size) + 2 - 1) / 2.
    - level1_scale.shape[-1] = 2.
    - The other dimensions are the same as those of input x.
- scale_alg constraints:
    - scale_alg supports values 0, 1, and 2.
    - dst_type_max is effective only when scale_alg is 2. When scale_alg is not 2, dst_type_max must be 0.0.
    - dst_type_max supports 0.0 or 6.0-12.0.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface (V1)  | [test_aclnn_dynamic_dual_level_mx_quant](examples/arch35/test_aclnn_dynamic_dual_level_mx_quant.cpp) | Invokes the DynamicDualLevelMxQuant operator through the [aclnnDynamicDualLevelMxQuant](docs/aclnnDynamicDualLevelMxQuant_en.md) interface; scale_alg defaults to 0 when it is not provided. |
| aclnn interface (V2)  | [test_aclnn_dynamic_dual_level_mx_quant_v2](examples/arch35/test_aclnn_dynamic_dual_level_mx_quant_v2.cpp) | Invokes the DynamicDualLevelMxQuant operator through the [aclnnDynamicDualLevelMxQuantV2](docs/aclnnDynamicDualLevelMxQuantV2_en.md) interface; supports scale_alg values 0, 1, and 2. |
| Graph mode | -  | Invokes the DynamicDualLevelMxQuant operator through the [operator IR](op_graph/dynamic_dual_level_mx_quant_proto.h) graph construction method.         |
