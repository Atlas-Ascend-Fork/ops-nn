# aclnnWeightQuantBatchMatmulV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:23:56.911Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Operator Function**: Performs matrix multiplication computation with pseudo-quantization input and supports quantization computation for the output. Compared with the aclnnWeightQuantBatchMatmulV2 API, this API has the following changes:

  The innerPrecise parameter is added to support the selection of high-precision or high-performance computation modes. In the A16W4 per_group scenario, when batchSize <= 16, it can be set to 1 to improve performance.
- **Formula**:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  In the formula, $weight$ is the input in the pseudo-quantization scenario, and its dequantization formula $ANTIQUANT(weight)$ is

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  When the user configures the quantScaleOptional input, the output is quantized. The quantization formula is as follows:

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  When quantScaleOptional is set to nullptr, the output is directly:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnWeightQuantBatchMatmulV3GetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call "aclnnWeightQuantBatchMatmulV3" to execute the computation.

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *antiquantScale,
  const aclTensor *antiquantOffsetOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *biasOptional,
  int              antiquantGroupSize,
  int              innerPrecise,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnWeightQuantBatchMatmulV3GetWorkspaceSize

- **Parameters**
  <table style="table-layout: fixed; width: 1550px">
    <colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
    </colgroup>
    <thread>
      <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Input `x` in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>Two dimensions are supported. Shape is (m, k), where Reduce dimension k must be the same size as the Reduce dimension k of `weight`.</td>
        <td>Non-contiguous tensors only support the transpose use case.</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Input `weight` in the formula.</td>
        <td>-</td>
        <td>INT8, INT4, INT32, FLOAT8_E4M3FN<sup>2</sup>, HIFLOAT8<sup>2</sup>, FLOAT<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
        <td>ND, FRACTAL_NZ</td>
        <td>Supports (k, n)</td>
        <td>Non-contiguous tensors only support the transpose use case.</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Antiquant scale parameter used to implement the input antiquant computation, which is the input `antiquantScale` in the antiquant formula.</td>
        <td>When the data type is FLOAT16 or BFLOAT16, the data type must be consistent with that of input `x`. When the data type is UINT64 or INT64, x only supports FLOAT16 and non-transpose, weight only supports int8 and ND transpose, the mode only supports per_channel, quantScaleOptional and quantOffsetOptional must be null pointers, m only supports [1, 96], and k and n must be aligned to 64. You need to first call the aclnnCast API to complete the conversion of FLOAT16 to FLOAT32, and then call the aclnnTransQuantParamV2 API to complete the conversion of FLOAT32 to UINT64, <a href="../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2.md" target="_blank">for details, see</a></td>
        <td>FLOAT16, BFLOAT16, UINT64<sup>1</sup>, INT64<sup>1</sup>, FLOAT8_E8M0<sup>2</sup></td>
        <td>ND</td>
        <td>          <ul>
            <li>per_tensor mode: input shape is (1,) or (1, 1).</li>
            <li>per_channel mode: input shape is (1, n) or (n,).</li>
            <li>per_group mode: input shape is (ceil(k, group_size), n).</li>
          </ul>
        </td>
        <td>Non-contiguous tensors are supported only in the transpose use case.</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Anti-quantization offset parameter for input dequantization. Input `antiquantOffset` in the dequantization formula.</td>
        <td>When the data type is FLOAT16 or BFLOAT16, the data type must be consistent with that of input `x`. When the data type is INT32, the data range is limited to [-128, 127], x only supports FLOAT16, weight only supports int8, and antiquantScale only supports UINT64/INT64. Optional input. Null pointer when not needed.</td>
        <td>FLOAT16, BFLOAT16, INT32<sup>1</sup></td>
        <td>ND</td>
        <td>Must be consistent with `antiquantScale`.</td>
        <td>Non-contiguous tensors only support the transpose scenario.</td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>Input</td>
        <td>Quantization parameter for output quantization computation, converted from the quantScale and quantOffset data in the quantization formula through the `aclnnTransQuantParam` API.</td>
        <td>Converted from the `quantScale` and `quantOffset` data in the quantization formula through the `aclnnTransQuantParam` API.</td>
        <td>UINT64<sup>1</sup></td>
        <td>ND</td>
        <td>          <ul>
            <li>per_tensor mode: input shape is (1,) or (1, 1).</li>
            <li>per_channel mode: input shape is (1, n) or (n,).</li>
          </ul></td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffsetOptional</td>
        <td>Input</td>
        <td>Quantization offset parameter for implementing output quantization computation, which is the input `quantOffset` in the quantization formula.</td>
        <td>Optional input. Null pointer when not needed</td>
        <td>FLOAT<sup>1</sup></td>
        <td>ND</td>
        <td>Same as `quantScaleOptional`</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>Bias input. The input `bias` in the formula. If the data type of input `x` is BFLOAT16, the data type must be BFLOAT16 or FLOAT. If the data type of input `x` is FLOAT16, the data type must be FLOAT16.</td>
        <td>Optional input. Null pointer when not needed.</td>
        <td>FLOAT, FLOAT16, BFLOAT16<sup>2</sup></td>
        <td>ND</td>
        <td>1-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantGroupSize</td>
        <td>Input</td>
        <td>Denotes the groupSize for the dequantization computation on the input `weight` in the pseudo-quantization per_group algorithm mode. Describes the size, in the Reduce direction, of the data to be dequantized corresponding to a group of dequantization parameters. Pass in 0 when the pseudo-quantization algorithm mode is not per_group. When the pseudo-quantization algorithm mode is per_group, the value range is [32, k-1] and the value must be a multiple of 32.</td>
        <td>-</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Indicates whether pseudo-quantization is in high-precision or high-performance computation mode (only 0 or 1 can be passed in). In the A16W4 per_group scenario where batchSize<=16, set this parameter to 1 and set the weight data format to FRACTAL_NZ to improve performance. This parameter is not recommended for other scenarios. It is recommended to pass in 0.</td>
        <td>-</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
       <tr>
        <td>y</td>
        <td>Output</td>
        <td>Computation output, `y` in the formula.</td>
        <td>When `quantScaleOptional` exists, the data type is INT8; when `quantScaleOptional` does not exist, the supported data types are FLOAT16 and BFLOAT16, which must be consistent with the data type of input `x`.</td>
        <td>INT8<sup>1</sup>, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied on the Device side.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation flow.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </tbody>
  </table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    - The superscript "1" in the data type column of the table above indicates a data type not supported by this series.

  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    - The superscript "2" in the data type column of the table above indicates a data type not supported by this series.

  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If a required parameter is passed a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The shape dimensions of x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, or y do not meet the requirement.</td>
    </tr>
    <tr>
      <td>The data types of x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, or y are not within the supported range.</td>
    </tr>
    <tr>
      <td>The reduce dimensions (k) of x and weight are inconsistent.</td>
    </tr>
    <tr>
      <td>When antiquantOffsetOptional has an input, its shape is inconsistent with antiquantScale.</td>
    </tr>
    <tr>
      <td>When quantOffsetOptional has an input, its shape is inconsistent with quantScale.</td>
    </tr>
    <tr>
      <td>The value of antiquantGroupSize does not meet the requirement.</td>
    </tr>
    <tr>
      <td>The value of innerPrecise does not meet the requirement.</td>
    </tr>
    <tr>
      <td>If quantOffsetOptional exists, quantScaleOptional is a null pointer.</td>
    </tr>
    <tr>
      <td>The values of input k and n are not within the range [1, 65535].</td>
    </tr>
    <tr>
      <td>If the x matrix is not transposed, m is not within the range [1, 2^31-1]; if the x matrix is transposed, m is not within the range [1, 65535].</td>
    </tr>
    <tr>
      <td>Empty tensors are not supported.</td>
    </tr>
    <tr>
      <td>The continuity of the passed in x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, and y does not meet the requirement.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The product model is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnWeightQuantBatchMatmulV3

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace applied on the side of the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the side of the Device, obtained by the first-phase API aclnnWeightQuantBatchMatmulV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor that contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<a id="a2_a3_series_products"></a>

<!-- npu="A3,910b" id9 -->
<details>
<summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term></summary>

  - **Deterministic computation description**: Defaults to a non-deterministic implementation. Supports enabling deterministic computation through aclrtCtxSetSysParamOpt.

  - **General Constraints**
    - When the weight data format is FRACTAL_NZ and the data type is INT4/INT32, or when the weight data format is ND and the data type is INT32, this is supported only in the INT4Pack scenario. You need to use the aclnnConvertWeightToINT4Pack API to complete the conversion of INT32 to INT4Pack and the conversion of ND to FRACTAL_NZ, <a href="../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md" target="_blank">For details, see</a>. If the data type is INT4, the innermost dimension of weight should be an even number.
    - For different pseudo-quantization algorithm modes, the weight data format FRACTAL_NZ is supported only in the following scenarios:
      - per_channel mode:
          The weight data type is INT8 and the y data type is non-INT8.
          The weight data type is INT4/INT32, weight is transposed, and the y data type is non-INT8.
      - per_group mode: The weight data type is INT4/INT32, weight is non-transposed, x is non-transposed, antiquantGroupSize is 64 or 128, k is aligned to antiquantGroupSize, n is aligned to 64, and the y data type is non-INT8.

  - **Performance Optimization Suggestions**
    - per_channel mode: To improve performance, it is recommended to use the transposed weight input. When the m range is [65, 96], it is recommended to use antiquantScale with the data type UINT64/INT64.
    - per_group mode: In the A16W4 scenario with batchSize <= 16, you can set the innerPrecise parameter to 1 and set the weight data format to FRACTAL_NZ to improve performance, but this may cause a certain degree of precision degradation.

</details>
<!-- end id9 -->

<a id="ascend_950pr_ascend950dt"></a>

<!-- npu="950" id10 -->
<details>
<summary><term>Ascend 950PR&950DT products</term></summary>

  - **Deterministic computation description**: Defaults to a deterministic implementation.

  <a id="general-constraints"></a>

  - **General Constraints**
    - The m, k, and n dimensions of the `x` and `weight` matrices are in the range [1, 2^31-1]. The Reduce dimension k of `weight` must be equal to the Reduce dimension k of `x`.
    - Supported quantization modes: pertensor[quantization mode](../../../docs/en/context/quant_mode_introduction.md), perchannel[quantization mode](../../../docs/en/context/quant_mode_introduction.md), pergroup[quantization mode](../../../docs/en/context/quant_mode_introduction.md), and MX[quantization mode](../../../docs/en/context/quant_mode_introduction.md).
    - \`x\` does not support transpose. Therefore, [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) are not supported. For weight, non-contiguous tensors are supported only in the transpose scenario. For antiquantScale and antiquantOffsetOptional, non-contiguous tensors only support the transpose scenario, and their contiguity must be consistent with weight.
    - Shapes supported by `antiquantScale` for different quantization modes:
      - pertensor[quantization mode](../../../docs/en/context/quant_mode_introduction.md): (1,) or (1,1).
      - perchannel[quantization mode](../../../docs/en/context/quant_mode_introduction.md): the input shape is (1, n) or (n,).
      - pergroup[quantization mode](../../../docs/en/context/quant_mode_introduction.md): the input shape is (⌈k/group_size⌉, n), where group_size indicates the size of each group that k is divided into.
      - MX[quantization mode](../../../docs/en/context/quant_mode_introduction.md): the input shape is (⌈k/group_size⌉, n), where group_size indicates the size of each group that k is divided into. Only 32 is supported.
    - `quantScaleOptional` and `quantOffsetOptional` are reserved parameters and not yet in use. Always pass a null pointer.
    - The \`innerPrecise\` parameter does not take effect.

    <a id="a16w8-scenario-constraints"></a>

    <details>
    <summary>A16W8 Scenario Constraints</summary>

    - **Input and Output Data Type Combination Requirements**

    | x | Weight | Weight Format | antiquantScale | antiquantOffsetOptional | quantScaleOptional | quantOffsetOptional | biasOptional | antiquantGroupSize | y | Scenario Description |
    | ----     | ------------------| --------------| -------------- | ------------------------| ------------------ | ------------------- | ------------ | ------------------ | ---- | ------- |
    | FLOAT16/BFLOAT16 | INT8 | ND | Same as x | Same as x/null | null | null | Same as x/FLOAT (only when x is BFLOAT16)/null | pergroup: [32, k-1] and a multiple of 32<br>Other: 0 | Same as x | T & C & G quantization |
    | FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E4M3FN | ND | Same as x | null | null | null | Same as x/null | 0 | Same as x | C quantization |

    - Value relationships among x, weight, antiquantScale, antiquantOffsetOptional, biasOptional, and antiquantGroupSize:

    | Scenario Description | X Data Type | Weight Data Type | X Shape | Weight Shape | antiquantScale Shape | antiquantOffsetOptional Shape | biasOptional Shape | antiquantGroupSize |
    | ------- | -------- | ------------- | ------- | ------------ | -------------------- | ----------------------------- | ------------------ | ------------------ |
    | T quantization | FLOAT16/BFLOAT16 | INT8 | (m, k) | (k, n) | (1,) or (1,1) | (1,) or (1,1)/null | (n,) or (1,n)/null | 0 |
    | C quantization | FLOAT16/BFLOAT16 | INT8 | (m, k) | (k, n) | (1,n) or (n,) | (1,n) or (n,)/null | (n,) or (1,n)/null | 0 |
    | G quantization | FLOAT16/BFLOAT16 | INT8 | (m, k) | (k, n) | (⌈k/group_size⌉, n) | (⌈k/group_size⌉, n)/null | (n,) or (1,n)/null | [32, k-1] and a multiple of 32 |
    | C quantization | FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E4M3FN | (m, k) | (k, n) | (1,n) or (n,) | null | (n,) or (1,n)/null | 0 |

    </details>

    <a id="a16w4-scenario-constraints"></a>

    <details>
    <summary>A16W4 Scenario Constraints</summary>

    - **Input and Output Data Type Combination Requirements**

    | x | Weight | Weight Format | antiquantScale | antiquantOffsetOptional | quantScaleOptional | quantOffsetOptional | biasOptional | antiquantGroupSize | y | Scenario Description |
    | ----     | ------------------| --------------| -------------- | ------------------------| ------------------ | ------------------- | ------------ | ------------------ | ---- | ------- |
    | FLOAT16/BFLOAT16 | INT4/INT32 | ND | Same as x | Same as x/null | null | null | Same as x/FLOAT (only when x is BFLOAT16)/null | 0 | Same as x | T quantization |
    | FLOAT16/BFLOAT16 | INT4/INT32 | ND | Same as x | Same as x/null | null | null | Same as x/FLOAT (only when x is BFLOAT16)/null | pergroup: [32, k-1] and a multiple of 32<br>Other: 0 | Same as x | C & G quantization |
    | FLOAT16/BFLOAT16 | FLOAT4_E2M1/FLOAT | ND | FLOAT8_E8M0 | null | null | null | Same as x/null | 32 | Same as x | MX quantization |

    - Value relationships among x, weight, antiquantScale, antiquantOffsetOptional, biasOptional, and antiquantGroupSize:

    | Scenario Description | X Data Type | Weight Data Type | X Shape | Weight Shape | antiquantScale Shape | antiquantOffsetOptional Shape | biasOptional Shape | antiquantGroupSize |
    | ------- | -------- | ------------- | ------- | ------------ | -------------------- | ----------------------------- | ------------------ | ------------------ |
    | T quantization | FLOAT16/BFLOAT16 | INT4/INT32 | (m, k) | (k, n) | (1,) or (1,1) | (1,) or (1,1)/null | (n,) or (1,n)/null | 0 |
    | C quantization | FLOAT16/BFLOAT16 | INT4/INT32 | (m, k) | (k, n) | (1,n) or (n,) | (1,n) or (n,)/null | (n,) or (1,n)/null | 0 |
    | G quantization | FLOAT16/BFLOAT16 | INT4/INT32 | (m, k) | (k, n) | (⌈k/group_size⌉, n) | (⌈k/group_size⌉, n)/null | (n,) or (1,n)/null | [32, k-1] and a multiple of 32 |
    | MX quantization | FLOAT16/BFLOAT16 | FLOAT4_E2M1/FLOAT | (m, k) | (k, n) | (⌈k/32⌉, n) | null | (n,) or (1,n)/null | 32 |

    - **Constraints**
      In addition to [common constraints](#constraints), the other constraints are as follows:
      - If the data type of `weight` is INT4 or FLOAT4_E2M1, the last dimension of weight must be 2-aligned. If the data type of `weight` is INT32 or FLOAT, the last dimension of weight must be 8-aligned.
      - If the `weight` data type is INT32/FLOAT, the conversion of INT32/FLOAT to tightly packed INT4/FLOAT4_E2M1 must be completed through the `aclnnConvertWeightToINT4Pack` API. [For details, see the sample](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md).

  <a id="ascend_950pr_ascend950dt_performance-optimization-suggestions"></a>

  - **Performance Optimization Suggestions**

    - pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md): When the [data format](../../../docs/en/context/data_format.md) is ND, it is recommended to use the transposed `weight` input;
    - perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md): When the [data format](../../../docs/en/context/data_format.md) is ND, it is recommended to use the transposed `weight` input;
    - pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md) and MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md): It is recommended to use the non-transposed `weight` input.

    </details>

</details>
<!-- end id10 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950,A3,910b" id11 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
<!-- end id11 -->

A16W8 example.

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v3.h"

  #define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

  #define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate: resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return 0;
  }

  int main() {
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Customize the construction based on the API interface.
    std::vector<int64_t> xShape = {16, 32};
    std::vector<int64_t> weightShape = {32, 16};
    std::vector<int64_t> yShape = {16, 16};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* y = nullptr;
    int32_t innerPrecise = 1;
    std::vector<float> xHostData(512, 1);
    std::vector<int8_t> weightHostData(512, 1);
    std::vector<float> yHostData(256, 0);

    std::vector<int64_t> antiquantScaleShape = {16};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(16, 1);


    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create other aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create antiquantScale aclTensor.
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(antiquantScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create xFp16 aclTensor.
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xFp16TensorPtr(xFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xFp16DeviceAddrPtr(xFp16DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create antiquantScale aclTensor.
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleFp16TensorPtr(antiquantScaleFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleFp16DeviceAddrPtr(antiquantScaleFp16DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create yFp16 aclTensor.
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // Call cast to generate the FP16 input.
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    std::unique_ptr<void, aclError (*)(void*)> workspaceCast0Ptr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceCast0Ptr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    std::unique_ptr<void, aclError (*)(void*)> workspaceCast1Ptr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceCast1Ptr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnWeightQuantBatchMatmulV3.
    ret = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(xFp16, weight, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, innerPrecise, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    std::unique_ptr<void, aclError (*)(void*)> workspaceMatmulPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceMatmulPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnWeightQuantBatchMatmulV3.
    ret = aclnnWeightQuantBatchMatmulV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3 failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Convert the output to FP32.
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    std::unique_ptr<void, aclError (*)(void*)> workspaceCast2Ptr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceCast2Ptr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, and copy the result on the device memory to the host memory. Modify the code based on the specific API definition.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. aclTensor and device resources are automatically released by unique_ptr.

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>:
<!-- end id12 -->

A16MxFp4 example.

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v3.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  template <typename T1>
  inline T1 CeilDiv(T1 a, T1 b)
  {
      return b == 0 ? a : (a + b - 1) / b;
  };
  template <typename T1>
  inline T1 CeilAlign(T1 a, T1 b)
  {
      return (a + b - 1) / b * b;
  };

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
      const aclTensor* weight, const aclTensor* weightInt4Pack, uint64_t* workspaceSize, aclOpExecutor** executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(
      void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // Boilerplate, resource initialization.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  template <typename T>
  int CreateAclTensorB4(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor, aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      if (format == aclFormat::ACL_FORMAT_ND) {
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
              shape.size(), *deviceAddr);
      } else {
          std::vector<int64_t> nzShape;
          if (dataType == aclDataType::ACL_INT4 || dataType == aclDataType::ACL_FLOAT4_E2M1) {
              nzShape = {CeilDiv(shape[1], (int64_t)16), CeilDiv(shape[0], (int64_t)16), 16, 16};
          } else {
              nzShape = {CeilDiv(shape[1], (int64_t)2), CeilDiv(shape[0], (int64_t)16), 16, 2};
          }
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(),
              nzShape.size(), *deviceAddr);
      }

      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
  {
      int64_t m = resultShape[0];
      int64_t n = resultShape[1];
      for (size_t i = 0; i < m; i++) {
          printf(i == 0 ? "[[" : " [");
          for (size_t j = 0; j < n; j++) {
              printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
              if (j == 2 && j + 3 < n) {
                  printf("..., ");
                  j = n - 4;
              }
          }
          printf(i < m - 1 ? "],\n" : "]]\n");
          if (i == 2 && i + 3 < m) {
              printf(" ... \n");
              i = m - 4;
          }
      }
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnWeightQuantBatchMatmulV3Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      aclDataType weightPackedDtype = aclDataType::ACL_FLOAT; // Optional: ACL_FLOAT data type
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs, which need to be customized based on the API.
      int64_t m = 16;
      int64_t k = 64;
      int64_t n = 64;
      int64_t groupSize = 32;
      int64_t weightDim0 = k;
      int64_t weightDim1 = n;
      bool isWeightTransposed = false;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {k / groupSize, n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* weightB4PackDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* y = nullptr;
      aclTensor* antiquantScale = nullptr;
      std::vector<int64_t> weightPackedShape;
      weightPackedShape = {weightDim0, weightDim1 / 8};
      std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0b0011110000000000); // 1.0 in fp16
      xHostData[0] = 0;                                                  // 0 in fp16, used to verify whether the result meets the requirement
      std::vector<float> weightHostData(GetShapeSize(weightShape), 1.0); // 1.0 in fp32, converted to 1.0 in fp4_e2m1 after int4pack
      std::vector<float> yHostData(GetShapeSize(yShape), 0);

      std::vector<uint8_t> antiquantScaleHostData(GetShapeSize(antiquantScaleShape), 0b01111111); // 1.0 in fp8_e8m0

      // Create x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create other aclTensor.
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create y aclTensor.
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create antiquantScale aclTensor.
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the yFp16 aclTensor
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Change it to the specific API name.
      aclFormat weightFormat = aclFormat::ACL_FORMAT_ND; // Optional: ACL_FORMAT_FRACTAL_NZ
      aclTensor* weightPacked = nullptr;

      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); // Each B8 data stores 2 B4 data items, so divide by 2 here.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
          weightB4PackHostData.resize(CeilAlign(weightDim1 / 2, (int64_t)8) * CeilAlign(weightDim0, (int64_t)16), 0);
      }
      // Create the weightInt4Pack aclTensor
      ret = CreateAclTensorB4(
          weightB4PackHostData, weightPackedShape, &weightB4PackDeviceAddr, weightPackedDtype, &weightPacked,
          weightFormat);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPackedTensorPtr(weightPacked, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightPackedDeviceAddrPtr(weightB4PackDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Convert weight from int32 to int4pack
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      void* workspacePackAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspacePackAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspacePackAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspacePackAddrPtr.reset(workspacePackAddr);
      }
      ret = aclnnConvertWeightToINT4Pack(workspacePackAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnWeightQuantBatchMatmulV3
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
          x, weightPacked, antiquantScale, nullptr, nullptr, nullptr, nullptr, groupSize, 0, yFp16, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnWeightQuantBatchMatmulV3
      ret = aclnnWeightQuantBatchMatmulV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronously wait for the task execution to complete
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Convert the output to FP32.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, and copy the result from the device memory to the host memory. Modify the code based on the specific API definition.
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      PrintMat(resultData, yShape);
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnWeightQuantBatchMatmulV3Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3Test failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
