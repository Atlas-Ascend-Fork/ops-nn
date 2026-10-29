# aclnnWeightQuantBatchMatmulV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:32:53.538Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**: Performs matrix multiplication with the input in a Fake Quantization scenario, and supports quantization computation on the output.
- **Formula**:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  In the formula, $weight$ is the input in the pseudo-quantization scenario, and its dequantization formula $ANTIQUANT(weight)$ is

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  - When no quantization operation is required on the output, the formula is:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  - When the output needs to be quantized, the quantization formula is:

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnWeightQuantBatchMatmulV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that includes the operator computation process, and then call the "aclnnWeightQuantBatchMatmulV2" API to perform the computation.

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *antiquantScale,
  const aclTensor *antiquantOffsetOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *biasOptional,
  int              antiquantGroupSize,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnWeightQuantBatchMatmulV2GetWorkspaceSize

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
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th style="white-space: nowrap">Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Left input matrix of the matrix multiplication, the input <code>x</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2 dimensions, supported shape (m, k)</td>
        <td>Supported only in the transpose-only field</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Right input matrix of the matrix multiplication, the input <code>weight</code> in the formula.</td>
        <td>-</td>
        <td>INT8, INT4, FLOAT8_E4M3FN<sup>2</sup>, HIFLOAT8<sup>2</sup>, INT32, FLOAT<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
        <td>ND, FRACTAL_NZ</td>
        <td>2 dimensions, supported shape (k, n)</td>
        <td>Supported in the transpose-only use case.</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Dequantization scale parameter for implementing the input dequantization computation, i.e., the input <code>antiquantScale</code> in the dequantization formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E8M0<sup>2</sup>, UINT64<sup>1</sup>, INT64<sup>1</sup></td>
        <td>ND</td>
        <td>1-2 dimensions</td>
        <td>Only supported in the transpose-only field</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>Optional input</td>
        <td>Dequantization offset parameter that implements input dequantization computation, corresponding to <code>antiquantOffset</code> in the dequantization formula.</td>
        <td>Null pointer when not needed.</td>
        <td>FLOAT16, BFLOAT16, INT32<sup>1</sup></td>
        <td>ND</td>
        <td>Must be consistent with <code>antiquantScale</code>.</td>
        <td>Supported only in the transpose-only field</td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>Optional input</td>
        <td>Quantization parameters for output quantization computation.</td>
        <td>Obtained by converting the data of <code>quantScale</code> and <code>quantOffset</code> in the quantization formula through the <code>aclnnTransQuantParam</code> API. Null pointer when not needed.</td>
        <td>UINT64<sup>1</sup></td>
        <td>ND</td>
        <td>1-2 dimensions</td>
        <td>Not supported</td>
      </tr>
      <tr>
        <td>quantOffsetOptional</td>
        <td>Optional input</td>
        <td>Quantization offset parameter for output quantization computation, <code>quantOffset</code> in the quantization formula.</td>
        <td>Null pointer when not needed.</td>
        <td>FLOAT<sup>1</sup></td>
        <td>ND</td>
        <td>Must be consistent with <code>quantScaleOptional</code>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Optional input</td>
        <td>Bias input, <code>bias</code> in the formula.</td>
        <td> Null pointer when not needed.</td>
        <td>FLOAT16, FLOAT, BFLOAT16<sup>2</sup></td>
        <td>ND</td>
        <td>1-2 dimensions</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantGroupSize</td>
        <td>Input</td>
        <td>Specifies the groupSize input for dequantization computation on input <code>weight</code> in pseudo-quantization pergroup and MX <a href="../../../docs/en/context/quant_mode_introduction.md" target="_blank">quantization mode</a>, describing the size of the data to be dequantized for a group of dequantization parameters in the Reduce direction.</td>
        <td>Pass 0 when the pseudo-quantization algorithm is neither pergroup nor MX <a href="../../../docs/en/context/quant_mode_introduction.md" target="_blank">quantization mode</a>.<br>When the pseudo-quantization algorithm is pergroup <a href="../../../docs/en/context/quant_mode_introduction.md" target="_blank">quantization mode</a>, pass a value in the range [32, k-1], which must be a multiple of 32.<br>Only 32 is supported in MX <a href="../../../docs/en/context/quant_mode_introduction.md" target="_blank">quantization mode</a>.</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td><code>y</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8<sup>1</sup></td>
        <td>ND</td>
        <td>2D</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be applied on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which includes the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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
  <!-- npu="310p" id9 -->
  - <term>Atlas inference products</term>:

    - The superscript "3" in the data type column of the table above indicates a data type not supported by this series.

  <!-- end id9 -->

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
        <th>Return Values</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>A required parameter is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="13">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="13">161002</td>
        <td>The shape dimensions of x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, or y do not meet the requirements.</td>
      </tr>
      <tr>
        <td>The data types of x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, or y are not within the supported range.</td>
      </tr>
      <tr>
        <td>The reduce dimensions (k) of x and weight are not equal.</td>
      </tr>
      <tr>
        <td>When antiquantOffsetOptional has an input, its shape is inconsistent with antiquantScale.</td>
      </tr>
      <tr>
        <td>When quantOffsetOptional has an input, its shape is inconsistent with quantScale.</td>
      </tr>
      <tr>
        <td>The shape of biasOptional does not meet requirements.</td>
      </tr>
      <tr>
        <td>The value of antiquantGroupSize does not meet requirements.</td>
      </tr>
      <tr>
        <td>When quantOffsetOptional is present, quantScaleOptional is a null pointer.</td>
      </tr>
      <tr>
        <td>The input k and n values are not within the range [1, 65535].</td>
      </tr>
      <tr>
        <td>When x is non-transposed, m is not within the range [1, 2^31-1]; when x is transposed, m is not within the range [1, 65535].</td>
      </tr>
      <tr>
        <td>Empty tensors are not supported.</td>
      </tr>
      <tr>
        <td>The contiguity of input x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, and y does not meet requirements.</td>
      </tr>
      <tr>
        <td>When x is bfloat16 and weight is float4_e2m1 or float32, the bias data type only supports bfloat16.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>The product model is not supported.</td>
      </tr>
    </tbody>
  </table>

## aclnnWeightQuantBatchMatmulV2

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
      <td>The workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size applied on the Device side, obtained by the first-phase API aclnnWeightQuantBatchMatmulV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which includes the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<a id="a2_a3_series_products"></a>

<!-- npu="A3,910b" id10 -->
<details>
<summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term></summary>

  - **Deterministic computation description**: Defaults to a non-deterministic implementation. Supports enabling deterministic computation through aclrtCtxSetSysParamOpt.

  - `x` (aclTensor *, computation input): If the matrix is non-transposed, m is in the range [1, 2^31-1]; if transposed, m is in the range [1, 65535].
  - `weight` (aclTensor *, computation input): Supports two dimensions. The Reduce dimension k must be equal in size to the Reduce dimension k of `x`. The supported data types are INT8, INT4, and INT32. When the [data format](../../../docs/en/context/data_format.md) of `weight` is FRACTAL_NZ and the data type is INT4 or INT32, or when the [data format](../../../docs/en/context/data_format.md) of `weight` is ND and the data type is INT32, these data types are supported only in the INT4Pack use case. In this case, use the `aclnnConvertWeightToINT4Pack` API to convert weights from INT32 to INT4Pack and from ND to FRACTAL_NZ. For details, see [the sample](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md). If the data type is INT4, the inner axis of `weight` must be even. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) only support transpose. The supported shape is (k, n), where k represents the size of the 1st dimension of the matrix and n represents the size of the 2nd dimension of the matrix.
    For different pseudo-quantization algorithm modes, the [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ of `weight` is supported only in the following use cases:
    - perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md):
      - The data type of `weight` is INT8, and the data type of y is non-INT8.
      - The data type of `weight` is INT4/INT32, `weight` is transposed, and the data type of y is non-INT8.
    - pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The data type of `weight` is INT4/INT32, `weight` is non-transposed, `x` is non-transposed, antiquantGroupSize is 64 or 128, k is aligned to antiquantGroupSize, n is aligned to 64, and the data type of y is not INT8.
  - `antiquantScale` (aclTensor *, computation input): Supports FLOAT16, BFLOAT16, UINT64, and INT64 data types (when the data type is FLOAT16 or BFLOAT16, the data type must remain consistent with that of input `x`; when the data type is UINT64 or INT64, `x` supports only FLOAT16 and is not transposed, `weight` supports only INT8 with ND transpose, only the perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md) is supported, quantScaleOptional and quantOffsetOptional must be null pointers, m supports only [1, 96], k and n must be aligned to 64, the aclnnCast API must be called first to convert FLOAT16 to FLOAT32. For details, see [Cast](https://gitcode.com/cann/ops-math/blob/master/math/cast/docs/aclnnCast.md). Then, the aclnnTransQuantParamV2 API must be called to convert FLOAT32 to UINT64. For details, see [TransQuantParamV2](../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2.md)). [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) supports only the transpose use case.
    For different Fake Quantization algorithm modes, the supported shapes of `antiquantScale` are as follows:
    - pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape is (1,) or (1, 1).
    - perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape is (1, n) or (n,).
    - pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape is (⌈k/group_size⌉, n), where group_size indicates the size of each group into which k is divided.
  - `antiquantOffsetOptional` (aclTensor *, computation input): Supports FLOAT16, BFLOAT16, and INT32 data types. When the data type is FLOAT16 or BFLOAT16, the data type must remain consistent with the data type of input `x`; when the data type is INT32, the data range is limited to [-128, 127], x supports only FLOAT16, weight supports only INT8, and `antiquantScale` supports only UINT64/INT64. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) supports only the transpose use case.
  - `quantScaleOptional` (aclTensor *, computation input): Supports the UINT64 data type, and the [data format](../../../docs/en/context/data_format.md) supports ND. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is not supported. Optional input, which is a null pointer when not needed; for different pseudo-quantization algorithm modes, the supported shapes are as follows:
    - pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape is (1,) or (1, 1).
    - perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape is (1, n) or (n,).
  - `quantOffsetOptional` (aclTensor *, computation input): Supports the FLOAT data type and the ND [data format](../../../docs/en/context/data_format.md). Optional input; a null pointer when not needed. If present, the shape must be consistent with `quantScaleOptional`. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
  - `biasOptional` (aclTensor *, computation input): Supports 1D or 2D, with shape (n,) or (1, n). Supports the FLOAT16 and FLOAT data types. When the data type of `x` is BFLOAT16, this parameter must be FLOAT; when the data type of `x` is FLOAT16, this parameter must be FLOAT16.
  - `antiquantGroupSize` (int, computation input): GroupSize input for performing dequantization on the input `weight` in the pseudo-quantization pergroup and MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md). It describes the size, in the Reduce direction, of the data to be dequantized corresponding to a group of dequantization parameters. Set this parameter to 0 when the pseudo-quantization algorithm is not the pergroup or MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md); when the pseudo-quantization algorithm is the pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md), the value must be in the range [32, k-1] and be a multiple of 32; in the MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md), only 32 is supported.
  - `y` (aclTensor *, computation output): Supports 2D, with shape (m, n). Supports the FLOAT16, BFLOAT16, and INT8 data types. When `quantScaleOptional` is present, the data type is INT8; when `quantScaleOptional` is absent, the data type is FLOAT16 or BFLOAT16 and is consistent with the data type of input `x`.

  - Performance optimization suggestions:
    - pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md): When the [data format](../../../docs/en/context/data_format.md) is ND, it is recommended to use the transposed `weight` input; when the [data format](../../../docs/en/context/data_format.md) is FRACTAL_NZ, it is recommended to use the non-transposed `weight` input.
    - pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md): It is recommended to use the non-transposed weight input.
    - perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md): When the [data format](../../../docs/en/context/data_format.md) is ND, it is recommended to use the transposed `weight` input; when the [data format](../../../docs/en/context/data_format.md) is FRACTAL_NZ, it is recommended to use the non-transposed `weight` input. When m is in the range [65, 96], it is recommended to use antiquantScale with the UINT64 or INT64 data type.

</details>
<!-- end id10 -->

<a id="Atlas inference products"></a>

<!-- npu="310p" id11 -->
<details>
<summary><term>Atlas inference products</term></summary>

  - **Deterministic computation description**: Defaults to a non-deterministic implementation. Supports enabling deterministic computation through aclrtCtxSetSysParamOpt.

  - `x` (aclTensor *, computation input): Supports the FLOAT16 data type. The shape supports 2-6 dimensions, and the input shape must be (batch, m, k), where batch represents the batch size of the matrix, ~~supporting 0-4 dimensions~~, m represents the size of the 1st dimension of a single batch matrix, k represents the size of the 2nd dimension of a single batch matrix, and the batch dimension must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with the batch dimension of `weight`. When the pseudo-quantization algorithm mode is the pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md), `m*k` cannot exceed 512000000.
  - `weight` (aclTensor *, computation input): Supports 2-6 dimensions. The batch dimension must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with the batch dimension of `x`. The weight data type supports INT8. The details are as follows:
    - When the [data format](../../../docs/en/context/data_format.md) is ND, the input shape must be (batch, k, n), where batch represents the batch size of matrices, supporting 0-4 dimensions, k represents the size of the 1st dimension of a single batch matrix, and n represents the size of the 2nd dimension of a single batch matrix.
    - When the [data format](../../../docs/en/context/data_format.md) is FRACTAL_NZ:
      - The input shape must be (batch, n, k), where batch indicates the batch size of the matrices, supporting 0 to 4 dimensions, k indicates the size of the first dimension of a single batch matrix, and n indicates the size of the second dimension of a single batch matrix.
      - Used together with aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight to convert the input format from ND to FRACTAL_NZ.
  - `antiquantScale` (aclTensor *, computation input): The data type supports FLOAT16 and must remain consistent with the input `x`.
    For different Fake Quantization algorithm modes, the supported shapes of `antiquantScale` are as follows:
    - pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape is (1,) or (1, 1).
    - perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape is (n, 1) or (n,). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape depends on the data format of `weight`, as follows:
      - When the data format of `weight` is ND, the input shape is (⌈k/group_size⌉, n), where group_size indicates the size of each group into which k is to be grouped.
      - When the data format of `weight` is FRACTAL_NZ, the input shape is (n, ⌈k/group_size⌉), where group_size indicates the size of each group into which k is to be grouped.
  - `antiquantOffsetOptional` (aclTensor *, computation input): The data type supports FLOAT16 and must remain consistent with the input `x`.
  - `quantScaleOptional` (aclTensor *, computation input): Reserved parameter, not used yet. Always pass in a null pointer.
  - `quantOffsetOptional` (aclTensor *, computation input): Reserved parameter, not used yet. Always pass in a null pointer.
  - `biasOptional` (aclTensor *, computation input): Supports the FLOAT16 data type. Supports 1-6 dimensions. With batch, the input shape must be (batch, 1, n), and the batch must remain consistent with the batch after broadcasting the batch dimensions of x and weight. Without batch, the input shape must be (n,) or (1, n).
  - `antiquantGroupSize` (int, computation input): Supports the FLOAT16 data type. Supports 2-6 dimensions, with the shape (batch, m, n). batch can be omitted, and broadcasting of the batch dimensions of x and weight is supported. The output batch is the same as the batch after broadcasting, m is the same as m of x, and n is the same as n of weight.
  - `y` (aclTensor *, computation output):

</details>
<!-- end id11 -->

<a id="ascend_950pr_ascend950dt"></a>

<!-- npu="950" id12 -->
<details>
<summary><term>Ascend 950PR&950DT products</term></summary>

  - **Deterministic computation description**: Defaults to a deterministic implementation.

  <a id="general-constraints"></a>

  - **General Constraints**
    - The m, k, and n dimensions of the `x` and `weight` matrices are in the range [1, 2^31-1]. The reduce dimension k of `weight` must be equal to the reduce dimension k of `x`.
    - Supported quantization modes: pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md), perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md), pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md), and MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md).
    - `x` does not support transpose, and therefore [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) are not supported. weight supports non-contiguous tensors only in the transpose-only field; antiquantScale and antiquantOffsetOptional non-contiguous tensors only support the transpose use case, and their contiguity must remain consistent with weight.
    - Shapes supported by `antiquantScale` for different quantization modes:
      - pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md): (1,) or (1,1).
      - perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md): The input shape is (1, n) or (n,).
      - pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md): input shape is (⌈k/group_size⌉, n), where group_size indicates the size of each group into which k is grouped.
      - MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md): input shape is (⌈k/group_size⌉, n), where group_size indicates the size of each group into which k is grouped; only 32 is supported.
    - `quantScaleOptional` and `quantOffsetOptional` are reserved parameters and not yet in use. Always pass a null pointer.

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
      - When the `weight` data type is INT32/FLOAT, use the `aclnnConvertWeightToINT4Pack` API to complete the conversion from INT32/FLOAT to tightly packed INT4/FLOAT4_E2M1. For details, see [Sample](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md).

  <a id="ascend_950pr_ascend950dt_performance-optimization-suggestions"></a>

  - **Performance Optimization Suggestions**

    - pertensor [quantization mode](../../../docs/en/context/quant_mode_introduction.md): It is recommended to use the transposed `weight` input.
    - perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md): It is recommended to use the transposed `weight` input.
    - pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md) and MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md): It is recommended to use the non-transposed `weight` input.

    </details>

</details>
<!-- end id12 -->

## Example

  The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950,A3,910b" id13 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
<!-- end id13 -->

A16W8 example:

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // Boilerplate: resource initialization
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
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

  int AclnnWeightQuantBatchMatmulV2Test(int32_t deviceId, aclrtStream stream)
  {
      int64_t m = 16;
      int64_t k = 32;
      int64_t n = 16;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* antiquantScale = nullptr;
      aclTensor* y = nullptr;
      // Fill 1.0 for FP16; 1.0 of BF16 is 0b0011111110000000
      std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0b0011110000000000); // 1.0 of fp16
      std::vector<int8_t> weightHostData(GetShapeSize(weightShape), 1);
      std::vector<uint16_t> antiquantScaleHostData(GetShapeSize(antiquantScaleShape), 0b0011110000000000);
      std::vector<float> yHostData(GetShapeSize(yShape), 0);

      // Create the x aclTensor, which can be of type ACL_FLOAT16/ACL_BFLOAT16
      auto ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the other aclTensor, which can be of type ACL_INT8/ACL_FLOAT8_E8M0/ACL_HIFLOAT8
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the y aclTensor to convert the output back to FP32
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the antiquantScale aclTensor
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the yFp16 aclTensor, the actual computation output, whose type is consistent with x
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16deviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Replace it with the specific API name
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
      ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
          x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
      ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait until task execution completes.
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
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to host memory. Modify this based on the specific API definition.
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
      // 1. (boilerplate) Initialize device/stream by referring to the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API definition.
      ret = AclnnWeightQuantBatchMatmulV2Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulV2Test failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- npu="950" id14 -->
- <term>Ascend 950PR&950DT products</term>:
<!-- end id14 -->

A16MxFp4 example: the `aclnnConvertWeightToINT4Pack` API must be called to assist the call:

  ``` cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_npu_format_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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
      // Boilerplate, resource initialization
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
      // Call aclrtMemcpy to copy host data to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
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
      // Call aclrtMemcpy to copy host data to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
          shape.size(), *deviceAddr);

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

  int AclnnWeightQuantBatchMatmulV2Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      aclDataType weightPackedDtype = aclDataType::ACL_FLOAT; // Optional: ACL_FLOAT type
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs. Customize the construction according to the API.
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
      xHostData[0] = 0;                                                  // 0 in fp16, used to verify whether the result meets requirements
      std::vector<float> weightHostData(GetShapeSize(weightShape), 1.0); // 1.0 in fp32, converted to 1.0 in fp4_e2m1 after int4pack
      std::vector<float> yHostData(GetShapeSize(yShape), 0);

      std::vector<uint8_t> antiquantScaleHostData(GetShapeSize(antiquantScaleShape), 0b01111111); // 1.0 in fp8_e8m0

      // Create the x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the other aclTensor.
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the y aclTensor.
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the antiquantScale aclTensor.
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the yFp16 aclTensor.
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Change it to the specific API name.
      aclFormat weightFormat = aclFormat::ACL_FORMAT_ND;
      aclTensor* weightPacked = nullptr;

      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); // One B8 data stores 2 B4 data items, so divide by 2 here.
      // Create the weightInt4Pack aclTensor.
      ret = CreateAclTensorB4(
          weightB4PackHostData, weightPackedShape, &weightB4PackDeviceAddr, weightPackedDtype, &weightPacked,
          weightFormat);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPackedTensorPtr(weightPacked, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightPackedDeviceAddrPtr(weightB4PackDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Convert weight from int32 to int4pack.
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

      // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
          x, weightPacked, antiquantScale, nullptr, nullptr, nullptr, nullptr, groupSize, yFp16, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
      ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait for task execution to complete.
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
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, copy the result from device memory to host memory, and modify it based on the definition of the specific API.
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
      // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnWeightQuantBatchMatmulV2Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulV2Test failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
