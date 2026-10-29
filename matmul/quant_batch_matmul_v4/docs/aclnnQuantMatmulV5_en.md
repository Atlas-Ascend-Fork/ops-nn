# aclnnQuantMatmulV5

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:44:51.442Z -->

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

- Description: Performs quantized matrix multiplication computation.

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    Compatible with the API functions of aclnnQuantMatmulV3 and aclnnQuantMatmulV4. Performs quantized matrix multiplication computation. The minimum supported input dimension is 1, and the maximum supported input dimension is 2. A similar API is aclnnMm (matrix multiplication that only supports 2D tensors as input).
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:

    This API is compatible with the Description of aclnnQuantMatmulV3 and aclnnQuantMatmulV4, and adds support for new features such as the G-B, B-B, T-CG, and MX [quantization modes](../../../docs/en/context/quant_mode_introduction.md). The x1 and x2 inputs additionally support data types such as FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1. This API completes the quantized matrix multiplication computation, supporting a minimum input dimension of 2 and a maximum input dimension of 6. Similar APIs include aclnnMm (matrix multiplication that supports only 2-dimensional tensors as input) and aclnnBatchMatMul (matrix multiplication that supports only 3-dimensional tensors, where the first dimension is the batch dimension).

  <!-- end id8 -->

- Formula:

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    Supports the K-C && K-T, T-C && T-T, G-B, and K-G [quantization modes](../../../docs/en/context/quant_mode_introduction.md). For the input and output data type combinations corresponding to different quantization modes, see [Constraints](#constraints).

    <details>

    <summary><strong>K-G quantization mode</strong></summary>

      - x1 is INT8, x2 is INT32, x1Scale is FLOAT32, x2Scale is UINT64/INT64, and yOffset is FLOAT32:

        $$
        out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
        $$

      - x1 and x2 are INT4, x1Scale and x2Scale are FLOAT32, x2Offset is FLOAT16, and out is FLOAT16/BFLOAT16 (pertoken-pergroup asymmetric quantization):

        $$
        out = x1Scale * x2Scale * (x1 @ x2 - x1 @ x2Offset)
        $$

    </details>

    <details>

    <summary><strong>K-C && K-T quantization mode</strong></summary>

      - x1Scale, no bias:

        $$
        out = x1@x2 * x2Scale * x1Scale
        $$

      - x1Scale and bias INT32 (no offset in this use case):

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

      - x1Scale and bias BFLOAT16/FLOAT16/FLOAT32 (no offset in this use case):

        $$
        out = x1@x2 * x2Scale * x1Scale + bias
        $$

    </details>

    <details>

    <summary><strong>T-C && T-T quantization mode</strong></summary>

      - No x1Scale or bias:

        $$
        out = x1@x2 * x2Scale + x2Offset
        $$

      - bias INT32:

        $$
        out = (x1@x2 + bias) * x2Scale + x2Offset
        $$

      - bias BFLOAT16/FLOAT32 (no offset in this scenario):

        $$
        out = x1@x2 * x2Scale + bias
        $$

    </details>

    <details>

    <summary><strong>G-B quantization mode</strong></summary>

      - x1, x2 are INT8, x1Scale, x2Scale are FLOAT32, bias is FLOAT32, out is FLOAT16/BFLOAT16 (pergroup-perblock quantization):

        $$
        out = (x1 @ x2) * x1Scale * x2Scale + bias
        $$

    </details>

  <!-- end id9 -->
  <!-- npu="310p" id10 -->
  - <term>Atlas inference products</term>:

    K-C [quantization mode](../../../docs/en/context/quant_mode_introduction.md) is supported. For the input and output data type combinations of different quantization modes, see [Constraints](#constraints).

    <details>

    <summary><strong>K-C Quantization Mode</strong></summary>

      - x1Scale, no bias:

        $$
        out = x1@x2 * x2Scale * x1Scale
        $$

      - x1Scale, bias INT32 (no offset in this use case):

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

    </details>

  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>:

    Supports T-C && T-T, K-C && K-T, G-B, B-B, MX, T-CG, and K-G [quantization modes](../../../docs/en/context/quant_mode_introduction.md). For the input and output data type combination corresponding to different quantization modes, see [Constraints](#constraints).

    <details>

    <summary><strong>K-G quantization mode</strong></summary>

      - x1 and x2 are INT4, x1Scale and x2Scale are FLOAT32, x2Offset is FLOAT16, and out is FLOAT16/BFLOAT16 (pertoken-pergroup asymmetric quantization):

        $$
        out = x1Scale * x2Scale * (x1 @ x2 - x1 @ x2Offset)
        $$

    </details>

    <details>

    <summary><strong>T-C && T-T quantization mode</strong></summary>

      - x1 and x2 are INT8, no x1Scale, x2Scale is INT64/UINT64, optional parameter x2Offset is FLOAT32, and optional parameter bias is INT32:

        $$
        out = (x1@x2 + bias) * x2Scale + x2Offset
        $$

      - x1 and x2 are INT8, without x1Scale, x2Scale is INT64/UINT64, and the optional parameter bias is INT32;
      Or x1 and x2 are of data type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1Scale is absent, x2Scale is of data type INT64/UINT64, and the optional parameter bias is of data type FLOAT32;
      Or x1 and x2 are of data type INT4, x1Scale is absent, x2Scale is of data type INT64/UINT64, and the optional parameter bias is of data type INT32:

        $$
        out = (x1@x2 + bias) * x2Scale
        $$

      - x1 and x2 are INT8, without x1Scale, x2Scale is BFLOAT16/FLOAT32, and the optional parameter bias is BFLOAT16/FLOAT32:

        $$
        out = x1@x2 * x2Scale + bias
        $$

      - x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1Scale is FLOAT32, x2Scale is FLOAT32, and the optional parameter bias is FLOAT32:

        $$
        out = x1@x2 * x2Scale * x1Scale + bias
        $$

    </details>

    <details>

    <summary><strong>K-C && K-T quantization mode</strong></summary>

      - x1 and x2 are INT8, x1Scale is FLOAT32, x2Scale is BFLOAT16/FLOAT32, and the optional parameter bias is INT32;
      Or x1 and x2 are INT4, x1Scale is FLOAT32, x2Scale is BFLOAT16/FLOAT32, and the optional parameter bias is INT32:

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

      - x1 and x2 are INT8, x1Scale is FLOAT32, x2Scale is BFLOAT16/FLOAT32, and the optional parameter bias is BFLOAT16/FLOAT32;
      Or x1 and x2 are INT8, x1Scale is FLOAT32, x2Scale is FLOAT32, and the optional parameter bias is FLOAT16/FLOAT32;
      Or x1 and x2 are INT4, x1Scale is FLOAT32, x2Scale is BFLOAT16/FLOAT32, and the optional parameter bias is BFLOAT16/FLOAT32;
      Or x1 and x2 are INT4, x1Scale is FLOAT32, x2Scale is FLOAT32, and the optional parameter bias is FLOAT16/FLOAT32;
      Or x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1Scale is FLOAT32, x2Scale is FLOAT32, and the optional parameter bias is FLOAT32:

        $$
        out = x1@x2 * x2Scale * x1Scale + bias
        $$

    </details>

    <details>

    <summary><strong>G-B && B-B && MX quantization mode</strong></summary>

      $$
      out[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
      $$

      gsM, gsN, and gsK respectively represent groupSizeM, groupSizeN, and groupSizeK. x1Slice represents the vector of length groupSizeK in row m of x1, and x2Slice represents the vector of length groupSizeK in column n of x2. Slicing along the K axis starts from j*groupSizeK, where j ranges over [0, kLoops), kLoops = ceil(K / groupSizeK), and K is the length of the K axis. A last slice shorter than groupSizeK is supported. bias is included only in the MX quantization mode. For the G-B, B-B, and MX quantization modes, the value combinations of [groupSizeM, groupSizeN, groupSizeK] support only [1, 128, 128], [128, 128, 128], and [1, 1, 32], respectively.

    </details>

    <details>

    <summary><strong>T-CG quantization mode</strong></summary>

      $$
      out = (x1@(x2 * x2Scale)) * yScale
      $$

    </details>

  <!-- end id11 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnQuantMatmulV5GetWorkspaceSize" API to obtain the workspace size required for the calculation and the executor that includes the calculation flow of the operator. Then call the "aclnnQuantMatmulV5" API to perform the calculation.

```cpp
aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *x1Scale,
  const aclTensor *x2Scale,
  const aclTensor *yScale,
  const aclTensor *x1Offset,
  const aclTensor *x2Offset,
  const aclTensor *yOffset,
  const aclTensor *bias,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulV5(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulV5GetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 248px">
  <col style="width: 121px">
  <col style="width: 210px">
  <col style="width: 327px">
  <col style="width: 250px">
  <col style="width: 115px">
  <col style="width: 138px">
  <col style="width: 145px">
  </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x1(aclTensor*)</td>
        <td>Input</td>
        <td>Input x1 in the formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Supports <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensor</a> only when the last m and k axes are transposed. Non-contiguous tensors are not supported along other axes.</li>
          </ul>
        </td>
        <td>INT4, INT8, INT32<sup>1</sup>, FLOAT8_E4M3FN<sup>2</sup>, FLOAT8_E5M2<sup>2</sup>, HIFLOAT8<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x2(aclTensor*)</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td>          <ul>
            <li>Does not support empty tensors.</li>
            <li>Does not support empty tensors in NZ format.</li>
            <li>Supports shapes with 4-8 dimensions in NZ format.</li>
            <li>In ND format, supports non-contiguous tensors with the last two axes transposed. <a href="../../../docs/en/context/non_contiguous_tensor.md">Non-contiguous tensors</a> in other scenarios are not supported.</li>
          </ul>
        </td>
        <td>INT4, INT8, INT32<sup>1</sup>, FLOAT8_E4M3FN<sup>2</sup>, FLOAT8_E5M2<sup>2</sup>, HIFLOAT8<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
        <td>ND, NZ</td>
        <td>2-6 (ND)<br>4-8 (NZ)</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x1Scale(aclTensor*)</td>
        <td>Optional input</td>
        <td>The input x1Scale in the formula.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
          </ul>
          </td>
        <td>FLOAT32, FLOAT8_E8M0<sup>2</sup></td>
        <td>ND</td>
        <td>1-7</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x2Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Represents the quantization parameter, that is, the input x2Scale in the formula.</td>
        <td>          <ul>
            <li>Does not support empty tensors.</li>
          </ul>
        </td>
        <td>UINT64, INT64, FLOAT32, BFLOAT16, FLOAT8_E8M0<sup>2</sup>, FLOAT16<sup>2</sup></td>
        <td>ND</td>
        <td>1-7</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>yScale(aclTensor*)</td>
        <td>Optional input</td>
        <td>Dequantization scale parameter of output y.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass in nullptr.</li>
          </ul>
        </td>
        <td>UINT64<sup>2</sup>, INT64<sup>2</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>x1Offset(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input x1Offset in the formula.</td>
        <td>        <ul>
        <li>
        Reserved parameter. Not supported in the current version. nullptr needs to be passed in.
        </li>
        </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Offset(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input x2Offset in the formula.</td>
        <td>          <ul>
            <li>Does not support empty tensors.</li>
          </ul>
        </td>
        <td>FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOffset(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input yOffset in the formula.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bias(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input bias in the formula.</td>
        <td>          <ul>
            <li>Does not support empty tensors.</li>
          </ul>
        </td>
        <td>INT32, FLOAT32, BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>1-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX1(bool)</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x1 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2(bool)</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x2 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize(int64_t)</td>
        <td>Optional input</td>
        <td>Specifies the quantization group sizes along the m, n, and k dimensions.</td>
        <td>Composed of the three values groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits, and the three values together occupy the low 48 bits of groupSize of the int64_t type (the value of the high 16 bits in groupSize is invalid). See the Formula below the table.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out(aclTensor*)</td>
        <td>Output</td>
        <td>Output out in the formula.</td>
        <td>          <ul>
            <li>Does not support empty Tensor.</li>
          </ul>
        </td>
        <td>FLOAT16, INT8, BFLOAT16, INT32, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor(aclOpExecutor*)</td>
        <td>Output</td>
        <td>Returns the op executor, which includes calculation sub-computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

  - Note: Optional inputs refer to optional quantization parameters, which support passing nullptr.

  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>:

    - The superscript "1" in the data type column of the table above indicates a data type not supported by this series.
    - The input parameters x1 and x2 do not support the INT32 type.
    - x2 supports only the ND format. In the full quantization scenario, if the input parameter x1 is an empty tensor with m=0 or x2 is an empty tensor with n=0, the output is an empty tensor.

  <!-- end id12 -->
  <!-- npu="A3,910b" id13 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    - The superscript "2" in the data type column of the table above indicates a data type not supported by this series.

  <!-- end id13 -->
  - Formula: <a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
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
        <td>x1, x2, x2Scale, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>x1, x2, bias, x1Scale, x2Scale, x2Offset, or out is an empty tensor.</td>
      </tr>
      <tr>
        <td>The data type or data format of x1, x2, bias, x1Scale, x2Scale, x2Offset, or out is not within the supported range.</td>
      </tr>
      <tr>
        <td>The shape of x1, x2, bias, x1Scale, x2Scale, x2Offset, or out does not meet the verification conditions.</td>
      </tr>
      <tr>
        <td>The input groupSize does not meet the verification conditions, or when the input groupSize is 0, groupSize cannot be inferred from the shape relationship between x1, x2 and x1Scale, x2Scale.</td>
      </tr>
    </tbody></table>

## aclnnQuantMatmulV5

- **Parameters**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
    <td>Memory address of the workspace applied on the Device side.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnQuantMatmulV5GetWorkspaceSize.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>The op executor, which includes the calculation flow of the operator.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the Stream for task execution.</td>
  </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: Returned status code. See [aclnn return codes](../../../docs/en/context/aclnn_return_code.md) for details.

## Constraints

- Deterministic computation:
  - aclnnQuantMatmulV5 defaults to a deterministic implementation.
<!-- npu="310p" id18 -->
<details>

<summary><term>Atlas inference products</term></summary>

- **General Constraints:**
  <a id="general-constraints"></a>
  - In the current version, yScale, x1Offset, x2Offset, and yOffset are not supported. Pass nullptr for these parameters.

  <details>

  <summary><strong>K-C quantization scenario constraints:</strong></summary>
  <a id="K-C-quantization-scenario-constraints"></a>

  - The input and output support the following data type combinations:
  <a id="Input and Output Support the Following Data Type Combinations K-C && K-T"></a>

    | x1 | x2 | x1Scale | x2Scale | x1Offset | x2Offset | yScale | yOffset | Bias | Out |
    | -----------| ------------ | ----------- | -----------  | ----------- |-----------  | -------  | -----------| ------------ | -------------------|
    | INT8       | INT8         | FLOAT32     | FLOAT32      | null        | null        | null     | null       | null/INT32   | FLOAT16            |

  - x1 constraints:
    - The last dimension of x1 cannot exceed 65535, and transposeX1 supports only false.
  - Constraints on x2:
    - The last dimension size of x2 cannot exceed 65535, and transposeX2 only supports true.
    - The dimensions are: (batch, k1, n1, n0, k0), where batch may be absent, k0 = 32, n0 = 16. The k in x1 shape and k1 in x2 shape must satisfy the following relationship: ceil(k / 32) = k1.
    - x2 needs to be processed by [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) to convert x2 in ND format into an AI processor-affine data layout.
  - Constraints on x1Scale: The data format supports ND, and the shape is 1D (t,), where t = m, and m is the same as m of x1.
  - x2Scale constraint: The data format supports ND. The shape is 1D (t,), where t = n, and n is consistent with the n of x2.
  - Bias constraint: The data format supports ND, and the shape supports 1D (n,) or 3D (batch, 1, n), where n is consistent with n of x2.

  </details>

</details>
<!-- end id18 -->
<!-- npu="A3,910b" id19 -->
<details>

<summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term></summary>

- **General Constraints:**
  <a id="general-constraint1"></a>
  - When transposeX1 is false, the dimensions of X1 represent: (batch, m, k). When transposeX1 is true, the dimensions of X1 represent: (batch, k, m). The batch dimension may be absent.
  - The constraints on transposeX2 are as follows:
    - In ND format, when transposeX2 is false, the dimensions of X2 are: (batch, k, n). When transposeX2 is true, the dimensions are: (batch, n, k). The batch dimension may be absent. The k dimension must be consistent with the k dimension in the shape of X1.
    - In NZ format:
      - When set to true, the X2 dimensions are: (batch, k1, n1, n0, k0), where batch may be absent, k0 = 32, n0 = 16. The k in x1 shape and k1 in x2 shape must satisfy the following relationship: ceil(k / 32) = k1.
      - When set to false, the X2 dimensions are: (batch, n1, k1, k0, n0), where batch may be absent, k0 = 16, n0 = 32. The k in x1 shape and k1 in x2 shape must satisfy the following relationship: ceil(k / 16) = k1.
      - The aclnnCalculateMatmulWeightSizeV2 API and aclnnTransMatmulWeight API can be used to convert the input format from ND to the NZ format.
  - When the original input type of x2Scale does not meet the combination in the quantization scenario constraints, call the aclnnTransQuantParamV2 API in advance to convert the scale to INT64 or UINT64 data type.
  - yScale is not supported in the current version. Pass nullptr.
  - The shape of out supports 2 to 6 dimensions, (batch, m, n), where batch is optional. The supported data types are FLOAT16, INT8, BFLOAT16, and INT32.
  - When x1 and x2 are INT8, out is INT32, and bias is INT32 or nullptr, each scale does not actually participate in the computation. The formula is as follows:
    - bias INT32

        $$
        out = x1@x2 + bias
        $$

    - Without bias

        $$
        out = x1@x2
        $$

  <details>

  <summary><strong>G-B quantization scenario constraints:</strong></summary>
  <a id="G-B quantization"></a>

  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-gb"></a>

    | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | yOffset | Out |
    | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | FLOAT32      | null       | BFLOAT16                               |

  - Relationship between x1 shape, x2 shape, x1Scale shape, x2Scale shape, bias shape, and groupSize:

    | Quantization Type | x1 Shape | x2 Shape | x1Scale Shape | x2Scale Shape | Bias Shape | [gsM，gsN，gsK] |
    | ----- | ------ | ------ | ----------- | ----------- | ----------- | ----------- |
    | G-B quantization | (m, k) |(n, k)|(m, ceil(k / 128))|(ceil(n / 128),ceil(k / 128))| (n, ) | [1, 128, 128]|

  - Note: In the table above, gsM, gsK, and gsN represent groupSizeM, groupSizeK, and groupSizeN, respectively.
  - Constraints on x1: Currently, k must be aligned with 128 and be a multiple of 4 * 128, and transposeX1 must be false.
  - Constraints on x2: Currently, n must be aligned to 256, k must be aligned to 128 and be a multiple of 4 * 128, and transposeX2 must be true.

  </details>

  <details>

  <summary><strong>T-C && T-T && K-C && K-T quantization scenario constraints:</strong></summary>
  <a id="T-C && T-T && K-C && K-T quantization"></a>

  - The input and output support the following data type combinations:
  <a id="Input and Output Support the Following Data Type Combinations K-C && K-T"></a>

    | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | yOffset | Out |
    | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
    | INT8                      | INT32                     | FLOAT32     | UINT64/INT64    | null        | null     | null         | FLOAT32    | FLOAT16/BFLOAT16                       |
    | INT8                      | INT8                      | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                |
    | INT8                      | INT8                      | null        | UINT64/INT64    | null/FLOAT32| null     | null/INT32   | null       | INT8                                   |
    | INT8                      | INT8                      | null/FLOAT32| FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
    | INT4/INT32                | INT4/INT32                | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                |
    | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16| null        | null     | null/INT32   | null       | INT32                                  |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | FLOAT32      | null       | BFLOAT16                               |
    | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
    | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |

  - x1 constraints:
    - When the data type is INT4, transposeX1 is false. The dimensions are (m, k), and k must be an even number.
    - When the data type is INT32, transposeX1 is false. Each INT32 value stores eight INT4 values, and the corresponding dimension representation is (m, ceil(k / 8)). k must be a multiple of 8.
    - When the data type is INT8 and the data type of x2 is INT32, transposeX1 is false. The dimensions are (m, k), and k must be an even number.
  - Constraints on x2:
    - When the data type is INT4:
      - Currently, only 2D ND format is supported.
      - When transposeX2 is true, the dimensions are (n, k), and k must be an even number.
      - When transposeX2 is false, the dimensions are (k, n), and n must be an even number.
    - When the data type is INT32, each INT32 value stores eight INT4 values.
       - Currently, only 2D ND format is supported.
       - When transposeX2 is true, the dimension is (n, ceil(k / 8)), and k must be a multiple of 8.
       - When transposeX2 is false, the dimension is (k, ceil(n / 8)), and n must be a multiple of 8.
       - Use the aclnnConvertWeightToINT4Pack API to complete the data format conversion of x2 from INT32 (1 int32 stores 1 int4 in bits 0-3) to INT32 (1 int32 stores 8 int4 values) or INT4 (1 int4 represents 1 int4). See specifically [aclnnConvertWeightToINT4Pack API](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md).
  - Constraints on x1Scale: The data format supports ND, and the shape is 1D (t,), where t = m, and m is the same as m of x1.
  - Constraints of x2Scale: The data format supports ND, and the shape is 1D (t,). t = 1 or n, where n is the same as the n of x2. When the x1 data type is INT8 and the x2 data type is int32, only t = n is supported.
  - x2Offset constraint: The data format supports ND. The shape is 1D (t,), where t = 1 or n, and n is consistent with n of x2.
  - bias constraint:
      - The data format supports ND. The shape supports 1D (n,) or 3D (batch, 1, n), where n is consistent with n of x2.
      - When x1 and x2 are INT32 or INT4, the shape of bias supports only 1D (n,).
      - When the shape of out is 2, 4, 5, or 6 dimensions, the shape of bias supports only 1 dimension (n,).
  - Constraint on yOffset: The shape supports 1 dimension (n). It is an auxiliary result computed offline during the computation process, with the value required to be 8\*x2\*x2Scale, and accumulated along the first dimension.

  </details>

  <details>

  <summary><strong>K-G Quantization Scenario Constraints:</strong></summary>
  <a id="K-G-quantization"></a>

  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-k-g"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | yOffset | Out |
      | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
      | INT8                      | INT32                     | FLOAT32     | UINT64/INT64    | null        | null     | null         | FLOAT32    | FLOAT16/BFLOAT16                       |
      | INT4                      | INT4                      | FLOAT32     | FLOAT32         | FLOAT16     | null     | null         | null       | BFLOAT16                               |

  - Relationship between x1, x2, x1Scale, x2Scale, and groupSize values:

    | Quantization Type | x1 Data Type | x2 Data Type | x1Scale Data Type | x2Scale Data Type | x1 Shape | x2 Shape | x1Scale Shape | x2Scale Shape | x2Offset Shape | yOffset Shape | [gsM, gsN, gsK] |
    | ----- | ------------------------- | ------------------------- | -------------- | ------------- | -------- | ------- | ------------ | ------ |------------ | ------------ | ------------ |
    | K-G quantization | INT8                    |INT32                   |FLOAT32              |UINT64/INT64 |(m, k) |(k, ceil(n / 8))|(m, 1)|(ceil(k / 256), n)|null| (n) | [0, 0, 256]|
    | K-G quantization | INT4                    |INT4                    |FLOAT32              |FLOAT32      |(m, k)|(n, k)|(m, 1)|(ceil(k / 256), n)|(ceil(k / 256), n)| null | [0, 0, 256]|

  - x1 constraints:
    - When the data type is INT8, k must be aligned to 256 and less than 29576. transposeX1 is false.
    - When the data type is INT4, k must be aligned to 1024. transposeX1 is false.
  - Constraints on x2:
    - When the data type is INT32, k must be aligned to 256. transposeX2 is false.
    - When the data type is INT4, k must be aligned to 1024, and n must be aligned to 256. transposeX2 is true.
  - Constraints on x2Scale:
    - When the data type is UINT64/INT64, since TransQuantParamV2 only supports one dimension, you need to view x2Scale as one-dimensional (k / groupSize * n), then call the aclnn API of the TransQuantParamV2 operator to convert x2Scale to the UINT64/INT64 data type, and then view the output as two-dimensional (k / groupSize, n), where the groupSize value is 256.
    - When x1 and x2 are INT4, the shape of x2Scale is (ceil(k / 256), n).

  </details>

</details>
<!-- end id19 -->
<!-- npu="950" id20 -->
<details>

<summary><term>Ascend 950PR&950DT products</term></summary>

- **General Constraints:**
  <a id="general-constraint2"></a>

  - int4 input is supported only when transposeX1 is false.
  - When transposeX1 is false, the shape of x1 is (batch, m, k). When transposeX1 is true, the shape of x1 is (batch, k, m). Here, batch represents the first 0 to 4 dimensions, where 0 dimension indicates that batch does not exist.
  - When transposeX2 is false, the shape of x2 is (batch, k, n). When transposeX2 is true, the shape of x2 is (batch, n, k). Here, batch represents the first 0 to 4 dimensions, and 0 dimension means batch does not exist. k is consistent with k in the shape of x1.
  - When the original input type of x2Scale does not meet the combination in the quantization scenario constraints, call the aclnnTransQuantParamV2 API in advance to convert the scale to INT64 or UINT64 data type.
  - yScale is supported only when x1 is FLOAT8_E4M3FN and x2 is FLOAT4_E2M1. It is not supported in other scenarios. The shape is 2D (1, n), where n is consistent with n of x2.
  - x2Offset is supported only when the data types of x1 and x2 are INT8 and the data type of out is INT8. For other input types, nullptr must be passed. The shape is 1D (t, ), where t = 1 or n, and n is consistent with n of x2.
  - yOffset is a reserved parameter and is not supported in the current version. Pass nullptr or an empty tensor.
  - bias-related constraints:
    - Optional parameter. Supports passing nullptr.
    - When the shape of out is 2-dimensional, the shape of bias supports 1-dimensional (n,) or 2-dimensional (1, n).
    - When the shape of out is 4-, 5-, or 6-dimensional, the shape of bias supports 1-dimensional (n,).
    - When the shape of out is 3 dimensions, the shape of bias supports 1D (n,) or 3D (batch, 1, n).
  - Constraints on groupSize:
    - Takes effect only in the MX, G-B, B-B, K-G, and T-CG [quantization modes](../../../docs/en/context/quant_mode_introduction.md).
    - The groupSize value is valid only when both x1Scale and x2Scale inputs are 2-dimensional or higher. In other scenarios, pass 0.
    - The input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK using the following formula. When one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shapes of x1, x2, x1Scale, and x2Scale for computation. Principle: If groupSizeM = 0, it indicates that the quantization group value in the m direction is inferred by the API. The inference formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is consistent with the m in the x1 shape, and scaleM is consistent with the m in the x1Scale shape.

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

  - The shape of out supports 2 to 6 dimensions, (batch, m, n). batch may be absent. Broadcasting of the batch dimensions of x1 and x2 is supported. The output batch is consistent with the batch after broadcasting, m is consistent with the m of x1, and n is consistent with the n of x2.
  - When x1 and x2 are INT8, out is INT32, and bias is INT32 or nullptr, each scale does not actually participate in the computation. The formula is as follows:
    - bias INT32

        $$
        out = x1@x2 + bias
        $$

    - Without bias

        $$
        out = x1@x2
        $$

  <details>

  <summary><strong>Constraints for T-C quantization && T-T quantization use cases:</strong></summary>
  <a id="T-C quantization && T-T quantization"></a>

  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-tc-tt"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | Out |
      | ------------------------- | ------------------------- | ----------- |   ----------- | -------- | -------| ------- |   -------------------------------------- |
      | INT8                      | INT8                      | null        | UINT64/INT64      | null     | null     | null/INT32   | FLOAT16/BFLOAT16                       |
      | INT8                      | INT8                      | null        | UINT64/INT64      | null/FLOAT32  | null     | null/INT32   |   INT8                              |
      | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16  | null     | null     | null/INT32/FLOAT32/BFLOAT16   |   BFLOAT16              |
      | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16  | null     | null     | null/INT32   |   INT32                                |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | null        | UINT64/INT64      | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |
      | HIFLOAT8                  | HIFLOAT8                  | null        | UINT64/INT64      | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32      |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     | FLOAT32           | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     | FLOAT32           | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      | INT4/INT32                  | INT4/INT32                  | null     | UINT64/INT64           | null     | null     | null/INT32 | FLOAT16               |

  - x1 constraints:
    - When the data type is INT4, transposeX1 is false. The dimensions are (m, k), and k must be an even number.
    - When the data type is INT32, transposeX1 is false. Each INT32 value stores eight INT4 values, and the corresponding dimension representation is (m, ceil(k / 8)). k must be a multiple of 8.
    - When the data type is INT8 and the data type of x2 is INT32, transposeX1 is false. The dimensions are (m, k), and k must be an even number.
  - Constraints on x2:
    - When the data type is INT4:
      - Currently, only 2D ND format is supported.
      - When transposeX2 is true, the dimensions are (n, k), and k must be an even number.
      - When transposeX2 is false, the dimensions are (k, n), and n must be an even number.
    - When the data type is INT32, each INT32 value stores eight INT4 values.
       - Currently, only 2D ND format is supported.
       - When transposeX2 is true, the dimension is (n, ceil(k / 8)), and k must be a multiple of 8.
       - When transposeX2 is false, the dimension is (k, ceil(n / 8)), and n must be a multiple of 8.
  - In the T-T quantization scenario, the shape of x1Scale is (1,) or nullptr, and the shape of x2Scale is (1,).
  - In the T-C quantization scenario, the shape of x1Scale is (1,) or nullptr, and the shape of x2Scale is (n,), where n is consistent with the n of x2.
  - When the data type of x1/x2 is FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, static quantization and dynamic quantization are distinguished. For static quantization, the x2Scale data type is UINT64/INT64; for dynamic quantization, the x2Scale data type is FLOAT32. When the x1/x2 data type is INT8, INT4, or INT32, dynamic T-C or dynamic T-T quantization is not supported.
  - In static quantization use cases, when x1/x2 is INT4 or INT32, x1 supports 2-6 dimensions and x2 supports only 2 dimensions.
  - Batch consistency: To satisfy batch consistency in the T-T quantization and T-C quantization scenarios, when comparing inputs with different M values, if x1Scale is not nullptr, the input value of x1Scale must not dynamically change with the input value of x1 and must remain unchanged.

  </details>

  <details>

  <summary><strong>Constraints on K-C quantization and K-T quantization scenarios:</strong></summary>
  <a id="K-C quantization && K-T quantization"></a>

  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-kc-kt"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | Out |
      | ------------------------- | ------------------------- | ----------- | ----------- | -------- | -------|   ------- | -------------------------------------- |
      | INT8                      | INT8                      | FLOAT32| FLOAT32/BFLOAT16  | null     | null     |  null/INT32/FLOAT32/BFLOAT16   | BFLOAT16              |
      | INT8                      | INT8                      | FLOAT32     | FLOAT32           | null     |  null     | null/INT32/FLOAT32/FLOAT16  | FLOAT16                 |
      | INT4                      | INT4                      | FLOAT32| FLOAT32/BFLOAT16  | null     | null     |  null/INT32/FLOAT32/BFLOAT16   | BFLOAT16              |
      | INT4                      | INT4                      | FLOAT32     | FLOAT32           | null     |  null     | null/INT32/FLOAT32/FLOAT16  | FLOAT16                 |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     | FLOAT32           | null     |  null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     | FLOAT32           | null     |  null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |

    - In the K-C quantization scenario, the shape of x1Scale is (m,), and the shape of x2Scale is (n,), where m matches the m of x1, and n matches the n of x2.
    - In the K-T quantization scenario, the shape of x1Scale is (m,), and the shape of x2Scale is (1,), where m matches the m of x1.

  </details>

  <details>

  <summary><strong>Constraints on G-B quantization and B-B quantization scenarios:</strong></summary>
  <a id="G-B quantization && B-B quantization"></a>

  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-GB/BB"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | Out |
      | ------------------------- | ------------------------- | ----------- | ----------- |   -------- | -------| ------- | -------------------------------------- |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     |   FLOAT32           | null     | null     | null | FLOAT16/BFLOAT16/  FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     |   FLOAT32           | null     | null     | null | FLOAT16/BFLOAT16/  FLOAT32               |
      | INT8                      |INT8                       | FLOAT32     |   FLOAT32           | null     | null     | FLOAT32 |BFLOAT16       |

  - Relationship between x1, x2, x1Scale, x2Scale, and groupSize values:

    | Quantization Type | x1 Shape | x2 Shape | x1Scale Shape | x2Scale Shape | yScale Shape | [gsM，gsN，gsK] | groupSize |
    |-------|--------|--------|-------------|-------------|------------|---|---|
    |B-B quantization|<li>Non-transposed: (batch, m, k)</li><li>Transposed: (batch, k, m)</li>|<li>Non-transposed: (batch, k, n)</li><li>Transposed: (batch, n, k)</li>|<li>Non-transposed: (batch, ceil(m / 128), ceil(k / 128))</li><li>Transposed: (batch, ceil(k / 128), ceil(m / 128))</li>|<li>Non-transposed: (batch, ceil(k / 128), ceil(n / 128))</li><li>Transposed: (batch, ceil(n / 128), ceil(k / 128))</li>|null|[128, 128, 128]|549764202624|
    |G-B quantization|<li>Non-transposed: (batch, m, k)</li><li>Transposed: (batch, k, m)</li>|<li>Non-transposed: (batch, k, n)</li><li>Transposed: (batch, n, k)</li>|<li>Non-transposed: (batch, m, ceil(k / 128))</li><li>Transposed: (batch, ceil(k / 128), m)</li>|<li>Non-transposed: (batch, ceil(k / 128), ceil(n / 128))</li><li>Transposed: (batch, ceil(n / 128), ceil(k / 128))</li>|null|[1, 128, 128]|4303356032|

  - Note: In the preceding table, gsM, gsK, and gsN represent groupSizeM, groupSizeK, and groupSizeN, respectively. Dimensions where gsM, gsK, or gsN is 0 are automatically inferred. The preceding table lists the cases where automatic inference is not required.
  - In the G-B quantization and B-B quantization use cases, the transpose attributes of x1 and x1Scale must be consistent, and the transpose attributes of x2 and x2Scale must be consistent (when the shape contains an axis of 1 and the dynamic graph NZ use case is not involved, the transpose attributes of x and scale can be inconsistent).
  - In G-B quantization scenarios, bias is supported only for INT8 input. Bias is not supported in other scenarios.
  - In B-B quantization scenarios, INT8 input is not supported, and bias is not supported.
  - Batch consistency: The B-B quantization use case does not support batch consistency. Even if the batch consistency switch is enabled, the output is not guaranteed to meet batch consistency requirements.

  </details>

  <details>

  <summary><strong>MX quantization use case constraints:</strong></summary>
  <a id="mx-quantization"></a>

  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-mx"></a>

      | Quantization Type | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | Out |
      |-------| ------------------------- | ------------------------- | ----------- | ----------- |   -------- | -------| ------- | -------------------------------------- |
      |MX full quantization| FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      |MX full quantization| FLOAT4_E2M1                | FLOAT4_E2M1                | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      |MX pseudo quantization| FLOAT8_E4M3FN             | FLOAT4_E2M1               | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/BFLOAT16/FLOAT16 |  BFLOAT16/FLOAT16                               |

  - Relationship among the x1 data type, x2 data type, x1, x2, x1Scale, x2Scale, and groupSize:

    | Quantization Type | x1 Data Type | x2 Data Type | x1 Shape | x2 Shape | x1Scale Shape | x2Scale Shape | Bias Shape | yScale Shape | [gsM, gsN, gsK] | groupSize |
    |-------|--------|--------|--------|--------|-------------|-------------|------------|---------------------------------------|--|--|
    |MX full quantization|FLOAT8_E4M3FN/FLOAT8_E5M2|FLOAT8_E4M3FN/FLOAT8_E5M2|<li>Non-transposed: (batch, m, k)</li><li>Transposed: (batch, k, m)</li>|<li>Non-transposed: (batch, k, n)</li><li>Transposed: (batch, n, k)</li>|<li>Non-transposed: (batch, m, ceil(k / 64), 2)</li><li>Transposed: (batch, ceil(k / 64), m, 2)</li>|<li>Non-transposed: (batch, ceil(k / 64), n, 2)</li><li>Transposed: (batch, n, ceil(k / 64), 2)</li>|(n,) or (batch, 1, n)|null|[1, 1, 32]|4295032864|
    |MX full quantization|FLOAT4_E2M1|FLOAT4_E2M1|<li>Non-transposed: (batch, m, k)</li><li>Transposed: (batch, k, m)</li>|<li>Non-transposed: (batch, k, n)</li><li>Transposed: (batch, n, k)</li>|<li>Non-transposed: (batch, m, ceil(k / 64), 2)</li><li>Transposed: (batch, ceil(k / 64), m, 2)</li>|<li>Non-transposed: (batch, ceil(k / 64), n, 2)</li><li>Transposed: (batch, n, ceil(k / 64), 2)</li>|(n,) or (batch, 1, n)|null|[1, 1, 32]|4295032864|
    |MX pseudo quantization|FLOAT8_E4M3FN|FLOAT4_E2M1|(m, k)|(n, k)|(m, ceil(k / 64), 2)|(n, ceil(k / 64), 2)|(1, n)|null|[0, 0, 32]/[1, 1, 32]|32/4295032864|

  - Note: In the preceding table, gsM, gsK, and gsN indicate groupSizeM, groupSizeK, and groupSizeN, respectively. Dimensions with gsM, gsK, or gsN set to 0 are automatically inferred. The preceding table lists the cases where automatic inference is not required.
  - In the MX full quantization use case, when the x2 data type is FLOAT8_E4M3FN/FLOAT8_E5M2, the transpose attributes of x1 and x1Scale must be consistent, and the transpose attributes of x2 and x2Scale must be consistent (when the shape contains an axis of 1 and the dynamic graph NZ use case is not involved, the transpose attributes of x and scale can be inconsistent).
  - In the MX full quantization scenario, when the x2 data type is FLOAT4_E2M1, the inner axis of x1 and x2 must be even, and k must be greater than 2.
  - In the MX full quantization scenario, only the last three axes of x1Scale and x2Scale support <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensors</a>.
  - In the MX pseudo quantization scenario, when the x2 data type is FLOAT4_E2M1, the batch axis is not supported, and k must be a multiple of 8.
    - Constraints on x1: only non-transpose is supported. Only the ND data format is supported.
    - Constraints on x2: only transpose is supported. The data format supports ND and NZ. If x2 is in NZ format, n must be a multiple of 8.
  - In the MX pseudo quantization scenario, bias is an optional parameter. The data type supports BFLOAT16 or FLOAT16 and must be the same as the output type. The data format supports ND. The shape supports two dimensions and is expressed as (1, n). If this parameter is not required, pass in nullptr.

  </details>

  <details>

  <summary><strong>T-CG quantization scenario constraints:</strong></summary>
  <a id="t-cg-quantization"></a>

  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-tcg"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | Out |
      | ------------------------- | ------------------------- | ----------- | ----------- |     -------- | -------| ------- | -------------------------------------- |
      | FLOAT8_E4M3FN             | FLOAT4_E2M1               | null        |    BFLOAT16/FLOAT16 | null     | INT64/UINT64    | null         |     BFLOAT16/FLOAT16 |

  - Relationship between x1, x2, x1Scale, x2Scale, and groupSize values:

    | Quantization Type | x1 Shape | x2 Shape | x1Scale Shape | x2Scale Shape | yScale Shape | [gsM，gsN，gsK] | groupSize |
    |-------|--------|--------|-------------|-------------|------------|---------------------------------------|--|
    |T-CG quantization|(m, k)|(n, k)|null|(n, ceil(k / 32))|(1, n)|[0, 0, 32]/[1, 1, 32]|32/4295032864|

  - In T-CG quantization mode, the yScale data type supports INT64 and UINT64, the data format supports ND, and the shape supports 2 dimensions, expressed as (1, n). When the original input type does not meet the data type combinations in the constraints and restrictions, call the aclnn API of the TransQuantParamV2 operator in advance to convert it to the UINT64 data type. When the input data type is INT64, INT64 is internally treated as UINT64.
  - In T-CG quantization mode, bias is a reserved parameter and is not supported in the current version. Pass nullptr.
  - In T-CG quantization mode, k must be a multiple of 32.
    - Constraints on x1: Only non-transpose is supported. Only the ND data format is supported.
    - Constraints on x2: Only transpose is supported when x2 is in ND format; only non-transpose is supported when x2 is in NZ format.
  - In T-CG quantization mode, the [groupSizeM, groupSizeN, groupSizeK] value combinations support [0, 0, 32] and [1, 1, 32], with corresponding groupSize values of 32 and 4295032864, respectively.
  - In T-CG quantization mode, out and x2Scale must have the same data type.

  </details>

  <details>

  <summary><strong>K-G quantization scenario constraints:</strong></summary>
  <a id="K-G-quantization"></a>

  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-k-g"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | Bias | yOffset | Out |
      | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
      | INT4                      | INT4                      | FLOAT32     | FLOAT32         | FLOAT16     | null     | null         | null       | BFLOAT16/FLOAT16                               |

  - Relationship between x1, x2, x1Scale, x2Scale, and groupSize values:

    | Quantization Type | x1 Data Type | x2 Data Type | x1Scale Data Type | x2Scale Data Type | x1 Shape | x2 Shape | x1Scale Shape | x2Scale Shape | x2offset | yOffset Shape | [gsM, gsN, gsK] |
    | ----- | ------------------------- | ------------------------- | -------------- | ------------- | -------- | ------- | ------------ | ------------ | ------------| ------------ | ------------ |
    | K-G quantization | INT4                    |INT4                    |FLOAT32              |FLOAT32             |(m, k)|(n, k)|(m, 1)|(ceil(k / 256), n)|(ceil(k / 256), n)| null | [0, 0, 256]|

  - x1 constraints:
    - When the data type is INT4, k must be aligned to 1024. transposeX1 is false.
  - Constraints on x2:
    - When the data type is INT4, k must be aligned to 1024, and n must be aligned to 256. transposeX2 is true.
  - Constraints on x2Scale:
    - When x1 and x2 are INT4, the shape of x2Scale is (ceil(k / 256), n).

  </details>

</details>
<!-- end id20 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950" id14 -->
- <term>Ascend 950PR&950DT products</term>:
<!-- end id14 -->

x1 is FLOAT8_E4M3FN, x2 is FLOAT4_E2M1, x2Scale is BFLOAT16, and yScale is UINT64.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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
      // Boilerplate for resource initialization.
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
      // Call aclrtMalloc to apply for device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  // Convert the uint16_t representation of bfloat16 to float.
  float Bf16ToFloat(uint16_t h)
  {
      uint32_t sign = (h & 0x8000U) ? 0x80000000U : 0x00000000U; // sign bit
      uint32_t exponent = (h >> 7) & 0x00FFU;                    // exponent bits
      uint32_t mantissa = h & 0x007FU;                           // mantissa bits
      // The exponent offset remains unchanged.
      // Shift the mantissa left by 23 - 7, and pad the remaining bits with 0s.
      uint32_t fBits = sign | (exponent << 23) | (mantissa << (23 - 7));
      // Cast to float.
      return *reinterpret_cast<float*>(&fBits);
  }

  int AclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API.
      std::vector<int64_t> x1Shape = {5, 64};
      std::vector<int64_t> x2Shape = {8, 64};
      std::vector<int64_t> x2ScaleShape = {8, 2};
      std::vector<int64_t> yScaleShape = {1, 8};
      std::vector<int64_t> outShape = {5, 8};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* yScaleDeviceAddr = nullptr;
      void* quantParamDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* yScale = nullptr;
      aclTensor* yOffset = nullptr;
      aclTensor* quantParam = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      std::vector<uint8_t> x1HostData(5 * 64, 0b00111000);                 // 0b00111000 is 1.0 in fp8_e4m3fn.
      std::vector<uint8_t> x2HostData(8 * 64 / 2, 0b0010 + (0b0010 << 4)); // 0b0010 is 1.0 in fp4_e2m1. Here, uint8 represents 2 fp4 values.
      std::vector<uint16_t> x2ScaleHostData(8 * 2, 0b0011111110000000); // 1.0 in bf16
      std::vector<float> yScaleHostData(8, 1.0);                        // 1.0 in fp32
      std::vector<uint64_t> quantParamHostData(8, 0);
      std::vector<uint16_t> outHostData(40, 0); // It is actually bfloat16.
      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_BF16, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the yScale aclTensor.
      ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yScaleTensorPtr(yScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yScaleDeviceAddrPtr(yScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the quantParam aclTensor.
      ret = CreateAclTensor(quantParamHostData, yScaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> quantParamTensorPtr(quantParam, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = true;
      int64_t groupSize = 32;

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;

      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(yScale, yOffset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceQuantParamAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceQuantParamAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceQuantParamAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceQuantParamAddrPtr.reset(workspaceQuantParamAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceQuantParamAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      workspaceSize = 0;
      executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5GetWorkspaceSize(
          x1, x2, nullptr, x2Scale, quantParam, nullptr, nullptr, nullptr, bias, transposeX1, transposeX2, groupSize, out,
          &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronously wait for the task to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to the host. Modify it based on the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0); // In C, fp16 data cannot be printed directly. You need to read it as uint16 and convert it to fp16 through binary conversion.
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %.1f\n", i, Bf16ToFloat(resultData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId according to the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- npu="950" id15 -->
- <term>Ascend 950PR&950DT products</term>:
<!-- end id15 -->

x1 is INT8, x2 is INT8, x1Scale is FLOAT32, x2Scale is FLOAT32, and bias is INT32.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

      int64_t
      GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // Boilerplate: initialize resources.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs. Custom construction is required based on the API
      std::vector<int64_t> x1Shape = {5, 16};
      std::vector<int64_t> x2Shape = {16, 8};
      std::vector<int64_t> biasShape = {8};
      std::vector<int64_t> x1ScaleShape = {5};
      std::vector<int64_t> x2ScaleShape = {8};
      std::vector<int64_t> outShape = {5, 8};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x2OffsetDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(80, 1);
      std::vector<int8_t> x2HostData(128, 1);
      std::vector<int32_t> biasHostData(8, 1);
      std::vector<float> x2ScaleHostData(8, 1);
      std::vector<float> x1ScaleHostData(5, 1);
      std::vector<uint16_t> outHostData(40, 1);  // actually in float16 half-precision format
      // Create the x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the bias aclTensor
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, bias,
                                               transposeX1, transposeX2, 0, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, and copy the result from device memory to the host. Modify it according to the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The fp16 data cannot be printed directly in C. Read it as uint16 and convert the binary data to fp16 manually.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
      // Fill in deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- npu="950" id16 -->
- <term>Ascend 950PR&950DT products</term>:
<!-- end id16 -->

x1 and x2 are FLOAT8_E4M3FN, x1Scale is FLOAT32, x2Scale is FLOAT32, no x2Offset, and bias is FLOAT32.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

      int64_t
      GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to apply for device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output, which needs to be custom constructed based on the API.
      std::vector<int64_t> x1Shape = {16, 16};
      std::vector<int64_t> x2Shape = {16, 16};
      std::vector<int64_t> biasShape = {16};
      std::vector<int64_t> x1ScaleShape = {1};
      std::vector<int64_t> x2ScaleShape = {1};
      std::vector<int64_t> outShape = {16, 16};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x2OffsetDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(256, 1);
      std::vector<int8_t> x2HostData(256, 1);
      std::vector<float> biasHostData(16, 1);
      std::vector<float> x2ScaleHostData(1, 1);
      std::vector<float> x1ScaleHostData(1, 1);
      std::vector<uint16_t> outHostData(256, 1);  // Actually in float16 half precision.

      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, x2Offset, nullptr, bias,
                                               transposeX1, transposeX2, 0, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // fp16 data cannot be printed directly in C. Read it as uint16 and manually convert the binary value to fp16.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Initialize the device/stream. For details, see the acl API documentation.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- npu="A3,910b" id17 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
<!-- end id17 -->

x1 is INT8, x2 is INT32, x1Scale is FLOAT32, and x2Scale is UINT64.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // Resource initialization (boilerplate).
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to apply for memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API.
      std::vector<int64_t> x1Shape = {1, 8192};     // (m,k)
      std::vector<int64_t> x2Shape = {8192, 128};  // (k,n)
      std::vector<int64_t> yoffsetShape = {1024};

      std::vector<int64_t> x1ScaleShape = {1,1};
      std::vector<int64_t> x2ScaleShape = {32, 1024}; // x2ScaleShape = [KShape / groupsize, N]
      std::vector<int64_t> outShape = {1, 1024};

      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *yoffsetDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *yoffset = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
      std::vector<int32_t> x2HostData(GetShapeSize(x2Shape), 1);
      std::vector<int32_t> yoffsetHostData(GetShapeSize(yoffsetShape), 1);
      std::vector<int32_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
      float tmp = 1;
      uint64_t x2ScaleValue = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
      std::vector<int64_t> x2ScaleHostData(GetShapeSize(x2ScaleShape), x2ScaleValue);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 1);  // Actually float16 half-precision mode

      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_UINT64, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the yoffset aclTensor.
      ret = CreateAclTensor(yoffsetHostData, yoffsetShape, &yoffsetDeviceAddr, aclDataType::ACL_FLOAT, &yoffset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yoffsetTensorPtr(yoffset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> yoffsetDeviceAddrPtr(yoffsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      int64_t groupSize = 256;

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, yoffset, nullptr,
                                              transposeX1, transposeX2, groupSize, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0);  // fp16 data cannot be printed directly in C. Read it out as uint16 and manually convert it to fp16 through binary.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId based on your actual device.
      int32_t deviceId = 1;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```
