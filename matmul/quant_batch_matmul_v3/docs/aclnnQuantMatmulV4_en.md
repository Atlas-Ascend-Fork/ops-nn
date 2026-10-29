# aclnnQuantMatmulV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:34:48.498Z -->

**Notice: This API will be deprecated in later versions. Use the latest aclnnQuantMatmulV5 API instead.**

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

- Description: Compatible with the aclnnQuantMatmulV3 API and supports the K-C && K-T [quantization mode](../../../docs/en/context/quant_mode_introduction.md) on top of it. Computes quantized matrix multiplication. The minimum supported input dimension is 2, and the maximum is 6. Similar APIs include aclnnMm (matrix multiplication that only supports 2D tensors as input) and aclnnBatchMatMul (matrix multiplication that only supports 3D tensors, where the first dimension is the Batch dimension).
- Formula:

  <!-- npu="950,A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
    - No pertoken, no bias:

      $$
      out = x1@x2 * scale + offset
      $$

    - bias INT32:

      $$
      out = (x1@x2 + bias) * scale + offset
      $$

    - bias BFLOAT16/FLOAT32 (no offset in this scenario):

      $$
      out = x1@x2 * scale + bias
      $$

    - pertoken without bias:

      $$
      out = x1@x2 * scale * pertokenScaleOptional
      $$

    - pertoken and bias INT32 (offset Not Applicable in This Scenario):

      $$
      out = (x1@x2 + bias) * scale * pertokenScaleOptional
      $$

    - pertoken and bias BFLOAT16/FLOAT16/FLOAT32 (offset Not Applicable in This Scenario):

      $$
      out = x1@x2 * scale * pertokenScaleOptional + bias
      $$

  <!-- end id7 -->
  <!-- npu="310p" id8 -->
  - <term>Atlas inference products</term>:
    - Without pertokenScaleOptional and bias:

      $$
      out = x1@x2 * scale + offset
      $$

    - Without pertokenScaleOptional, with INT32 bias:

      $$
      out = (x1@x2 + bias) * scale + offset
      $$

    - With pertokenScaleOptional, without bias

      $$
      out = x1@x2 * scale * pertokenScaleOptional
      $$

    - With pertokenScaleOptional, with INT32 bias

      $$
      out = (x1@x2 + bias) * scale * pertokenScaleOptional
      $$

    - With pertokenScaleOptional, with FLOAT32 bias

      $$
      out = x1@x2 * scale * pertokenScaleOptional + bias
      $$

  <!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnQuantMatmulV4GetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that includes the operator computation process, and then call the "aclnnQuantMatmulV4" API to perform the computation.

```cpp
aclnnStatus aclnnQuantMatmulV4GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *scale,
    const aclTensor *offset,
    const aclTensor *pertokenScaleOptional,
    const aclTensor *bias,
    bool             transposeX1,
    bool             transposeX2,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulV4(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantMatmulV4GetWorkspaceSize

- **Parameters:**
  <table style="table-layout: fixed; width: 1552px"><colgroup>
  <col style="width: 198px">
  <col style="width: 121px">
  <col style="width: 220px">
  <col style="width: 450px">
  <col style="width: 165px">
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
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
  <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input x1 in the formula.</td>
      <td>        <ul>
            <li>Supports non-contiguous tensors where the last two axes are transposed. Other cases of <a href="../../../docs/en/context/non_contiguous_tensor.md"> non-contiguous tensor</a> are not supported.</li>
            <li>If transposeX1 is false, the shape is (batch, m, k).</li>
            <li>If transposeX1 is true, the shape is (batch, k, m). batch may be absent.</li>
        </ul>
    </td>
      <td>INT8, INT32, INT4</td>
      <td>ND</td>
      <td>2-6</td>
      <td>×</td>
  </tr>
  <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>        <ul>
            <li>In NZ format, the shape supports 4-8 dimensions.</li>
                <li>If transposeX2 is true, each dimension indicates: (batch, k1, n1, n0, k0). batch may be absent. k0 = 32 and n0 = 16. k in the x1 shape and k1 in the x2 shape must satisfy the following relationship: ceil(k / 32) = k1.</li>
                <li>If transposeX2 is false, each dimension indicates: (batch, n1, k1, k0, n0). batch may be absent. k0 = 16 and n0 = 32. k in the x1 shape and k1 in the x2 shape must satisfy the following relationship: ceil(k / 16) = k1.</li>
                <li>The aclnnCalculateMatmulWeightSizeV2 API and the aclnnTransMatmulWeight API can be used to convert the input format from ND to NZ.</li>
            <li>In ND format, non-contiguous tensors with the last two axes transposed are supported. <a href="../../../docs/en/context/non_contiguous_tensor.md">Non-contiguous tensors</a> in other use cases are not supported.</li>
            <li>If transposeX2 is false, the shape is (batch, k, n).</li>
            <li>If transposeX2 is true, the shape is (batch, n, k), where batch may be absent and k is consistent with k in the x1 shape.</li>
        </ul>
      </td>
      <td>INT8, INT32, INT4</td>
      <td>ND, NZ</td>
      <td>2-6 (ND)<br>4-8 (NZ)</td>
      <td>×</td>
  </tr>
  <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Indicates the quantization parameter, that is, the input scale in the formula.</td>
      <td>        <ul>
            <li>The shape is 1-D (t,). t = 1 or n, where n is consistent with n in the x2 shape.</li>
            <li>If the original input type does not match the type combinations in <a href="#constraints">Constraints</a>, call the aclnn API of the TransQuantParamV2 operator in advance to convert scale to the INT64 or UINT64 data type.</li>
        </ul>
      </td>
      <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
  </tr>
  <tr>
      <td>offset</td>
      <td>Optional Input</td>
      <td>Input offset in the formula.</td>
      <td>        <ul>
            <li>The shape is 1-dimension (t,), t = 1 or n, where n is consistent with n of x2.</li>
            <li>When the data type of out is INT8, offset can exist. For other input data types, nullptr must be passed in.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
  </tr>
  <tr>
      <td>pertokenScaleOptional</td>
      <td>Optional Input</td>
      <td>        <ul>
            <li>Input pertokenScaleOptional in the formula.</li>
            <li>The shape is 1D (t,). t = m, where m is consistent with the m of x1.</li>
        </ul>
      </td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
  </tr>
  <tr>
      <td>bias</td>
      <td>Optional Input</td>
      <td>Input bias in the formula.</td>
      <td>        <ul>
            <li>The shape supports 1 dimension (n,) or 3 dimensions (batch, 1, n), where n is consistent with the n of x2.</li>
            <li>When the shape of out is 2, 4, 5, or 6 dimensions, the shape of bias only supports 1 dimension (n,).</li>
        </ul>
      </td>
      <td>INT32, BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1, 3</td>
      <td>-</td>
  </tr>
  <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Indicates whether the input shape of x1 contains transpose.</td>
      <td>        <ul>
            <li>When false, the shape is (batch, m, k).</li>
            <li>When true, the shape is (batch, k, m), where batch may be absent.</li>
        </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Indicates whether the input shape of x2 includes transpose.</td>
      <td>        <ul>
            <li>When false, the shape is (batch, k, n).</li>
            <li>When true, the shape is (batch, n, k), where batch may be absent and k is consistent with k in the shape of x1.</li>
            <li>In NZ format:</li>
                <li>When transposeX2 is true, the shape is (batch, k1, n1, n0, k0), where batch may be absent, k0 = 32, and n0 = 16. k in the shape of x1 and k1 in the shape of x2 must satisfy the following relationship: ceil(k / 32) = k1.</li>
                <li>When transposeX2 is false, in each case the dimensions indicate: (batch, n1, k1, k0, n0), where batch may be absent, k0 = 16, and n0 = 32. k in the shape of x1 and k1 in the shape of x2 must satisfy the following relationship: ceil(k / 16) = k1.</li>
      </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output out in the formula.</td>
      <td>batch may be absent. The batch dimensions of x1 and x2 support broadcast. The output batch is consistent with the batch after broadcast. m is consistent with m of x1, and n is consistent with n of x2.</td>
      <td>FLOAT16, INT8, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>(batch, m, n)</td>
      <td>✓</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied on the Device side.</td>
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
  </tbody></table>

  <!-- npu="310p" id9 -->
  - <term>Atlas inference products</term>:
    - The size of the last dimension of x1 cannot exceed 65535. The last dimension of x1 refers to m when transposeX1 is true, or k when transposeX1 is false.
    - The size of the last dimension of x2 cannot exceed 65535. The last dimension of x2 refers to k when transposeX2 is true or n when transposeX2 is false.
    - x1 supports the INT8 data type.
    - x2 supports the INT8 data type. When x2 is in NZ format, transposeX2 set to false is not supported. When pertokenScaleOptional is not an empty tensor, you must call aclnnTransMatmulWeight to process x2 in ND format to obtain the data layout format affinity with the AI processor.
    - The data type of bias supports INT32. When pertokenScaleOptional is not an empty tensor, the data type of bias also supports FLOAT32.
    - When pertokenScaleOptional is not an empty tensor, the data type of scale supports FLOAT32. When pertokenScaleOptional is an empty tensor, the data type of scale supports UINT64 and INT64.
    - The data type of out supports FLOAT16 and INT8. When pertokenScaleOptional is not an empty tensor, the data type of out only supports FLOAT16.

  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The last dimension size of x1 cannot exceed 65535. However, if the dtypes of x1 and x2 are INT8, the dtype of scale is FLOAT32 or BFLOAT16, and the dimensions of both scale and pertokenScaleOptional are 1, a value greater than 65535 is supported. The last dimension of x1 refers to m when transposeX1 is true, or k when transposeX1 is false.
    - The last dimension size of x2 cannot exceed 65535. However, if the dtypes of x1 and x2 are INT8, the dtype of scale is FLOAT32 or BFLOAT16, and the dimensions of both scale and pertokenScaleOptional are 1, a value greater than 65535 is supported. The last dimension of x2 refers to k when transposeX2 is true, or n when transposeX2 is false.
    - The data type of x1 supports INT8, INT32, and INT4. When the data type is INT32 or INT4, the INT4 quantization scenario applies. Currently, only the 2- to 6-dimensional ND format is supported when transposeX1 is false. When the data type of x1 is INT4, the dimensions are (batch, m, k), and k must be an even number. When the data type of x1 is INT32, each INT32 value stores 8 INT4 values, and the corresponding dimensions are (batch, m, k // 8), where k must be a multiple of 8.
    - The data type of x2 supports INT8, INT32, and INT4. When the data type is INT32 or INT4, the INT4 quantization scenario applies. Currently, only the 2-dimensional ND format is supported.
    - When the data type is INT4, the dimensions in each case are as follows: if transposeX2 is true, the dimensions are (n, k), and k must be an even number; if transposeX2 is false, the dimensions are (k, n), and n must be an even number.
    - When the data type is INT32, each INT32 value stores 8 INT4 values. The dimensions in each case are as follows: if transposeX2 is true, the dimensions are (n, k // 8), and k must be a multiple of 8; if transposeX2 is false, the dimensions are (k, n // 8), and n must be a multiple of 8.
    - You can call the aclnnConvertWeightToINT4Pack API to convert the data format of x2 from INT32 (one int32 stores one int4 in the 0-3 bit positions) to INT32 (one int32 stores 8 int4 values) or INT4 (one int4 represents one int4). See Specifically [aclnnConvertWeightToINT4Pack API](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md).
    - The bias data type supports INT32, BFLOAT16, FLOAT16, and FLOAT32. When x1 and x2 are INT32 or INT4, the shape of bias only supports 1 dimension (n,).
    - When x1 and x2 are INT32 or INT4, transposeX1 only supports false.
    - The out data type supports FLOAT16, INT8, BFLOAT16, and INT32.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The x1 data type supports INT8 and INT4.
    - The x2 data type supports INT8 and INT4.
    - The bias data type supports INT32, BFLOAT16, FLOAT16, and FLOAT32.
    - The out data type supports FLOAT16, INT8, BFLOAT16, and INT32.
    - Under the full quantization scenario, when x2 is in ND format, the output is an empty tensor if m of input x1 is 0 or n of x2 is 0. Under the A8W8 full quantization scenario, when x2 is in NZ format, the output is an empty tensor if m of input x1 is 0 or n of x2 is 0.

  <!-- end id11 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 262px">
    <col style="width: 121px">
    <col style="width: 819px">
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
      <td>x1, x2, scale, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or data format of x1, x2, bias, pertokenScaleOptional, scale, offset, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, bias, pertokenScaleOptional, scale, offset, or out does not satisfy the validation conditions.</td>
    </tr>
    <tr>
      <td>x1, x2, bias, pertokenScaleOptional, scale, offset, or out is an empty tensor and is not one of the empty tensor use cases supported above.</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantMatmulV4

- **Parameters:**
    <table style="table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 153px">
    <col style="width: 121px">
    <col style="width: 880px">
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
        <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnQuantMatmulV4GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, includes the computation flow of the operator.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. See specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="910,310p" id12 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>: aclnnQuantMatmulV4 defaults to a deterministic implementation.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnQuantMatmulV4 defaults to a deterministic implementation.

  <!-- end id13 -->

<!-- npu="A3,910b" id14 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Before calling this API, you can use [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) to process x2 in ND format and obtain an AI processor-affine data layout format.
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950PR&950DT products</term>:
  - In the A4W4 full quantization scenario, only the ND format is supported for x2.
  - In the A8W8 full quantization scenario, x2 supports the ND and NZ formats. Before calling this API, you can convert x2 from the ND format to the NZ format using [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) or [aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md). During the conversion, 0 padding must be used to prevent dirty data from being introduced.
  - In the A8W8 full quantization scenario, if any of the last two dimensions of the original ND-format x2 is 1 (that is, n=1 or k=1), conversion to the NZ format is not supported, and x2 should use the ND format.

<!-- end id15 -->

The input and output support the following data type combinations:

<!-- npu="310p" id16 -->
- <term>Atlas inference products</term>:

  | x1 | x2 | Scale | Offset | Bias | pertokenScaleOptional | Out |
  | ------- | ------- | ------ | ------ | ------- | ------- | ------- |
  | INT8 | INT8 | UINT64/INT64 | null | null/INT32 | null | FLOAT16 |
  | INT8 | INT8 | UINT64/INT64 | null/FLOAT32 | null/INT32 | null | INT8 |
  | INT8 | INT8 | FLOAT32 | null | null/INT32/FLOAT32 | FLOAT32 | FLOAT16 |

<!-- end id16 -->
<!-- npu="A3,910b" id17 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

  | x1 | x2 | Scale | Offset | Bias | pertokenScaleOptional | Out |
  | ------- | ------- | ------ | ------ | ------- | ------- | ------- |
  | INT8 | INT8 | UINT64/INT64 | null | null/INT32 | null | FLOAT16 |
  | INT8 | INT8 | UINT64/INT64 | null/FLOAT32 | null/INT32 | null | INT8 |
  | INT8 | INT8 | FLOAT32/BFLOAT16 | null | null/INT32/BFLOAT16/FLOAT32 | null/FLOAT32 | BFLOAT16 |
  | INT8 | INT8 | FLOAT32 | null | null/INT32/FLOAT16/FLOAT32 | FLOAT32 | FLOAT16 |
  | INT4/INT32 | INT4/INT32 | UINT64/INT64 | null | null/INT32  | null | FLOAT16 |
  | INT8 | INT8 | FLOAT32/BFLOAT16 | null | null/INT32 | null | INT32 |
  | INT4/INT32 | INT4/INT32 | FLOAT32/BFLOAT16 | null | null/INT32/BFLOAT16/FLOAT32 | FLOAT32 | BFLOAT16 |
  | INT4/INT32 | INT4/INT32 | FLOAT32 | null | null/INT32/FLOAT16/FLOAT32 | FLOAT32 | FLOAT16 |

<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950PR&950DT products</term>:

  For the following data type combinations, when pertokenScaleOptional is null, the T-C && T-T [quantization mode](../../../docs/en/context/quant_mode_introduction.md) is supported. When pertokenScaleOptional is not null, the K-C quantization && K-T [quantization mode](../../../docs/en/context/quant_mode_introduction.md) is supported.

  | x1 | x2 | Scale | Offset | Bias | pertokenScaleOptional | Out |
  | ------- | ------- | ------ | ------ | ------- | ------- | ------- |
  | INT8 | INT8 | UINT64/INT64 | null | null/INT32 | null | FLOAT16/BFLOAT16 |
  | INT8 | INT8 | UINT64/INT64 | null/FLOAT32 | null/INT32 | null | INT8 |
  | INT8 | INT8 | FLOAT32/BFLOAT16 | null | null/INT32/FLOAT32/BFLOAT16 | null/FLOAT32 | BFLOAT16 |
  | INT8 | INT8 | FLOAT32 | null | null/INT32/FLOAT32/FLOAT16 | FLOAT32 | FLOAT16 |
  | INT8 | INT8 | FLOAT32/BFLOAT16 | null | null/INT32 | null | INT32 |
  | INT4 | INT4 | UINT64/INT64 | null | null/INT32 | null | FLOAT16 |
  | INT4 | INT4 | FLOAT32/BFLOAT16 | null | null/INT32/FLOAT32/BFLOAT16 | FLOAT32 | BFLOAT16 |
  | INT4 | INT4 | FLOAT32 | null | null/INT32/FLOAT32/FLOAT16 | FLOAT32 | FLOAT16 |

<!-- end id18 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950,A3,910b" id19 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v4.h"

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

      // Call aclCreateTensor to create an aclTensor.
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

  int aclnnQuantMatmulV4Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API.
      std::vector<int64_t> x1Shape = {5, 2};
      std::vector<int64_t> x2Shape = {2, 3};
      std::vector<int64_t> biasShape = {3};
      std::vector<int64_t> offsetShape = {3};
      std::vector<int64_t> pertokenScaleShape = {5};
      std::vector<int64_t> scaleShape = {3};
      std::vector<int64_t> outShape = {5, 3};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *pertokenScale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
      std::vector<int32_t> biasHostData = {1, 1, 1};
      std::vector<float> scaleHostData = {1, 1, 1};
      std::vector<float> offsetHostData = {1, 1, 1};
      std::vector<float> pertokenScaleHostData = {1, 1, 1, 1, 1};
      std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1};  // Actually in float16 half-precision format.
      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the pertokenScale aclTensor.
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &pertokenScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
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

      // 3. Call the CANN operator library API. Change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulV4.
      ret = aclnnQuantMatmulV4GetWorkspaceSize(x1, x2, scale, nullptr, pertokenScale, bias, transposeX1, transposeX2,
                                              out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Request device memory based on the workspaceSize calculated by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV4.
      ret = aclnnQuantMatmulV4(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4 failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait synchronously for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output values and copy the results from device memory to host memory. Modify based on the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // fp16 data cannot be directly printed in C. Read the data as uint16 and convert it to fp16 via binary.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %hu\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. Refer to the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV4Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id19 -->
<!-- npu="A3,910b" id20 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
<!-- end id20 -->

NZ format x2 (transposeX2=false).

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_v4.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
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
      // Boilerplate: resource initialization
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
      // Call aclrtMemcpy to copy data from host memory to device memory.
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

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate device memory.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulV4Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs. The construction needs to be customized based on the API.
      std::vector<int64_t> x1Shape = {5, 2};
      std::vector<int64_t> x2Shape = {2, 3};
      std::vector<int64_t> biasShape = {3};
      std::vector<int64_t> offsetShape = {3};
      std::vector<int64_t> pertokenScaleShape = {5};
      std::vector<int64_t> scaleShape = {3};
      std::vector<int64_t> outShape = {5, 3};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *pertokenScale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
      std::vector<int32_t> biasHostData = {1, 1, 1};
      std::vector<float> scaleHostData = {1, 1, 1};
      std::vector<float> offsetHostData = {1, 1, 1};
      std::vector<float> pertokenScaleHostData = {1, 1, 1, 1, 1};
      std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1};  // Actually float16 half precision
      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor in NZ format.
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create pertokenScale aclTensor.
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &pertokenScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      void *workspaceAddr = nullptr;

      // Call the first-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // For scale of the FLOAT data type, the aclnn API of the TransQuantParamV2 operator must be called in advance.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulV4.
      workspaceSize = 0;
      ret = aclnnQuantMatmulV4GetWorkspaceSize(x1, x2, quantParam, nullptr, nullptr, bias, transposeX1, transposeX2,
                                              out, &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV4(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV4.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV4.
      ret = aclnnQuantMatmulV4(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // fp16 data cannot be printed directly in C. Read it as uint16 and convert the binary to fp16 manually.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %hu\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
      // Fill in deviceId based on your actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV4Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

NZ format x2 (transposeX2=true).

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_v4.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
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

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to apply for device memory.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulV4Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs. Custom construction is required based on the API.
      std::vector<int64_t> x1Shape = {5, 2};
      std::vector<int64_t> x2Shape = {2, 3};
      std::vector<int64_t> x2TransposedShape = {3, 2};
      std::vector<int64_t> biasShape = {3};
      std::vector<int64_t> offsetShape = {3};
      std::vector<int64_t> pertokenScaleShape = {5};
      std::vector<int64_t> scaleShape = {3};
      std::vector<int64_t> outShape = {5, 3};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2TransposedDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *x2Transposed = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *pertokenScale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
      std::vector<int8_t> x2TransposedHostData = {1, 1, 1, 1, 1, 1};
      std::vector<int32_t> biasHostData = {1, 1, 1};
      std::vector<float> scaleHostData = {1, 1, 1};
      std::vector<float> offsetHostData = {1, 1, 1};
      std::vector<float> pertokenScaleHostData = {1, 1, 1, 1, 1};
      std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1};  // Actually float16 half-precision mode.
      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor in NZ format.
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Transposed aclTensor in NZ format.
      ret = CreateAclTensorX2(x2TransposedHostData, x2TransposedShape, &x2TransposedDeviceAddr,
                              aclDataType::ACL_INT8, &x2Transposed);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TransposedHPTensorPtr(x2Transposed,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2TransposedHPDeviceAddrPtr(x2TransposedDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the pertokenScale aclTensor.
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &pertokenScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = true;

      // 3. Call the CANN operator library API. Change it to the actual API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      void *workspaceAddr = nullptr;

      // The shape of x2 needs to be transposed to the nk format before performing transdata.
      std::vector<int64_t> dimsData = {1, 0};
      // Create the dims aclIntArray.
      aclIntArray *dims = aclCreateIntArray(dimsData.data(), dimsData.size());
      // Call the first-phase API of aclnnPermute.
      ret = aclnnPermuteGetWorkspaceSize(x2, dims, x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrPermute(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrPermute.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnPermute.
      ret = aclnnPermute(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      workspaceSize = 0;
      // Call the first-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // For scale with data type FLOAT, call the aclnn API of the TransQuantParamV2 operator in advance.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulV4.
      workspaceSize = 0;
      ret = aclnnQuantMatmulV4GetWorkspaceSize(x1, x2Transposed, quantParam, nullptr, nullptr, bias, transposeX1,
                                              transposeX2, out, &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV4(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV4.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV4.
      ret = aclnnQuantMatmulV4(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Wait synchronously for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result on device memory to the host. This code needs to be modified according to the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // fp16 data cannot be printed directly in C. Read it out as uint16 and manually convert it to fp16 in binary.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %hu\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual
      // Set deviceId based on the actual device
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV4Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- npu="A3,910b" id21 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
<!-- end id21 -->

INT4 quantization scenario (data types of x1 and x2 are INT4, transposeX2=false).

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_convert_weight_to_int4_pack.h"
  #include "aclnnop/aclnn_quant_matmul_v4.h"
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
      // (Boilerplate) Initialize resources
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
      // Obtain the number of memory bytes to allocate and copy through hostData
      auto size = hostData.size() * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory
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

  int aclnnQuantMatmulV4Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs. Customize the construction based on the API.
      int64_t m = 16;
      int64_t k = 8;
      int64_t n = 32;
      aclDataType x1Dtype = aclDataType::ACL_INT4;
      aclDataType x2Int4PackDtype = aclDataType::ACL_INT4;
      std::vector<int64_t> x1Shape = {m, k};
      std::vector<int64_t> x2Shape = {k, n};
      std::vector<int64_t> x2Int4PackShape = {k, n};
      std::vector<int64_t> biasShape = {n};
      std::vector<int64_t> scaleShape = {n};
      std::vector<int64_t> outShape = {m, n};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2Int4PackDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *x2Int4Pack = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *pertokenScale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(m * k / 2, 17);  // int8: 0001 0001
      std::vector<int8_t> x2HostData(k * n, 1);
      std::vector<int8_t> x2Int4PackHostData(n * k / 2, 1);
      std::vector<int32_t> biasHostData(n, 1);
      std::vector<float> scaleHostData(n, 1);
      std::vector<uint16_t> outHostData(m * n, 1);

      // Create x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, x1Dtype, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create x2Int4Pack aclTensor.
      ret =
          CreateAclTensor(x2Int4PackHostData, x2Int4PackShape, &x2Int4PackDeviceAddr, x2Int4PackDtype, &x2Int4Pack);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2Int4PackTensorPtr(x2Int4Pack,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2Int4PackDeviceAddrPtr(x2Int4PackDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API. Replace with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      // You can call the aclnnConvertWeightToINT4Pack API first to construct the x2 input data.
      // Call the first-phase API of aclnnConvertWeightToINT4Pack.
      ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(x2, x2Int4Pack, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceINT4PackAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceINT4PackAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnConvertWeightToINT4Pack.
      ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

      // For scale of the FLOAT data type, call the aclnn API of the TransQuantParamV2 operator in advance.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulV4.
      ret = aclnnQuantMatmulV4GetWorkspaceSize(x1, x2Int4Pack, quantParam, nullptr, pertokenScale, bias, transposeX1,
                                              transposeX2, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV3(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV3.reset(workspaceAddr);
      }

      // Call the second-phase API of aclnnQuantMatmulV4.
      ret = aclnnQuantMatmulV4(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4 failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, copy the result from the device memory to the host memory, and modify it based on the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // fp16 data cannot be printed directly in C. Read it as uint16 and manually convert the binary to fp16.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %hu\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream by referring to the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV4Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV4Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
