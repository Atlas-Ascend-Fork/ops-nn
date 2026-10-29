# aclnnAddRmsNormDynamicMxQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:46:15.990Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description: This API provides the same function as `aclnnAddRmsNormDynamicMxQuant`, with the optional input x3Optional added to support three-way residual addition. When x3Optional is a null pointer, the V2 API behaves exactly the same as `aclnnAddRmsNormDynamicMxQuant`. This API fuses the Add operator, RmsNorm normalization (which removes the mean subtraction part compared with the LayerNorm operator), and DynamicMxQuant dynamic MX quantization into a single computation, reducing data movement operations: it first adds the inputs, then performs RmsNorm normalization, and finally performs dynamic MX quantization in groups of blocksize=32 along the trailing axis of the input, outputting the quantization result yOut, addition result xOut, quantization scale mxscaleOut, and reciprocal of the standard deviation rstdOut. The algorithm for computing mxscaleOut is specified by scaleAlg (both OCP and cuBLAS implementations are supported when dstType is FP8, and only OCP is supported when dstType is FP4). For the detailed computation process, see the formula below.
- Formula:

  If x3Optional is null:
  $$
  x=x_{1}+x_{2}
  $$

  When x3Optional is not empty (additions are performed in the order within brackets, matching residual addition semantics):
  $$
  x=(x_{3}+x_{1})+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  When scaleAlg is 0:

  - Group the RmsNorm output y along the last axis into groups of k = 32 values, and each group of k values $\{V_i\}_{i=1}^{k}$ is dynamically quantized to $\{mxscale,\{P_i\}_{i=1}^{k}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, roundMode), \quad i = 1, 2, \ldots, k
    $$

  - emax: Exponent bits of the largest normal number in the corresponding data type.

    | DataType | Emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

  When scaleAlg is 1, only FP8 data types are involved:
  - Divide the long vector into blocks, each of length k. Compute a block scaling factor $S_{fp32}^b$ for each block, and map all elements in the block to the target low-precision data type FP8 using the same $S_{fp32}^b$.
  - Find the maximum absolute value of the values in the block:

    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$

  - Map FP32 to the range representable by the target data type FP8:

    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$

  - Convert to the scaling value $S_{ue8m0}^b$ representable in FP8 format
  - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block floating-point scale factor $S_{fp32}^b$.
  - Round up the exponent to prevent overflow during quantization:

    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a subnormal number, and} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{ otherwise} \end{cases}
    $$

  - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
  - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
  - Final step applied to quantization: $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize` first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call `aclnnAddRmsNormDynamicMxQuantV2` to perform the computation.

```Cpp
aclnnStatus aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *x3Optional,
  double           epsilon,
  int64_t          scaleAlg,
  char            *roundMode,
  int64_t          dstType,
  bool            outputRstd,
  aclTensor       *yOut,
  aclTensor       *xOut,
  aclTensor       *mxscaleOut,
  aclTensor       *rstdOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormDynamicMxQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
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
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the source data tensor in the normalization process. Corresponds to x1 in the formula.</td>
      <td><ul><li>Supports Null Tensor.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the source data tensor in the normalization process. Corresponds to x2 in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape and data type must keep consistent with x1.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the weight tensor in the normalization process. Corresponds to gamma in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape must be one-dimensional and match the last dimension of x1.</li><li>The data type defaults to that of input x1; if they are different, it is explicitly set to FLOAT32.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the bias term in the normalization process. It corresponds to beta in the formula.</td>
      <td><ul><li>Support Null Tensor.</li><li>Optional parameter. Supports passing in null pointer.</li><li>If provided, the shape and data type must keep consistent with gamma.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x3Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the optional input in the addition computation and the residual path. Corresponding Formula: x3Optional.</td>
      <td><ul><li>Optional parameter. Supports passing in a null pointer.</li><li>If a null pointer is passed in, the V2 API behaves exactly the same as aclnnAddRmsNormDynamicMxQuant.</li><li>If not null, the computation formula becomes x = (x3Optional + x1) + x2, and the shape and data type must keep consistent with those of x1.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Indicates the value added to the denominator to ensure numerical stability. It is the epsilon in the corresponding formula.</td>
      <td><ul><li>A small positive value, such as 1e-6, is recommended.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Indicates the calculation method of mxscaleOut. It is the scaleAlg in the corresponding formula.</td>
      <td><ul><li>Value is 0 or 1. Value 0 indicates the Open Compute Project (OCP) implementation, and value 1 indicates the cuBLAS implementation.</li><li>Only value 0 is supported when dstType is FLOAT4_E2M1/FLOAT4_E1M2.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>Input</td>
      <td>Indicates the data conversion mode, corresponding to roundMode in the formula.</td>
      <td><ul><li>Does not support passing in a null pointer. A supported string must be explicitly passed in.</li><li>Only {"rint"} is supported when dstType is 35/36.</li><li>{"rint", "floor", "round"} are supported when dstType is 40/41.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the type of yOut after data conversion, corresponding to DType in the formula.</td>
      <td><ul><li>The input range is {35, 36, 40, 41}, corresponding to FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and FLOAT4_E1M2 respectively.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputRstd (bool)</td>
      <td>Input</td>
      <td>Indicates whether to output the valid rstdOut.</td>
      <td><ul><li>Supports True and False.</li><li>When outputRstd is False, rstdOut is an invalid output.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the normalized and quantized result, corresponding to Pi and di in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape must be the same as that of input x1.</li><li>The data type must be kept consistent with dstType.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the sum of x3Optional+x1+x2 (the sum of x1+x2 when x3Optional is null), corresponding to x in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape and data type must be the same as those of input x1.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale corresponding to each group, that is, mxscale and Sb in the corresponding formula.</td>
      <td><ul><li>Supports null tensors.</li><li>On the tail axis, the shape is the corresponding value of x divided by blocksize=32 and rounded up, followed by even padding. The pad fill value is 0.</li><li>The remaining dimensions are kept consistent with the first several dimensions of the shape of input x1. The first several dimensions refer to the dimensions of x1 minus the dimensions of gamma, indicating the dimensions that do not require norm.</li><li>For details about the calculation process, see Constraints.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the reciprocal of the standard deviation after normalization. Corresponding Formula: the reciprocal of Rms(x).</td>
      <td><ul><li>Supports Null Tensor.</li><li>When outputRstd is True, the number of dimensions keeps consistent with that of input x1, all dimensions except the last one keep consistent with the corresponding dimensions of x1, and the last dimension is 1. Example of the relationship between the rstdOut shape and the x1 shape: if the x1 shape is (2,3,4,8), the rstdOut shape is (2,3,4,1).</li><li>When outputRstd is False, rstdOut is an invalid output.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>The passed-in x1, x2, gamma, yOut, xOut, and mxscaleOut are null pointers.</td>
    </tr>
    <tr>
      <td>When outputRstd is True, the passed-in rstdOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="9">561103</td>
      <td>The number of dimensions and data type of the input or output parameters are out of range, or the data types of dstType and yOut do not match.</td>
    </tr>
    <tr>
      <td>The shapes of inputs x1 and x2 and outputs yOut and xOut are not exactly the same.</td>
    </tr>
    <tr>
      <td>When x3Optional is not a null pointer, the shape or data type of x3Optional is inconsistent with that of x1.</td>
    </tr>
    <tr>
      <td>The shapes of gamma and beta (if present) are not exactly the same, or the number of their dimensions is not 1, or the axis length is not equal to the size of the trailing axis of x1, or their types are different.</td>
    </tr>
    <tr>
      <td>The dimensions of gamma are different from the dimensions of x1 to be normalized, or the dimensions of rstdOut are different from the dimensions of x1 not to be normalized, or the axis length of rstdOut corresponding to the normalized dimension of x1 is not 1.</td>
    </tr>
    <tr>
      <td>scaleAlg is not 0 or 1, or roundMode is a null pointer, or roundMode is not one of {"rint", "floor", "round"}.</td>
    </tr>
    <tr>
      <td>When dstType is FP8, roundMode is not "rint".</td>
    </tr>
    <tr>
      <td>When dstType is fp4, scaleAlg is not 0, or the trailing axis of input x1 is not divisible by 2.</td>
    </tr>
    <tr>
      <td>The number of dimensions of mxscaleOut is not equal to the number of dimensions of input x1 plus 1, or the axis length does not comply with the Constraints.</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormDynamicMxQuantV2

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>The workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size allocated on the Device, obtained by calling the first-phase API aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Boundary Value Scenario Description**
  - If the input is Inf: 1. The output yOut is 0; 2. The output xOut is Inf; 3. The output mxscaleOut is 255, and the fill value of even pads is 0; 4. The output rstdOut is 0.
  - If the input is NaN: 1. The output yOut is 0; 2. The output xOut is Nan; 3. The output mxscaleOut is 255, and the fill value of even pads is 0; 4. The output rstdOut is NaN.
  - The preceding boundary value rules apply to x3Optional, x1, and x2.

- **Data Format Description**

  It is recommended to use the ND format for all input and output tensors. Other data formats are converted to the ND format by the framework by default for processing.

- **Supported Data Types by Product Model**

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    | `x1` Data Type | `x2` Data Type | `gamma` Data Type | `beta` Data Type | `x3Optional` Data Type | `yOut` Data Type | `xOut` Data Type | `mxscaleOut` Data Type | `rstdOut` Data Type |
    | ----------- | ----------- | -------------- | ------------- | ----------- | -------------- | ------------- | ------------------- | ---------------- |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT16     | FLOAT4_E2M1    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT16     | FLOAT4_E1M2    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT16     | FLOAT8_E4M3FN  | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT16     | FLOAT8_E5M2    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16      | BFLOAT16    | FLOAT4_E2M1    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16      | BFLOAT16    | FLOAT4_E1M2    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16      | BFLOAT16    | FLOAT8_E4M3FN  | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16      | BFLOAT16    | FLOAT8_E5M2    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT16     | FLOAT4_E2M1    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT16     | FLOAT4_E1M2    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT16     | FLOAT8_E4M3FN  | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT16     | FLOAT8_E5M2    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | FLOAT32        | FLOAT32       | BFLOAT16    | FLOAT4_E2M1    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | FLOAT32        | FLOAT32       | BFLOAT16    | FLOAT4_E1M2    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | FLOAT32        | FLOAT32       | BFLOAT16    | FLOAT8_E4M3FN  | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | FLOAT32        | FLOAT32       | BFLOAT16    | FLOAT8_E5M2    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |

    Description: The data type of x3Optional is kept consistent with x1; if a null pointer is passed in for x3Optional, the x3Optional column in the preceding table does not take effect.

  <!-- end id7 -->

- The shape constraints of mxscaleOut are as follows:
  - rank(mxscaleOut) = rank(x1) + 1.
  - mxscaleOut.shape[-2] = ceil(ceil(x1.shape[-1] / 32) / 2), which indicates that the number of blocks after grouping the last axis of x1 by blocksize=32 is rounded up to an even number and then divided by 2.
  - mxscaleOut.shape[-1] = 2.
  - The other dimensions are the same as those of input x1.

- The shape constraints of x1 are as follows:
  - If the data type of output yOut is FLOAT4_E2M1 or FLOAT4_E1M2, the value of the last axis of x1 must be an even number.

- Constraints on x3Optional are as follows:
  - x3Optional is an optional input and supports passing in a null pointer. When a null pointer is passed in, the V2 API is equivalent to `aclnnAddRmsNormDynamicMxQuant`.
  - If x3Optional is not null, the shape and data type of x3Optional must be exactly the same as those of x1.
  - x3Optional supports non-contiguous tensors.

- Deterministic computation:
  - aclnnAddRmsNormDynamicMxQuantV2 defaults to a deterministic implementation.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_dynamic_mx_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host to the device memory.
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
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs. The construction is custom based on the API.
    std::vector<int64_t> xShape = {2, 64};
    std::vector<int64_t> gammaShape = {64};
    std::vector<int64_t> mxscaleShape = {2, 1, 2};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x3OptionalDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;

    void* yDeviceAddr = nullptr;
    void* mxscaleDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x3Optional = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* y = nullptr;
    aclTensor* mxscale = nullptr;
    aclTensor* x = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t gammaShapeSize = GetShapeSize(gammaShape);
    int64_t mxscaleShapeSize = GetShapeSize(mxscaleShape);

    std::vector<short> x1HostData(xShapeSize, 0x3800);
    std::vector<short> x2HostData(xShapeSize, 0x3800);
    std::vector<short> x3OptionalHostData(xShapeSize, 0x3c00);
    std::vector<short> gammaHostData(gammaShapeSize, 0x3e00);

    std::vector<int8_t> yHostData(xShapeSize, 0);
    std::vector<int8_t> mxscaleHostData(mxscaleShapeSize, 0);
    std::vector<short> xHostData(xShapeSize, 0);

    float epsilon = 1e-6;

    // Create the x1 aclTensor.
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x2 aclTensor.
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x3Optional aclTensor (optional input. A non-null value is passed here to demonstrate the V2 x3Optional path.)
    ret = CreateAclTensor(x3OptionalHostData, xShape, &x3OptionalDeviceAddr, aclDataType::ACL_FLOAT16, &x3Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the y aclTensor (FLOAT8_E4M3FN).
    ret = CreateAclTensor(yHostData, xShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mxscale aclTensor (FLOAT8_E8M0).
    ret = CreateAclTensor(mxscaleHostData, mxscaleShape, &mxscaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAddRmsNormDynamicMxQuantV2.
    // Passing nullptr to beta indicates that no bias is used; passing a non-null pointer to x3Optional enables the V2 x = (x3Optional+x1)+x2 path.
    // Pass "rint" to roundMode (FP8 output supports only "rint"); pass nullptr to rstdOut (can be omitted when outputRstd=false).
    ret = aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize(
        x1, x2, gamma, nullptr, x3Optional, epsilon, 0, const_cast<char*>("rint"), 36, false, y, x, mxscale, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddRmsNormDynamicMxQuantV2.
    ret = aclnnAddRmsNormDynamicMxQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnAddRmsNormDynamicMxQuantV2 failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, and copy the result from the device memory to the host.
    auto size = GetShapeSize(xShape);
    std::vector<int8_t> yRet(size, 0);
    ret = aclrtMemcpy(
        yRet.data(), yRet.size() * sizeof(yRet[0]), yDeviceAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, yRet[i]);
    }

    // 6. Release the aclTensor.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(x3Optional);
    aclDestroyTensor(gamma);
    aclDestroyTensor(y);
    aclDestroyTensor(mxscale);
    aclDestroyTensor(x);

    // 7. Release device resources.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(x3OptionalDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(mxscaleDeviceAddr);
    aclrtFree(xDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
