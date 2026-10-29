# aclnnSwigluMxQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:45:57.653Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Adds the DynamicMxQuant operation after the Swish gated linear unit activation function to implement the SwigluMxQuant computation of x.
- Formula when swigluMode is 0:

  $$
  swigluOut_i = Swiglu(x_i)=Swish(A_i)*B_i
  $$

  $$
  y, mxscale = DynamicMxQuant(swigluOut_i)
  $$

  where A<sub>i</sub> denotes the first half of x<sub>i</sub>, and B<sub>i</sub> denotes the second half of x<sub>i</sub>.

- Formula when swigluMode is 1:

  $$
  x\_glu = x\_glu.clamp(min=None, max=clampLimit)
  $$

  $$
  x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + gluBias)
  $$

  $$
  y, mxscale = DynamicMxQuant(swigluOut_i)
  $$

  where x\_glu denotes the even-indexed part of x<sub>i</sub>, and x\_linear denotes the odd-indexed part of x<sub>i</sub>.

- Formula when swigluMode is 2:

  $$
  x\_glu = x\_glu.clamp(min=None, max=clampLimit)
  $$

  $$
  x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + gluBias)
  $$

  $$
  y, mxscale = DynamicMxQuant(swigluOut_i)
  $$

  Where x\_glu and x\_linear are obtained by splitting x<sub>i</sub> into the first half and the second half along the activateDim axis (non-interleaved splitting). When activateLeft is true, x\_glu is the first half; otherwise, it is the second half. In this mode, axis must be -1.

- Formula when swigluMode is 3:

  $$
  x\_glu = x\_glu * sigmoid(x\_glu)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clampLimit)
  $$

  $$
  x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)
  $$

  $$
  swigluOut_i = x\_glu * x\_linear
  $$

  $$
  y, mxscale = DynamicMxQuant(swigluOut_i)
  $$

  where x\_glu and x\_linear are obtained by splitting x<sub>i</sub> along the activateDim axis into the first half and the second half (non-interleaved chunking). In this mode, the gluAlpha and gluBias parameters are ignored (sigmoid uses a fixed alpha=1 and no bias is added), and axis must be -1.

- Use case 1: when scaleAlg is 0, the DynamicMxQuant computation logic:
    - The input x is grouped along the axis dimension into groups of k = blocksize = 32 numbers. A group of k numbers  $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized into $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$

    - ​The quantized $P_{i}$ are placed at the positions of the corresponding $V_{i}$ to form the output yOut, and mxscale forms the output mxscaleOut according to the groups along the axis dimension.

    - emax: The exponent of the maximum normal number of the corresponding data type.

        | DataType | Emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

- Use case 2: when scaleAlg is 1 (only FP8 types are involved), the DynamicMxQuant computation logic:
  - Divide the long vector into blocks, each with a length of k. Compute a block scale factor $S_{fp32}^b$ for each block individually, and then map all elements within the block to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block has fewer than k elements, treat the missing values as 0 and process it as a complete block.
  - Find the maximum absolute value of the numbers in the block:
    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$
  - Map FP32 to the representable range of the target data type FP8, where $Amax(DType)$ is the maximum value that the target precision can represent.
    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$
  - Convert the block scale factor $S_{fp32}^b$ to the representable scale value $S_{ue8m0}^b$ in FP8 format.
  - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block floating-point scale factor $S_{fp32}^b$.
  - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:
    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a subnormal number and} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{ otherwise} \end{cases}
    $$
  - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
  - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
  - The final step applied to quantization: for each element in a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ denotes the scale factor of the block, which is $S_{ue8m0}^b$ here, and $[d^i]_{i=1}^k$ denotes the quantized data in the block.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSwigluMxQuantGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnSwigluMxQuant" API to perform the computation.

```Cpp
aclnnStatus aclnnSwigluMxQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupIndexOptional,
  int64_t          activateDim,
  bool             activateLeft,
  int64_t          swigluMode,
  double           clampLimit,
  double           gluAlpha,
  double           gluBias,
  int64_t          groupMode,
  int64_t          axis,
  int64_t          dstType,
  char            *roundModeOptional,
  int64_t          scaleAlg,
  double           maxDtypeValue,
  const aclTensor *yOut,
  const aclTensor *mxscaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwigluMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluMxQuantGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 177px">
  <col style="width: 104px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input data to be processed, corresponding to x in the formula.</td>
      <td><ul><li>The shape is [X1,X2,...Xn,2H], with at most 7 dimensions and at least 2 dimensions.</li><li>The dimension of input x corresponding to activateDim must be a multiple of 2.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
      <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>group_index required for MoE grouping.</td>
      <td><ul><li>shape supports a 1-dimensional tensor with shape [groupNum], where groupNum is greater than or equal to 1 and less than or equal to 256.</li><li>Each value in data must be an integer greater than or equal to 0, and the sum of all values must be less than or equal to the total number of rows in x to be quantized.</li><li>Optional parameter. A null pointer is supported.</li><li>When this input exists and either activateDim or axis is a non-tail axis, the shape of input x must be 2-dimensional.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>activateDim (int64_t)</td>
      <td>Input</td>
      <td>Specifies the split axis for the swish computation.</td>
      <td><ul><li>The value range of activateDim is [-1, -2, xDim - 2, xDim - 1] (where xDim is the number of dimensions of input x).</li><li>If activateDim is not the last axis, swigluMode must be 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Indicates whether to perform swiglu activation on the left half of the input.</td>
      <td><ul><li>If the value is false, activation is performed on the right half of the input. If swigluMode is 1, activation is performed on the even-indexed parts of x (this parameter does not take effect).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>swigluMode (int64_t)</td>
      <td>Input</td>
      <td>Specifies the calculation mode of swiglu.</td>
      <td><ul><li>Value range: [0, 3].</li><li>0: the traditional swiglu calculation method.</li><li>1: a variant of swiglu. It uses odd-even chunking and supports clampLimit, the activation coefficient, and bias.</li><li>2: a variant of swiglu. It uses first-half/second-half splitting, applies clamp first and then sigmoid(alpha) activation, and supports bias. This mode requires axis=-1.</li><li>3: a variant of swiglu. It uses first-half/second-half splitting, applies sigmoid activation first (alpha is fixed to 1) and then clamp, and does not support bias. This mode requires axis=-1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>clampLimit (double)</td>
      <td>Input</td>
      <td>Indicates the threshold value used by the swiglu variant.</td>
      <td><ul><li>This parameter is an optional parameter.</li><li>Used to clip the input. The value must be greater than 0 and less than infinity to prevent excessively large values from affecting the stability of the swiglu computation.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gluAlpha (double)</td>
      <td>Input</td>
      <td>Indicates the scaling parameter used by the swiglu variant.</td>
      <td><ul><li>This parameter is an optional parameter.</li><li>Adjusts the scaling of the linear part in the glu activation function.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gluBias (double)</td>
      <td>Input</td>
      <td>Bias parameter used by the swiglu variant.</td>
      <td><ul><li>This parameter is an optional parameter.</li><li>Adds an offset in the linear computation of swiglu.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>groupMode (int64_t)</td>
      <td>Input</td>
      <td>Specifies the mode corresponding to groupIndexOptional.</td>
      <td><ul><li>Value range: [0, 1].</li><li>0: count mode.</li><li>1: cumsum mode.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Indicates the axis on which DynamicMxQuant quantization is performed.</td>
      <td><ul><li>Value range: [-1, -2, xDim - 2, xDim - 1] (where xDim indicates the number of dimensions of input x).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the data type of output y.</td>
      <td><ul><li>The value range of dstType is [35, 36, 40, 41], corresponding to FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and FLOAT4_E1M2 respectively.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Specifies the rounding mode for output y.</td>
      <td><ul><li>The value range is: ["rint", "round", "floor"].</li><li>When the data type of output y is FLOAT8_E5M2 or FLOAT8_E4M3FN, only the "rint" mode is supported.</li><li>A null pointer can be passed in. In this case, "rint" is used.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Specifies the calculation method of mxscaleOut, corresponding to scaleAlg in the formula.</td>
      <td><ul><li>The value range of scaleAlg is [0, 1, 2]. A value of 0 indicates use case 1, and a value of 1 indicates use case 2.</li><li>When dstType is FLOAT4_E2M1/FLOAT4_E1M2, only the value 0 is supported.</li><li>Currently, only 0 and 1 are supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>maxDtypeValue (double)</td>
      <td>Input</td>
      <td>Indicates the maximum value of the target data type specified during DynamicMxQuant.</td>
      <td><ul><li>The value of maxDtypeValue must be greater than or equal to 0. This parameter is reserved and does not take effect in the current version.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>yOut(aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantized result of input x, corresponding to Pi and di in the formula.</td>
      <td><ul><li>When activateDim corresponds to the last axis of x, the shape is [X1,X2,...Xn,H].</li><li>When activateDim does not correspond to the last axis of x, the shape is [X1,X2,...,XactivateDim / 2,...,2H].</li><li>When the data type of yOut is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of yOut must be a multiple of 2.</li></ul></td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscaleOut(aclTensor*)</td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group, corresponding to mxscale and Sb in the formula.</td>
      <td><ul><li>shape on the axis dimension is the value of the corresponding axis of yOut divided by blocksize=32 and rounded up, and is padded with even padding. The pad value is 0.</li><li>When axis is a non-tail axis and groupIndexOptional exists, shape on the axis dimension is the value of the corresponding axis of yOut divided by 64 and rounded down, plus groupNum.</li><li>When axis is a non-tail axis, the mxscaleOut output needs to be interleaved for every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-8</td>
      <td>√</td>
    </tr>
       <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, yOut, or mxscaleOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type of the input or output is not in the supported range.</td>
    </tr>
    <tr>
      <td>The dimensions of the input or output parameters are not in the supported range.</td>
    </tr>
    <tr>
      <td>activateDim is not in the specified value range.</td>
    </tr>
    <tr>
      <td>swigluMode is not in the specified value range.</td>
    </tr>
    <tr>
      <td>clampLimit is not in the specified value range.</td>
    </tr>
    <tr>
      <td>axis is not in the specified value range.</td>
    </tr>
    <tr>
      <td>dstType is not in the specified value range.</td>
    </tr>
    <tr>
      <td>roundModeOptional is not in the specified value range.</td>
    </tr>
    <tr>
      <td>scaleAlg is not in the specified value range.</td>
    </tr>
    <tr>
      <td>axis is not -1 when swigluMode is 2 or 3.</td>
    </tr>
  </tbody></table>

## aclnnSwigluMxQuant

- **Parameters:**
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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnSwigluMxQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
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

- Deterministic computation:
  - aclnnSwigluMxQuant defaults to a deterministic implementation.

- The dimension of input x corresponding to activateDim must be a multiple of 2, and the number of dimensions of x must be greater than 1.
- If activateDim is not the last axis, swigluMode must be 0.
- If swigluMode is 2 or 3, axis must be -1 (only last-axis quantization is supported).
- When the data type of output yOut is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of yOut must be a multiple of 2.
- If the data type of output yOut is FLOAT4_E2M1 or FLOAT4_E1M2, scaleAlg must be 0.
- Each element of groupIndexOptional must be an integer greater than or equal to 0, and the sum of all elements must not exceed the total number of rows of x to be quantized (that is, the product of the remaining axes of input x excluding the last axis).
- The portion of outputs yOut and mxscaleOut beyond the sum of all elements of groupIndexOptional is not cleared. This portion of memory contains garbage data.
- If activateDim is not the last axis, or axis is not the last axis, and groupIndexOptional is provided, input x must be 2-dimensional.
- If the data type of output yOut is FLOAT8_E4M3FN or FLOAT8_E5M2, roundModeOptional must be rint.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_mx_quant.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to request device memory.
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> xShape = {2, 64};
  std::vector<int64_t> groupIndexShape = {1};
  std::vector<int64_t> outShape = {2, 32};
  std::vector<int64_t> scaleOutShape = {2, 1, 2};

  void* xDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;

  std::vector<uint16_t> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<uint8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> scaleOutHostData = {0, 0, 0, 0};

  bool activateLeft = true;
  int64_t dstType = 36;
  int64_t activateDim = -1;
  int64_t swigluMode = 1;
  float clampLimit = 7.0;
  float gluAlpha = 1.0;
  float gluBias = 1.702;
  int64_t groupMode = 0;
  int64_t axis = -1;
  int64_t scaleAlg = 0;
  float maxDtypeValue = 0.0;

  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the scaleOut aclTensor.
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSwigluMxQuant.
  ret = aclnnSwigluMxQuantGetWorkspaceSize(x, nullptr, activateDim, activateLeft, swigluMode, clampLimit, gluAlpha, gluBias,groupMode, axis, dstType, "rint", scaleAlg, maxDtypeValue, out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSwigluMxQuant.
  ret = aclnnSwigluMxQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<uint8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. Release device resources. Modify it according to the specific API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
