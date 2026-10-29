# aclnnDynamicMxQuantWithDualAxisV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:27:10.607Z -->

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

- Description: Simultaneously performs MX quantization to the FLOAT4 and FLOAT8 destination data types on the -1 axis and -2 axis. On the given -1 axis and -2 axis, for every 32 values, computes the quantization scaling factors mxscale1 and mxscale2 of the two groups of values as the corresponding parts of outputs mxscale1Out and mxscale2Out. Then divides all elements of the two groups of values by the corresponding mxscale1 or mxscale2, converts them to the corresponding dstType according to round_mode, and obtains the quantization results y1 and y2 as the corresponding parts of outputs y1Out and y2Out, respectively. When dstType is FLOAT8_E4M3FN or FLOAT8_E5M2, the value of scaleAlg specifies the algorithm for computing mxscale.

- Axis merging: In the operator implementation, all axes before the -2 axis (exclusive) are merged. That is, for a tensor with the input shape $(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})$, the dimensions $(d_0, d_1, ..., d_{n-3})$ before the -2 axis are merged into one dimension. This is equivalent to reshaping the input to $(d_0 \times d_1 \times ... \times d_{n-3}, d_{n-2}, d_{n-1})$ before performing the quantization.

- Formula:
  - Use case 1: When scaleAlg is 0, that is, the OCP Microscaling Formats (Mx) Specification implementation:
    - Group the input x along the -1 axis by 32 elements, where a group of 32 elements $\{\{V_i\}_{i=1}^{32}\}$ is quantized into $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax
      $$

      $$
      mxscale1 = 2^{shared\_exp}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
      $$

    - Simultaneously, group the input x along the -2 axis by 32 elements, where a group of 32 elements $\{\{V_j\}_{j=1}^{32}\}$ is quantized into $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

      $$
      shared\_exp = floor(log_2(max_j(|V_j|))) - emax
      $$

      $$
      mxscale2 = 2^{shared\_exp}
      $$

      $$
      P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
      $$

    - The quantized $P_{i}$ on the -1 axis is composed into output y1Out according to the corresponding positions of $V_{i}$, and mxscale1 is composed into output mxscale1Out according to the grouping on the -1 axis dimension. The quantized $P_{j}$ on the -2 axis is composed into output y2Out according to the corresponding positions of $V_{j}$, and mxscale2 is composed into output mxscale2Out according to the grouping on the -2 axis dimension.

    - emax: The exponent of the maximum normal number of the corresponding data type.

      | DataType | Emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |
      |  FLOAT4_E1M2  |  0   |
      | FLOAT8_E4M3FN |  8   |
      |  FLOAT8_E5M2  |  15  |

  - Use case 2: When scaleAlg is 1, only FP8 types are involved (round-up algorithm):
    - **-1 axis quantization**: Input x is grouped by count 32 along the -1 axis, and each group has a length of 32. A block scaling factor $S_{fp32}^b$ is calculated for each group separately, and all elements in the group are then mapped to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last group has fewer than 32 elements, the missing values are treated as 0 and the group is processed as a complete group.
      - Find the maximum absolute value in the group:
        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{32})
        $$
      - Map FP32 to the representable range of the target data type FP8, where $Amax(DType)$ is the maximum value that the target precision can represent.
        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$
      - Convert the block scale factor $S_{fp32}^b$ to the representable scale value $S_{ue8m0}^b$ in FP8 format.
      - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block floating-point scale factor $S_{fp32}^b$.
      - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:
        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a subnormal number, and} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{ otherwise} \end{cases}
        $$
      - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
      - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - Apply the final quantization step. For each element in the group, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result of the -1 axis output is $\left(S^b, [d^i]_{i=1}^{32}\right)$, where $S^b$ represents the scaling factor of the block, i.e., $S_{ue8m0}^b$, and $[d^i]_{i=1}^{32}$ represents the quantized data in the group.
    - **-2 axis quantization**: Simultaneously, group the input x along the -2 axis by 32, adopt the same round-up algorithm as the -1 axis, and independently compute the block scaling factor and quantize each group. The quantization result of the -2 axis output is $\left(S^b, [d^j]_{j=1}^{32}\right)$.
    - The -1 axis quantization result composes output y1Out, and the corresponding block scaling factors compose output mxscale1Out. The -2 axis quantization result composes output y2Out, and the corresponding block scaling factors compose output mxscale2Out.

  - Scenario 3: when scaleAlg is 2, only the FP4_E2M1 type is involved:
    - When dstTypeMax = 0.0/6.0/7.0:
      - **-1 axis quantization**: Group the input x along the -1 axis by 32. A group of 32 numbers $\{\{V_i\}_{i=1}^{32}\}$ is quantized to $\{mxscale1, \{P_i\}_{i=1}^{32}\}$:
        $$
        shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{ if} the one/two bits before the high bit of the mantissa \text{ are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{ otherwise} \end{cases} \\
        $$
        $$
        mxscale1 = 2^{shared\_exp}
        $$
        $$
        P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32\\
        $$
      - **-2-axis quantization**: Simultaneously, input x is grouped along the -2 axis by count of 32. A group of 32 numbers $\{\{V_j\}_{j=1}^{32}\}$ is quantized to $\{mxscale2, \{P_j\}_{j=1}^{32}\}$ using the same carry computation method as the -1 axis.
      - ​The quantized $P_{i}$ on the -1 axis compose the output y1Out according to the positions of the corresponding $V_{i}$, and mxscale1 composes the output mxscale1Out according to the grouping on the -1 axis dimension. The quantized $P_{j}$ on the -2 axis compose the output y2Out according to the positions of the corresponding $V_{j}$, and mxscale2 composes the output mxscale2Out according to the grouping on the -2 axis dimension.
    - When dstTypeMax != 0.0/6.0/7.0:
      - **-1-axis quantization**: Input x is grouped along the -1 axis by count of 32, and each group has a length of 32. A block scaling factor $S_{fp32}^b$ is computed separately for each group, and then all elements in the group are mapped to the target low-precision type FP4_E2M1 with the same $S_{fp32}^b$. If the last group has fewer than 32 elements, the missing values are regarded as 0, and the group is processed as a complete group.
        - Find the maximum absolute value of the values in the group:
          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{32})
          $$
        - Map FP32 to within the range representable by the target data type, where $Amax(DType)$ is the maximum value representable by the target precision when dstTypeMax=0; $Amax(DType)$ is the value passed in for dstTypeMax when dstTypeMax!=0.
          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$
        - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
        - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point scaling factor $S_{fp32}^b$ of the block.
        - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:
          $$
          E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{ otherwise} \end{cases}
          $$
        - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
        - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - Apply the final step of quantization. For each element in a group, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result of the -1 axis output is $\left(S^b, [d^i]_{i=1}^{32}\right)$, where $S^b$ represents the block scaling factor, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^{32}$ represents the quantized data in the group.
      - **-2 axis quantization**: Simultaneously, group the input x along the -2 axis by a count of 32, adopt the same algorithm as the -1 axis, and independently compute the block scaling factor and perform quantization for each group. The quantization result of the -2 axis output is $\left(S^b, [d^j]_{j=1}^{32}\right)$.
      - The -1 axis quantization results compose output y1Out, and the corresponding block scaling factors compose output mxscale1Out. The -2 axis quantization results compose output y2Out, and the corresponding block scaling factors compose output mxscale2Out.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation flow, and then call the "aclnnDynamicMxQuantWithDualAxisV2" API to perform computation.

```cpp
aclnnStatus aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize(
  const aclTensor *x,
  char            *roundModeOptional,
  int64_t          dstType,
  int64_t          scaleAlg,
  double           dstTypeMax,
  const aclTensor *y1Out,
  const aclTensor *mxscale1Out,
  const aclTensor *y2Out,
  const aclTensor *mxscale2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicMxQuantWithDualAxisV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input x, corresponding to V<sub>i</sub> and d<sub>i</sub> in the formula.</td>
      <td>When the target type is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of x must be even.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Represents the data conversion mode, corresponding to round_mode in the formula.</td>
      <td><ul><li>When dstType is 40/41, corresponding to data types FLOAT4_E2M1/FLOAT4_E1M2 of outputs y1Out and y2Out, {"rint", "floor", "round"} are supported.</li><li>When dstType is 35/36, corresponding to data types FLOAT8_E5M2/FLOAT8_E4M3FN of outputs y1Out and y2Out, only {"rint"} is supported.</li><li>If a null pointer is passed in, the "rint" mode is used.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Represents the type of y1Out and y2Out after the specified data conversion, corresponding to DType in the formula.</td>
      <td>Input range is {35, 36, 40, 41}, corresponding to the data types of outputs y1Out and y2Out {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Represents the computation method of mxscale1Out and mxscale2Out, corresponding to scaleAlg in the formula.</td>
      <td><ul><li>Values 0, 1, and 2 are supported. Value 0 represents scenario 1, 1 represents scenario 2, and 2 represents scenario 3.</li><li>Only value 0 is supported when dstType is FLOAT4_E1M2.</li><li>Only values 0 and 2 are supported when dstType is FLOAT4_E2M1.</li><li>Only values 0 and 1 are supported when dstType is FLOAT8.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (double)</td>
      <td>Input</td>
      <td>Represents the value of maxType, corresponding to Amax(DType) in the formula.</td>
      <td><ul><li>Supported values are 0.0 and 6.0-12.0. If the value is 0.0, Amax(DType) is the maximum value of the quantization result data type. If the value is 6.0-12.0, Amax(DType) is the passed-in value.</li><li>This value can be set only when the data type is FP4_E2M1 and scaleAlg is 2.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of quantizing input x along the -1 axis, corresponding to P<sub>i</sub> and d<sup>i</sup> in the formula.</td>
      <td>The shape is the same as input x.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>mxscale1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale of each grouping along the -1 axis, corresponding to mxscale1 and S<sup>b</sup> in the formula.</td>
      <td><ul><li>The shape is the value of the -1 axis of x divided by 32 and rounded up, then padded to an even number, with a padding value of 0.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the corresponding result of quantizing input x on the -2 axis, corresponding to P<sub>j</sub> and d<sup>j</sup> in the formula.</td>
      <td>The shape is the same as that of input x.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>mxscale2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale corresponding to each grouping on the -2 axis, corresponding to mxscale2 and S<sup>b</sup> in the formula.</td>
      <td><ul><li>shape is the value on the -2 axis of x divided by 32 and rounded up, and is padded to an even number with 0.</li><li>The mxscale2Out output needs to be interleaved for every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation flow of the operator.</td>
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

  <table style="undefined;table-layout: fixed;width: 1056px"><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
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
      <td>x is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td> The data types and data formats of x, roundModeOptional, dstType, scaleAlg, dstTypeMax, y1Out, mxscale1Out, y2Out, and mxscale2Out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x, y1Out, y2Out, mxscale1Out, or mxscale2Out does not meet the validation conditions.</td>
    </tr>
    <tr>
      <td>roundModeOptional, dstType, scaleAlg, or dstTypeMax does not match the currently supported values.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not within the range of supported platforms.</td>
    </tr>
  </tbody></table>

## aclnnDynamicMxQuantWithDualAxisV2

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1056px"><colgroup>
  <col style="width: 173px">
  <col style="width: 150px">
  <col style="width: 733px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

 - The shape constraints for x, mxscale1Out, and mxscale2Out are as follows:
    - rank(mxscale1Out) = rank(x) + 1.
    - rank(mxscale2Out) = rank(x) + 1.
    - mxscale1Out.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2.
    - mxscale2Out.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2.
    - mxscale1Out.shape[-1] = 2.
    - mxscale2Out.shape[-1] = 2.
    - Other dimensions are consistent with input x.
    - Example: When the shape of input x is [B, M, N] and the target data type is FP8 type, the corresponding shapes of y1 and y2 are [B, M, N], the shape of mxscale1 is [B, M, (ceil(N/32)+2-1)/2, 2], and the shape of mxscale2 is [B, (ceil(M/32)+2-1)/2, N, 2].
 - aclnnDynamicMxQuantWithDualAxisV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_dynamic_mx_quant_with_dual_axis_v2.h"

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
      GetShapeSize(const std::vector<int64_t>& shape)
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

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory
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

  int aclnnDynamicMxQuantWithDualAxisV2Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Custom construction is required based on the API.
      std::vector<int64_t> xShape = {1, 4};
      std::vector<int64_t> y1OutShape = {1, 4};
      std::vector<int64_t> y2OutShape = {1, 4};
      std::vector<int64_t> mxscale1OutShape = {1, 1, 2};
      std::vector<int64_t> mxscale2OutShape = {1, 4, 2};
      void* xDeviceAddr = nullptr;
      void* y1OutDeviceAddr = nullptr;
      void* mxscale1OutDeviceAddr = nullptr;
      void* y2OutDeviceAddr = nullptr;
      void* mxscale2OutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* y1Out = nullptr;
      aclTensor* mxscale1Out = nullptr;
      aclTensor* y2Out = nullptr;
      aclTensor* mxscale2Out = nullptr;
      std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
      std::vector<uint8_t> y1OutHostData = {0, 72, 96, 120};
      std::vector<uint8_t> y2OutHostData = {0, 120, 120, 120};
      std::vector<uint8_t> mxscale1OutHostData = {128, 0};
      std::vector<uint8_t> mxscale2OutHostData = {0, 0, 122, 0, 125, 0, 128, 0};
      char* roundModeOptional = const_cast<char*>("rint");
      int64_t dstType = 36;
      int64_t scaleAlg = 0;
      double dstTypeMax = 0.0;
      // Create the x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the y1Out aclTensor.
      ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the mxscale1Out aclTensor.
      ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale1OutTensorPtr(mxscale1Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> mxscale1OutDeviceAddrPtr(mxscale1OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the y2Out aclTensor.
      ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the mxscale2Out aclTensor.
      ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale2OutTensorPtr(mxscale2Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> mxscale2OutDeviceAddrPtr(mxscale2OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Call the CANN operator library API. Replace the API name with the actual one.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // Call the first-phase API of aclnnDynamicMxQuantWithDualAxisV2.
      ret = aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize(x, roundModeOptional, dstType, scaleAlg, dstTypeMax, y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnDynamicMxQuantWithDualAxisV2.
      ret = aclnnDynamicMxQuantWithDualAxisV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2 failed. ERROR: %d\n", ret); return ret);

      //(boilerplate) Synchronously wait for task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Obtain the output value, copy the result in the device memory to the host, and modify it according to the definition of the specific API.
      auto size1 = GetShapeSize(y1OutShape);
      auto size2 = GetShapeSize(y2OutShape);
      std::vector<uint8_t> y1OutData(
          size1, 0);  // The fp8 data cannot be printed directly in C. Read it as uint8 and manually convert the binary value to fp8.
      std::vector<uint8_t> y2OutData(
          size2, 0);  // The fp8 data cannot be printed directly in C. Read it as uint8 and manually convert the binary value to fp8.
      ret = aclrtMemcpy(y1OutData.data(), y1OutData.size() * sizeof(y1OutData[0]), y1OutDeviceAddr,
                        size1 * sizeof(y1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      ret = aclrtMemcpy(y2OutData.data(), y2OutData.size() * sizeof(y2OutData[0]), y2OutDeviceAddr,
                        size2 * sizeof(y2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("y1Out[%ld] is: %d\n", i, y1OutData[i]);
      }
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("y2Out[%ld] is: %d\n", i, y2OutData[i]);
      }
      size1 = GetShapeSize(mxscale1OutShape);
      size2 = GetShapeSize(mxscale2OutShape);
      std::vector<uint8_t> mxscale1OutData(
          size1, 0);  // The fp8 data cannot be printed directly in C. Read it as uint8 and manually convert the binary value to fp8.
      std::vector<uint8_t> mxscale2OutData(
          size2, 0);  // The fp8 data cannot be printed directly in C. Read it as uint8 and manually convert the binary value to fp8.
      ret = aclrtMemcpy(mxscale1OutData.data(), mxscale1OutData.size() * sizeof(mxscale1OutData[0]), mxscale1OutDeviceAddr,
                        size1 * sizeof(mxscale1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale1Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      ret = aclrtMemcpy(mxscale2OutData.data(), mxscale2OutData.size() * sizeof(mxscale2OutData[0]), mxscale2OutDeviceAddr,
                        size2 * sizeof(mxscale2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale2Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("mxscale1Out[%ld] is: %d\n", i, mxscale1OutData[i]);
      }
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("mxscale2Out[%ld] is: %d\n", i, mxscale2OutData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Device/stream initialization. For details, see the acl API reference.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicMxQuantWithDualAxisV2Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
