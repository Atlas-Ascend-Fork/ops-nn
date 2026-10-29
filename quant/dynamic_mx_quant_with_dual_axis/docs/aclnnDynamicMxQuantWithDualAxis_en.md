# aclnnDynamicMxQuantWithDualAxis

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:20:56.283Z -->

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

- Description: Performs MX quantization with the destination data type FLOAT4 or FLOAT8 along the -1 axis and the -2 axis simultaneously. On the given -1 axis and -2 axis, for every 32 elements, the quantization scales mxscale1 and mxscale2 of the two groups are computed as the corresponding parts of the outputs mxscale1Out and mxscale2Out. Then all elements of the two groups are divided by the corresponding mxscale1 or mxscale2 and converted to the corresponding dstType according to round_mode, obtaining the quantization results y1 and y2 as the corresponding parts of the outputs y1Out and y2Out. When dstType is FLOAT8_E4M3FN or FLOAT8_E5M2, the value of scaleAlg specifies the algorithm for computing mxscale.

- Axis merging: In the operator implementation, all axes before the -2 axis (exclusive) are merged. That is, for a tensor with the input shape $(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})$, the dimensions $(d_0, d_1, ..., d_{n-3})$ before the -2 axis are merged into one dimension. This is equivalent to quantizing the input after reshaping it to $(d_0 \times d_1 \times ... \times d_{n-3}, d_{n-2}, d_{n-1})$.

- Formula:
  - Use case 1: When scaleAlg is 0, the OCP Microscaling Formats (Mx) Specification implementation is used:
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

    - The quantized $P_{i}$ along the -1 axis are arranged according to the positions of the corresponding $V_{i}$ to form the output y1Out, and mxscale1 is arranged according to the corresponding groups along the -1 axis dimension to form the output mxscale1Out. The quantized $P_{j}$ along the -2 axis are arranged according to the positions of the corresponding $V_{j}$ to form the output y2Out, and mxscale2 is arranged according to the corresponding groups along the -2 axis dimension to form the output mxscale2Out.

    - emax: The exponent of the maximum normal number of the corresponding data type.

      | DataType | Emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |
      |  FLOAT4_E1M2  |  0   |
      | FLOAT8_E4M3FN |  8   |
      |  FLOAT8_E5M2  |  15  |

  - Use case 2: when scaleAlg is 1, only the FP8 type is involved (round-up algorithm):
    - **-1 axis quantization**: Group the input x along the -1 axis by a count of 32, with each group being 32 elements long. Calculate a block scaling factor $S_{fp32}^b$ for each group separately, and then map all elements in the group to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last group has fewer than 32 elements, treat the missing values as 0 and process the group as a full one.
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
      - Final step of quantization: for each element in the group, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result on the -1 axis is $\left(S^b, [d^i]_{i=1}^{32}\right)$, where $S^b$ represents the block scaling factor, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^{32}$ represents the quantized data in the group.
    - **-2 axis quantization**: Meanwhile, the input x is grouped along the -2 axis with a group count of 32. The same round-up algorithm as for the -1 axis is used to independently compute the block scaling factor for each group and quantize the group. The quantization result on the -2 axis is $\left(S^b, [d^j]_{j=1}^{32}\right)$.
    - The -1 axis quantization results form the output y1Out, and the corresponding block scaling factors form the output mxscale1Out. The -2 axis quantization results form the output y2Out, and the corresponding block scaling factors form the output mxscale2Out.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call the "aclnnDynamicMxQuantWithDualAxis" API to perform the computation.

```cpp
aclnnStatus aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(
  const aclTensor *x,
  char            *roundModeOptional,
  int64_t          dstType,
  int64_t          scaleAlg,
  const aclTensor *y1Out,
  const aclTensor *mxscale1Out,
  const aclTensor *y2Out,
  const aclTensor *mxscale2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicMxQuantWithDualAxis(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 240px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 120px">
  <col style="width: 130px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input x, corresponding to V<sub>i</sub> in the formula.</td>
      <td><ul><li>If the destination type is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of x must be an even number.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Represents the data conversion mode, corresponding to round_mode in the formula.</td>
      <td><ul><li>If dstType is 40/41, and the data types of outputs y1Out and y2Out are FLOAT4_E2M1/FLOAT4_E1M2, {"rint", "floor", "round"} is supported.</li><li>If dstType is 35/36, and the data types of outputs y1Out and y2Out are FLOAT8_E5M2/FLOAT8_E4M3FN, only {"rint"} is supported.</li><li>If a null pointer is passed in, the "rint" mode is used.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Represents the type of y1Out and y2Out after the specified data conversion.</td>
      <td><ul><li>The input range is {35, 36, 40, 41}, which respectively correspond to the data types of outputs y1Out and y2Out: {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Represents the calculation method of mxscale1Out and mxscale2Out, corresponding to scaleAlg in the formula.</td>
      <td><ul><li>Supported values are 0 and 1. Value 0 corresponds to use case 1, and value 1 corresponds to use case 2.</li><li>Only value 0 is supported when dstType is FLOAT4_E2M1/FLOAT4_E1M2.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantized result of input x along the -1 axis, corresponding to P<sub>i</sub> in the formula.</td>
      <td><ul><li>The shape is consistent with input x.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>mxscale1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale corresponding to each grouping along the -1 axis, corresponding to mxscale1 in the formula.</td>
      <td><ul><li>The shape is the value of the -1 axis of x divided by 32 and rounded up, and then padded to an even number, with a padding value of 0.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantized result of input x along the -2 axis, corresponding to P<sub>j</sub> in the formula.</td>
      <td><ul><li>shape is consistent with input x.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>mxscale2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale corresponding to each grouping along the -2 axis, corresponding to mxscale2 in the formula.</td>
      <td><ul><li>shape is the -2 axis value of x divided by 32 and rounded up, then padded to an even number; the padding value is 0.</li><li>The mxscale2Out output must be interleaved for every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
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

  <table style="undefined;table-layout: fixed;width: 1000px"><colgroup>
  <col style="width: 253px">
  <col style="width: 130px">
  <col style="width: 720px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error code</th>
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
      <td> The data types and data formats of x, roundModeOptional, dstType, scaleAlg, y1Out, mxscale1Out, y2Out, or mxscale2Out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x, y1Out, y2Out, mxscale1Out, or mxscale2Out does not meet the validation conditions.</td>
    </tr>
    <tr>
      <td>roundModeOptional, dstType, or scaleAlg is not within the currently supported values.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not within the range of supported platforms.</td>
    </tr>
  </tbody></table>

## aclnnDynamicMxQuantWithDualAxis

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
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

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_dynamic_mx_quant_with_dual_axis.h"

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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to request device memory.
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

  int aclnnDynamicMxQuantWithDualAxisTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API.
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

      // Call the CANN operator library API. Change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // Call the first-phase API of aclnnDynamicMxQuantWithDualAxis.
      ret = aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(x, roundModeOptional, dstType, scaleAlg, y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Request device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnDynamicMxQuantWithDualAxis.
      ret = aclnnDynamicMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxis failed. ERROR: %d\n", ret); return ret);

      // (boilerplate) Wait synchronously for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Obtain the output values, copy the results from the device memory to the host memory, and modify them based on the definition of the specific API.
      auto size1 = GetShapeSize(y1OutShape);
      auto size2 = GetShapeSize(y2OutShape);
      std::vector<uint8_t> y1OutData(
          size1, 0);  // fp8 data cannot be directly printed in C. Read it as uint8 and manually convert the binary data to fp8.
      std::vector<uint8_t> y2OutData(
          size2, 0);  // fp8 data cannot be directly printed in C. Read it as uint8 and manually convert the binary data to fp8.
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
          size1, 0);  // fp8 data cannot be directly printed in C. Read it as uint8 and manually convert the binary data to fp8.
      std::vector<uint8_t> mxscale2OutData(
          size2, 0);  // fp8 data cannot be directly printed in C. Read it as uint8 and manually convert the binary data to fp8.
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
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Fill in deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicMxQuantWithDualAxisTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
