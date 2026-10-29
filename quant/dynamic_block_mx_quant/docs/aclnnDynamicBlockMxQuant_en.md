# aclnnDynamicBlockMxQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:11:43.638Z -->

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

- Description: Converts the input variables to the destination data type by performing MX quantization, with a data block (32\*32) as the basic block. In each basic block, a scale algorithm is selected based on the value of scaleAlg to compute the quantization parameter scale (1\*1) of the current block. The parameter is broadcast to scale1 (32\*1) and scale2 (1\*32) as the output. Meanwhile, each value in the basic block is divided by scale and converted to the corresponding dst_type based on round_mode to obtain the quantized result y.

- Formula:
  - Scenario 1, when scaleAlg is 0:
    - Groups the input x into blocks, with a data block (32\*32) as the basic block. The values in one data block $\{\{V_i\}_{i=1}^{32 * 32}\}$ are quantized to $\{scale, \{P_i\}_{i=1}^{32 * 32}\}$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax
      $$

      $$
      scale = 2^{shared\_exp}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/scale, round\_mode), \space i\space from\space 1\space to\space 32 * 32
      $$

    - At the same time, the scale (1\*1) is broadcast into scale1 (32\*1) and scale2 (1\*32) as the outputs scale1 and scale2. The quantized $P_{i}$ values are arranged at the positions of the corresponding $V_{i}$ to form the output y.

    - emax: the exponent bits of the maximum normal number of the corresponding data type.

      | DataType | Emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |
      |  FLOAT4_E1M2  |  0   |
      | FLOAT8_E4M3FN |  8   |
      |  FLOAT8_E5M2  |  15  |

  - Use case 2: When scaleAlg is 2, only the FLOAT4_E2M1 data type is involved:
    - The long vector is divided into blocks, each of length k. A block scaling factor $S_{fp32}^b$ is computed for each block independently, and then all elements in the block are mapped to the target low-precision data type FLOAT4_E2M1 using the same $S_{fp32}^b$. The storage data type of scale is FLOAT8_E8M0. If the last block has fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
    - Find the maximum absolute value in the block:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
    - A new attribute dstTypeMax of the double type is introduced. It defaults to 0 (indicating that dstType is the maximum value of the target data type). If another value is passed in, scale is calculated based on the passed value. Currently, the valid values are 0.0, 6.0, and 7.0. This value can be set only in the FLOAT4_E2M1 use case.
    - Maps FP32 to the range representable by the target data type FLOAT4_E2M1.
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{dst\_type\_max}
      $$
    - Convert the block scale factor $S_{fp32}^b$ to the representable scale value $S_{ue8m0}^b$ in FP8 format.
    - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block floating-point scale factor $S_{fp32}^b$.
    - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{ otherwise} \end{cases}
      $$
    - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
    - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - The final step applied to quantization: for each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ indicates the scale factor of the block, which is $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ indicates the quantized data in the block.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDynamicBlockMxQuantGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnDynamicBlockMxQuant" API to perform the computation.

```cpp
aclnnStatus aclnnDynamicBlockMxQuantGetWorkspaceSize(
  const aclTensor *x,
  char            *roundModeOptional,
  int64_t          dstType,
  int64_t          scaleAlg,
  double           dstTypeMax,
  const aclTensor *yOut,
  const aclTensor *scale1Out,
  const aclTensor *scale2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicBlockMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicBlockMxQuantGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>Indicates the input x, corresponding to V<sub>i</sub> in the formula.</td>
      <td>If dstType is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of x must be even. Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the data conversion mode, corresponding to round_mode in the formula.</td>
      <td><ul><li>When dstType is 40/41 and the data type of output yOut is FLOAT4_E2M1/FLOAT4_E1M2, {"rint", "floor", "round"} are supported;</li><li>When dstType is 35/36 and the data type of output yOut is FLOAT8_E5M2/FLOAT8_E4M3FN, only {"rint"} is supported;</li><li>If a null pointer is passed in, the "rint" mode is used.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the type of yOut after the specified data conversion.</td>
      <td>The input range is {35, 36, 40, 41}, corresponding to the data types of output yOut {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Indicates the calculation method of scale1Out and scale2Out.</td>
      <td>Currently, values 0 and 2 are supported, indicating the OCP Microscaling Formats (Mx) Specification implementation and the Dynamic Dtype Range implementation, respectively.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (double)</td>
      <td>Input</td>
      <td>Indicates the maximum value of the target data type.</td>
      <td><ul><li>Takes effect when scaleAlg=2 and dstType is FLOAT4_E2M1. The scale is calculated based on the input value.</li><li>Currently, only 0.0/6.0/7.0 are supported for dstTypeMax.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantized result of input x, which is P<sub>i</sub> in the corresponding formula.</td>
      <td>The shape is the same as that of input x.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale corresponding to each group along the -1 axis, which is scale1 in the corresponding formula.</td>
      <td>The shape is the value of the -1 axis of x divided by 32 and rounded up, and even-padded with 0.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale of each group on the -2 axis, corresponding to scale2 in the corresponding formula.</td>
      <td><ul><li>The shape is the value of the -2 axis of x divided by 32 and rounded up, and even-padded with 0;</li><li>The scale2Out output needs to be interleaved for every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
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
      <td>The data types or data formats of x, roundModeOptional, dstType, scaleAlg, dstTypeMax, yOut, scale1Out, or scale2Out are not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of x, yOut, scale1Out, or scale2Out do not meet the check requirements.</td>
    </tr>
    <tr>
      <td>roundModeOptional, dstType, scaleAlg, and dstTypeMax do not match the currently supported values.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not within the range of supported platforms.</td>
    </tr>
  </tbody></table>

## aclnnDynamicBlockMxQuant

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained by the first-phase API aclnnDynamicBlockMxQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

 - Shape constraints on x, scale1Out, and scale2Out are as follows:
    - The dimension of x should be 2-3.
    - rank(scale1) = rank(x) + 1.
    - rank(scale2) = rank(x) + 1.
    - scale1.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2.
    - scale2.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2.
    - scale1.shape[-1] = 2.
    - scale2.shape[-1] = 2.
    - Other dimensions are consistent with input x.
    - Example: If input x has shape [B, M, N] and the destination data type is FP8, the corresponding y has shape [B, M, N], scale1 has shape [B, M, (ceil(N/32) + 2 - 1) / 2, 2], and scale2 has shape [B, (ceil(M / 32) + 2 - 1) / 2, N, 2].
 - Deterministic computation: aclnnDynamicBlockMxQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include <cstdio>
  #include "aclnnop/aclnn_dynamic_block_mx_quant.h"


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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
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

  int aclnnDynamicBlockMxQuantTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API.
      std::vector<int64_t> xShape = {1, 4};
      std::vector<int64_t> yOutShape = {1, 4};
      std::vector<int64_t> scale1OutShape = {1, 1, 2};
      std::vector<int64_t> scale2OutShape = {1, 4, 2};
      void* xDeviceAddr = nullptr;
      void* yOutDeviceAddr = nullptr;
      void* scale1OutDeviceAddr = nullptr;
      void* scale2OutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* yOut = nullptr;
      aclTensor* scale1Out = nullptr;
      aclTensor* scale2Out = nullptr;
      std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
      std::vector<uint8_t> yOutHostData = {0, 72, 96, 120};
      std::vector<uint8_t> scale1OutHostData = {128, 0};
      std::vector<uint8_t> scale2OutHostData = {128, 0, 128, 0, 128, 0, 128, 0};
      char* roundModeOptional = const_cast<char*>("rint");
      int64_t dstType = 36;
      int64_t scaleAlg = 0;
      double dstTypeMax = 0;
      // Create the x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the yOut aclTensor.
      ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale1Out aclTensor.
      ret = CreateAclTensor(scale1OutHostData, scale1OutShape, &scale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale1Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale1OutTensorPtr(scale1Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> scale1OutDeviceAddrPtr(scale1OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale2Out aclTensor.
      ret = CreateAclTensor(scale2OutHostData, scale2OutShape, &scale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale2Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale2OutTensorPtr(scale2Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> scale2OutDeviceAddrPtr(scale2OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // Call the aclnnDynamicBlockMxQuant first-phase API.
      ret = aclnnDynamicBlockMxQuantGetWorkspaceSize(x, roundModeOptional, dstType, scaleAlg, dstTypeMax, yOut, scale1Out, scale2Out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory according to the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnDynamicBlockMxQuant.
      ret = aclnnDynamicBlockMxQuant(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuant failed. ERROR: %d\n", ret); return ret);

      //(Boilerplate) Synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Obtain the output value and copy the result from device memory to host memory. Modify it based on the definition of the specific API.
      auto size1 = GetShapeSize(yOutShape);
      std::vector<uint8_t> yOutData(
          size1, 0);  // fp4 data cannot be directly printed in C. Read the data as uint8 and manually convert it from binary to fp4.
      ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                        size1 * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
      }
      auto size2 = GetShapeSize(scale1OutShape);
      auto size3 = GetShapeSize(scale2OutShape);
      std::vector<uint8_t> scale1OutData(
          size2, 0);  // fp8 data cannot be directly printed in C. Read the data as uint8 and manually convert it from binary to fp8.
      std::vector<uint8_t> scale2OutData(
          size3, 0);  // fp8 data cannot be directly printed in C. Read the data as uint8 and manually convert it from binary to fp8.
      ret = aclrtMemcpy(scale1OutData.data(), scale1OutData.size() * sizeof(scale1OutData[0]), scale1OutDeviceAddr,
                        size2 * sizeof(scale1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale1Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      ret = aclrtMemcpy(scale2OutData.data(), scale2OutData.size() * sizeof(scale2OutData[0]), scale2OutDeviceAddr,
                        size3 * sizeof(scale2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale2Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("scale1Out[%ld] is: %d\n", i, scale1OutData[i]);
      }
      for (int64_t i = 0; i < size3; i++) {
          LOG_PRINT("scale2Out[%ld] is: %d\n", i, scale2OutData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicBlockMxQuantTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuantTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
