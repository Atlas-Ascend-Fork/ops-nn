# aclnnFusedQuantMatmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:08:21.743Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- Description: Fused quantized matrix multiplication and Gelu computation.
- Formula:

  - x1Scale, bias INT32 (no offset in this scenario):

    $$
    qbmmout = (x1@x2 + bias) * x2Scale * x1Scale
    $$

  - x1Scale, bias BFLOAT16/FLOAT16/FLOAT32 (this scenario has no offset):

    $$
    qbmmout = x1@x2 * x2scale * x1Scale + bias
    $$

  - x1Scale without bias:

    $$
    qbmmout = x1@x2 * x2Scale * x1Scale
    $$

  - The OP type is defined by the fusedOpType input and supports the following:

    - gelu_tanh operation:

      $$
      out = gelu\_tanh(qbmmout)
      $$

    - gelu_erf operation:

      $$
      out = gelu\_erf(qbmmout)
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnFusedQuantMatmulGetWorkspaceSize" to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnFusedQuantMatmul" to execute the computation.

```c++
aclnnStatus aclnnFusedQuantMatmulGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *x1Scale,
  const aclTensor *x2Scale,
  const aclTensor *yScaleOptional,
  const aclTensor *x1OffsetOptional,
  const aclTensor *x2OffsetOptional,
  const aclTensor *yOffsetOptional,
  const aclTensor *biasOptional,
  const aclTensor *x3Optional,
  const char      *fusedOpType,
  int64_t          groupSizeOptional,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnFusedQuantMatmul(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnFusedQuantMatmulGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 198px">
  <col style="width: 121px">
  <col style="width: 220px">
  <col style="width: 397px">
  <col style="width: 220px">
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
        <td>x1</td>
        <td>Input</td>
        <td>Input x1 in the formula.</td>
        <td>          <ul>
            <li><a href="../../../docs/en/context/non_contiguous_tensor.md">Non-contiguous tensors</a> are supported only when the last m and k axes are transposed. Non-contiguous tensors are not supported along other axes.</li>
            <li>The size of the last dimension cannot exceed 65535.</li>
          </ul>
        </td>
        <td>INT4, INT8, INT32</td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td>          <ul>
            <li>For ND format, non-contiguous tensors with the last two axes transposed are supported. Non-contiguous tensors in other use cases are not supported. See <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensor</a>.</li>
            <li>The size of the last dimension cannot exceed 65535.</li>
          </ul>
        </td>
        <td>INT4, INT8, INT32</td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1Scale</td>
        <td>Input</td>
        <td>Represents the quantization parameter, that is, the input x1Scale in the formula.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Scale</td>
        <td>Input</td>
        <td>Denotes the quantization parameter, that is, the input x2Scale in the formula.</td>
        <td>-</td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yScaleOptional</td>
        <td>Input</td>
        <td>Quantization scale parameter of output y, used in static quantization.</td>
        <td>Reserved parameter, not supported in the current version. Must be set to nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x1OffsetOptional</td>
        <td>Input</td>
        <td>The input x1Offset in the formula.</td>
        <td>Reserved parameter. Not supported in the current version. Set it to nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2OffsetOptional</td>
        <td>Input</td>
        <td>The input x2Offset in the formula.</td>
        <td>Reserved parameter, not supported in the current version. Pass in nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOffsetOptional</td>
        <td>Input</td>
        <td>Input yOffset in the formula.</td>
        <td>Reserved parameter, not supported in the current version. Pass in nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>Input bias in the formula.</td>
        <td>Pass nullptr if there is no bias.</td>
        <td>INT32, FLOAT32, BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>1, 3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x3Optional</td>
        <td>Input</td>
        <td>Input of the fused binary operation.</td>
        <td>Only fused unary operations are supported currently. This is a reserved parameter and is not supported in the current version. nullptr must be passed in.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>fusedOpType</td>
        <td>Input</td>
        <td>Input fusedOpType in the formula, specifying the fusion mode supported by the QuantBatchMatmul operator.</td>
        <td>The fusion mode must be "gelu_erf" or "gelu_tanh".</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSizeOptional</td>
        <td>Input</td>
        <td>Used to input the quantization group sizes along the m, n, and k dimensions.</td>
        <td>Reserved parameter, not supported in the current version. Pass in nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output out in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1083px"><colgroup>
  <col style="width: 251px">
  <col style="width: 129px">
  <col style="width: 703px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="1">161001</td>
      <td>The input x1, x2, x1Scale, x2Scale, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The shape of x1, x2, biasOptional, x1Scale, x2Scale, or out fails the validation check.</td>
    </tr>
    <tr>
      <td>The data type and data format of x1, x2, biasOptional, x1Scale, x2Scale, or out are out of the supported range.</td>
    </tr>
    <tr>
      <td>x1, x2, biasOptional, x1Scale, x2Scale, or out is an empty tensor.</td>
    </tr>
    <tr>
      <td>The passed fusedOpType is not one of "gelu_tanh" or "gelu_erf".</td>
    </tr>
    <tr>
      <td>The passed yScaleOptional, x1OffsetOptional, x2OffsetOptional, yOffsetOptional, x3Optional, or groupSizeOptional parameter is currently not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedQuantMatmul

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
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
      <td>Memory address of the workspace allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnFusedQuantMatmulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: aclnnFusedQuantMatmul defaults to a deterministic implementation.

  <!-- end id7 -->

- The input and output support the following data type combinations:

  | x1 | x2 | x1Scale | x2Scale | x2OffsetOptional | yScaleOptional | biasOptional | yOffsetOptional | Out |
  | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
  | INT8                      | INT8                      | FLOAT32| FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |

- The current API supports x1 pertoken quantization and x2 perchannel/pertensor quantization. For different [quantization modes](../../../docs/en/context/quant_mode_introduction.md), the constraints on input dtype combinations of x1, x2, x1Scale, and x2Scale are as follows:
  - The x1 data type supports INT8, INT32, and INT4.
    - When the data type is INT32 or INT4, it is an INT4 quantization scenario:
      - Currently, only ND input is supported.
      - Currently, only non-transposed input is supported.
      - The inner axis of x1 must be an even number.
    - When the data type is INT32, each INT32 data stores eight INT4 data values, with the corresponding dimension representation: (batch, m, k // 8). k must be a multiple of 8.
  - x2 supports INT8, INT32, and INT4 data types.
    - When the data type is INT32 or INT4, it is an INT4 quantization scenario:
      - This API supports only the 2D ND format.
      - The inner axis of x2 must be an even number.
    - When the data type is INT32, each INT32 value stores eight INT4 values:
    - Use the aclnnConvertWeightToINT4Pack API to convert the data format of x2 from INT32 (one int4 stored in bits 0-3 of one int32) to INT32 (eight int4 values stored in one int32) or INT4 (one int4 represented by one int4). For details, see [aclnnConvertWeightToINT4Pack API](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md).
  - x1Scale constraints are as follows:
    - The shape supports 1D with a shape of (m,). The data type supports FLOAT32.
  - x2Scale constraints are as follows:
    - The shape supports 1D with a shape of (n,) or (1,), where n is consistent with n of x2. The data type supports FLOAT32 and BFLOAT16.
  - Constraints for biasOptional are as follows:
    - The shape supports 1D and 3D. In the INT4 quantization scenario, only 1D biasOptional is supported, with the shape (n). For 3D, the biasOptional shape is (batch, 1, n).
    - The data type supports int32, float32, bfloat16, or float16.
  - Constraints for out are as follows:
    - The shape supports 2D to 6D, (batch, m, n). The data type supports FLOAT16 and BFLOAT16.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="A3,910b" id8 -->
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
<!-- end id8 -->

x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_fused_quant_matmul.h"

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
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnFusedQuantMatmulTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API.
      std::vector<int64_t> x1Shape = {2, 8192};   // (m,k)
      std::vector<int64_t> x2Shape = {8192, 128}; // (k,n)
      std::vector<int64_t> x1ScaleShape = {2}; // x1ScaleShape = [m]
      std::vector<int64_t> x2ScaleShape = {128}; // x2ScaleShape = [n]
      std::vector<int64_t> outShape = {2, 128};

      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* out = nullptr;
      std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
      std::vector<int8_t> x2HostData(GetShapeSize(x2Shape), 1);
      std::vector<float> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
      std::vector<float> x2ScaleHostData(GetShapeSize(x2ScaleShape), 1);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 1);

      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int64_t groupSize = 0;
      const char fusedOpType[] = "gelu_tanh";

      // 3. Call the CANN operator library API. Change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;

      ret = aclnnFusedQuantMatmulGetWorkspaceSize(
          x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, fusedOpType, groupSize, out,
          &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnFusedQuantMatmul.
      ret = aclnnFusedQuantMatmul(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmul failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device side to the host side. Change it according to the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0); // fp16 data cannot be printed directly in C. Read it as uint16 and convert the binary value to fp16.
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API reference.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnFusedQuantMatmulTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmulTest failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```
