# aclnnDynamicDualLevelMxQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:17:07.795Z -->

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

- Description: MX quantization with the destination data type of the FLOAT4 class. Quantization is performed only on the trailing axis, and all preceding axes are flattened. The input is divided into multiple data blocks by the given level0BlockSize, and level 0 quantization is performed on each block to output the quantization scale level0ScaleOut. Then, the level 0 quantization result is used as the new input and divided into multiple data blocks by the given level1BlockSize. Level 1 quantization is performed on each block to output the quantization scale level1ScaleOut. The data type is converted according to round_mode to obtain the quantization result yOut. See [Diagram](../../../docs/en/figures/dynamic_dual_level_mx_quant.png).
  - Optional function: Fuses the smooth scale operation. Before the data input x is quantized, x=x*smooth_scale (broadcast element-wise multiplication) is performed.

- Formula:
  - The input x is grouped on the trailing axis by $k_0$ = level0BlockSize values, and a group of $k_0$ values $\{\{x_i\}_{i=1}^{k_0}\}$ is dynamically quantized to $\{level0Scale, \{temp_i\}_{i=1}^{k_0}\}$, $k_0$ = level0BlockSize. Then temp is grouped on the trailing axis by $k_1$ = level1BlockSize values, and a group of $k_1$ values $\{\{temp_i\}_{i=1}^{k_1}\}$ is dynamically quantized to $\{level1Scale, \{y_i\}_{i=1}^{k_1}\}$, $k_1$ = level1BlockSize

  $$
  input\_max_i = max_i(abs(x_i))
  $$

  $$
  level0Scale = input\_max_i / (FP4\_E2M1\_MAX)
  $$

  $$
  temp_i = cast\_to\_x\_type(x_i / level0Scale), \space i\space from\space 1\space to\space level0BlockSize
  $$

  $$
  shared\_exp = floor(log_2(max_i(|temp_i|))) - emax
  $$

  $$
  level1Scale = 2^{shared\_exp}
  $$

  $$
  y_i = cast\_to\_FP4\_E2M1(temp_i/level1Scale, round\_mode), \space i\space from\space 1\space to\space level1BlockSize
  $$

  - ​The quantized $y_{i}$ values are placed at the positions of the corresponding $x_{i}$ values to form the output yOut. The level0Scale values of the groups on the trailing axis form the output level0ScaleOut, and the level1Scale values of the groups on the trailing axis form the output level1ScaleOut.

  - max_i represents the maximum value in the i-th group.

  - emax: The exponent of the maximum normal number of the corresponding data type.

      | DataType | Emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDynamicDualLevelMxQuantGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnDynamicDualLevelMxQuant" API to perform the computation.

```cpp
aclnnStatus aclnnDynamicDualLevelMxQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScaleOptional,
  char            *roundModeOptional,
  int64_t          level0BlockSize,
  int64_t          level1BlockSize,
  const aclTensor *yOut,
  const aclTensor *level0ScaleOut,
  const aclTensor *level1ScaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicDualLevelMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicDualLevelMxQuantGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 240px">
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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input x, corresponding to <em>x</em><sub>i</sub> in the formula.</td>
      <td><ul><li>The last dimension of x must be an even number;</li><li> Does not support empty tensors.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the optional input smoothScaleOptional.</td>
      <td><ul><li>If smooth scale computation is not fused, pass nullptr to smooth_scale;</li><li>If smooth_scale is not nullptr, the dtype of smooth_scale must be consistent with x, and the shape is 1-dimensional with a length equal to the last dimension of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16 (consistent with input x)</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*) </td>
      <td>Input</td>
      <td>Specifies the data conversion mode, corresponding to round_mode in the formula.</td>
      <td><ul><li>Supports {"rint", "round", "floor"};</li><li> The default value is "rint".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level0BlockSize (int64_t)</td>
      <td>Input</td>
      <td>Represents the block_size of the first-level quantization, corresponding to level0BlockSize in the formula.</td>
      <td>Input range: {512}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level1BlockSize (int64_t)</td>
      <td>Input</td>
      <td>Represents the block_size for second-level quantization, corresponding to level1BlockSize in the formula.</td>
      <td>The input range is {32}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization result of input x, corresponding to <em>y</em><sub>i</sub> in the formula.</td>
      <td><ul><li>The shape is consistent with input x;</li><li> Does not support empty tensors.</li></ul></td>
      <td>FLOAT4_E2M1</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>level0ScaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the scale of first-level quantization, corresponding to level0Scale in the formula.</td>
      <td><ul><li>The shape value on the trailing axis is the value of the trailing axis of x divided by level0BlockSize and rounded up;</li><li> Does not support empty tensors.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>level1ScaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the scale of the second level quantization, corresponding to level1Scale in the corresponding formula.</td>
      <td><ul><li>The shape size is the dim of x plus 1;</li><li> The shape values on the last two axes are ((ceil(x.shape[-1] / level1Blocksize) + 2 - 1) / 2, 2), with even padding and a pad fill value of 0;</li><li> Does not support empty tensors.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the device.</td>
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
  </tbody>
  </table>

- **Return value**

  aclnnStatus: Returns the status code. See [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
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
      <td>The data types and data formats of x, smoothScaleOptional, yOut, level0ScaleOut, and level1ScaleOut are not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of x, smoothScaleOptional, yOut, level0ScaleOut, or level1ScaleOut do not meet the verification conditions.</td>
    </tr>
    <tr>
      <td>roundModeOptional, level0BlockSize, or level1BlockSize is not within the currently supported values.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not among the supported platforms.</td>
    </tr>
  </tbody></table>

## aclnnDynamicDualLevelMxQuant

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
      <td>Workspace memory address applied for on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied for on the Device, obtained by calling the first-phase API aclnnDynamicDualLevelMxQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return code, see [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The shape constraints for x, level0ScaleOut, and level1ScaleOut are described as follows:
    - rank(level1ScaleOut) = rank(x) + 1.
    - level0ScaleOut.shape[-1] = ceil(x.shape[-1] / level0Blocksize).
    - level1ScaleOut.shape[-2] = (ceil(x.shape[-1] / level1Blocksize) + 2 - 1) / 2.
    - level1ScaleOut.shape[-1] = 2.
    - Other dimensions are consistent with input x.
- Deterministic computation description: aclnnDynamicDualLevelMxQuant defaults to a deterministic implementation.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_dynamic_dual_level_mx_quant.h"

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
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
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

  int aclnnDynamicDualLevelMxQuantTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs. Custom construction is required based on the API.
      std::vector<int64_t> xShape = {1, 512};
      std::vector<int64_t> smoothScaleOptionalShape = {512};
      std::vector<int64_t> yOutShape = {1, 512};
      std::vector<int64_t> level0ScaleOutShape = {1, 1};
      std::vector<int64_t> level1ScaleOutShape = {1, 8, 2};
      void* xDeviceAddr = nullptr;
      void* smoothScaleOptionalDeviceAddr = nullptr;
      void* yOutDeviceAddr = nullptr;
      void* level0ScaleOutDeviceAddr = nullptr;
      void* level1ScaleOutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* smoothScaleOptional = nullptr;
      aclTensor* yOut = nullptr;
      aclTensor* level0ScaleOut = nullptr;
      aclTensor* level1ScaleOut = nullptr;

      // Corresponding BF16 values (0->0, 16640->8, 17024->64, 17408->512)
      std::vector<uint16_t> xHostData(512, 16640);
      std::vector<uint16_t> smoothScaleOptionalHostData(512, 0);
      // Corresponding float4_e2m1 values (0->0, 72->4, 96->32, 120->256)
      std::vector<uint8_t> yOutHostData(512, 0);
      // Corresponding float32 value (0->0)
      std::vector<float> level0ScaleOutHostData = {{0}};
      // Corresponding float8_e8m0 value (128->2)
      std::vector<std::vector<std::vector<uint8_t>>> level1ScaleOutHostData(1, std::vector<std::vector<uint8_t>>(8, std::vector<uint8_t>(2, 0)));
      const char* roundModeOptional = "rint";
      int64_t level0Blocksize = 512;
      int64_t level1Blocksize = 32;

      // Create x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create smoothScaleOptional aclTensor.
      ret = CreateAclTensor(smoothScaleOptionalHostData, smoothScaleOptionalShape, &smoothScaleOptionalDeviceAddr, aclDataType::ACL_BF16, &smoothScaleOptional);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> smoothScaleOptionalTensorPtr(smoothScaleOptional, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> smoothScaleOptionalDeviceAddrPtr(smoothScaleOptionalDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create yOut aclTensor.
      ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &yOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create level0ScaleOut aclTensor.
      ret = CreateAclTensor(level0ScaleOutHostData, level0ScaleOutShape, &level0ScaleOutDeviceAddr, aclDataType::ACL_FLOAT, &level0ScaleOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> level0ScaleOutTensorPtr(level0ScaleOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> level0ScaleOutDeviceAddrPtr(level0ScaleOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create level1ScaleOut aclTensor.
      ret = CreateAclTensor(level1ScaleOutHostData, level1ScaleOutShape, &level1ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &level1ScaleOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> level1ScaleOutTensorPtr(level1ScaleOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> level1ScaleOutDeviceAddrPtr(level1ScaleOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Call the CANN operator library API. Change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // Call the first-phase API of aclnnDynamicDualLevelMxQuant.
      ret = aclnnDynamicDualLevelMxQuantGetWorkspaceSize(x, smoothScaleOptional, (char*)roundModeOptional, level0Blocksize, level1Blocksize, yOut, level0ScaleOut, level1ScaleOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnDynamicDualLevelMxQuant.
      ret = aclnnDynamicDualLevelMxQuant(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuant failed. ERROR: %d\n", ret); return ret);

      // (boilerplate) Synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Obtain the output values and copy the results from device memory to host memory. Modify them according to the specific API definition.
      auto size = GetShapeSize(yOutShape) / 2;
      std::vector<uint8_t> yOutData(
          size, 0);  // In C, fp4 data cannot be printed directly. Read it as uint8 and manually convert the binary data to fp4.
      ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                        size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
      }
      size = GetShapeSize(level0ScaleOutShape);
      std::vector<float> level0ScaleOutData(
          size, 0);
      ret = aclrtMemcpy(level0ScaleOutData.data(), level0ScaleOutData.size() * sizeof(level0ScaleOutData[0]), level0ScaleOutDeviceAddr,
                        size * sizeof(level0ScaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy level0ScaleOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("level0ScaleOut[%ld] is: %f\n", i, level0ScaleOutData[i]);
      }
      size = GetShapeSize(level1ScaleOutShape);
      std::vector<uint8_t> level1ScaleOutData(
          size, 0);  // In C, fp8 data cannot be printed directly. Read it as uint8 and manually convert the binary data to fp8.
      ret = aclrtMemcpy(level1ScaleOutData.data(), level1ScaleOutData.size() * sizeof(level1ScaleOutData[0]), level1ScaleOutDeviceAddr,
                        size * sizeof(level1ScaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy level1ScaleOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("level1ScaleOut[%ld] is: %d\n", i, level1ScaleOutData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicDualLevelMxQuantTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuantTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
