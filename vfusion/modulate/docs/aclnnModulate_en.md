# aclnnModulate

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:55:45.282Z -->

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

## Function

- Description: Performs adaptive scaling and shifting on features.
- Formula:

  $$
  out = self \times (1 + scaleOptional) + shiftOptional
  $$

  Where $self \in R ^ {B \times L \times D}$, $scaleOptional \in R ^ {B \times 1 \times D}$, $shiftOptional \in R ^ {B \times 1 \times D}$ (or $scaleOptional \in R ^ {B \times D}$, $shiftOptional \in R ^ {B \times D}$).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnModulateGetWorkspaceSize" API first to obtain the workspace size required for the calculation and the executor that includes the calculation process of the operator, and then call the "aclnnModulate" API to perform the calculation.

```Cpp
aclnnStatus aclnnModulateGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* scaleOptional,
    const aclTensor* shiftOptional,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnModulate(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnModulateGetWorkspaceSize

- **Parameters**:
  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
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
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Indicates the input feature tensor, that is, self in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, seq_len, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Input</td>
      <td>Optional parameter. Indicates the scaling factor, that is, scaleOptional in the formula.</td>
      <td>The data type needs to be the same as self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, 1, hidden_dim) or (batch_size, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shiftOptional</td>
      <td>Input</td>
      <td>Optional parameter, indicating the shift coefficient, that is, shiftOptional in the formula.</td>
      <td>The data type needs to be the same as self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, 1, hidden_dim) or (batch_size, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the output tensor after scaling, that is, out in the formula.</td>
      <td>The data type and shape need to be the same as those of self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the calculation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of self, scaleOptional, and shiftOptional are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self, scaleOptional, and shiftOptional do not meet the constraints.</td>
    </tr>
    <tr>
      <td>self is an empty tensor, and scaleOptional or shiftOptional is not an empty tensor.</td>
    </tr>
  </tbody>
  </table>

## aclnnModulate

- **Parameters**:

  <table><thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address applied by the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied by the device side, obtained by the first-phase API aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, includes calculation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>
- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnModulate defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the build and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_modulate.h"

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
  // Call aclrtMalloc to apply for device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
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

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {2, 1, 1};
  std::vector<int64_t> scaleOptionalShape = {2, 1, 1};
  std::vector<int64_t> shiftOptionalShape = {2, 1, 1};
  std::vector<int64_t> outShape = {2, 1, 1};
  void* selfDeviceAddr = nullptr;
  void* scaleOptionalDeviceAddr = nullptr;
  void* shiftOptionalDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* scaleOptional = nullptr;
  aclTensor* shiftOptional = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData{10, 20};
  std::vector<float> scaleOptionalHostData{20, 30};
  std::vector<float> shiftOptionalHostData{30, 40};
  std::vector<float> outHostData{0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the scaleOptional aclTensor.
  ret = CreateAclTensor(scaleOptionalHostData, scaleOptionalShape, &scaleOptionalDeviceAddr, aclDataType::ACL_FLOAT, &scaleOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the shiftOptional aclTensor.
  ret = CreateAclTensor(shiftOptionalHostData, shiftOptionalShape, &shiftOptionalDeviceAddr, aclDataType::ACL_FLOAT, &shiftOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and change the name to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // Call the first-phase API aclnnModulate.
  ret = aclnnModulateGetWorkspaceSize(self, scaleOptional, shiftOptional, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModulateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnModulate.
  ret = aclnnModulate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModulate failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait synchronously until the task execution ends.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device side to the host side, and modify the code according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar, and modify the code according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(scaleOptional);
  aclDestroyTensor(shiftOptional);
  aclDestroyTensor(out);

  // 7. Release device resources, and modify the code according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(scaleOptionalDeviceAddr);
  aclrtFree(shiftOptionalDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
