# aclnnAdaptiveAvgPool3d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:51:15.508Z -->

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

Performs 3D adaptive average pooling on the tensor self given the specified 3D output shape information (outputSize). Unlike aclnnAvgPool3d, aclnnAdaptiveAvgPool3d only requires specifying the output size, and it automatically derives the kernel size and corresponding stride.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnAdaptiveAvgPool3dGetWorkspaceSize" to obtain the workspace size required for the computation and the executor that contains the computation process. Then, call "aclnnAdaptiveAvgPool3d" to execute the computation.

```Cpp
aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *outputSize,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAdaptiveAvgPool3d(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaptiveAvgPool3dGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>self</td>
      <td>Input</td>
      <td>Tensor to be calculated.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW, ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize</td>
      <td>Input</td>
      <td>Specifies the output shape size in the DHW dimensions.</td>
      <td>The array length is always 3.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, with the same data type as self.</td>
      <td>The shape of out must be the same as the shape deduced from the shapes of self and outputSize.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW, ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: The data types of parameters `self` and `out` do not support BFLOAT16.

  <!-- end id7 -->

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).
    The first-phase API performs input parameter validation and reports an error in the following cases:

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
      <col style="width: 267px">
      <col style="width: 124px">
      <col style="width: 775px">
      </colgroup>
      <thead>
      <tr>
        <th>Error Code</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>

    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>self, outputSize, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data type or data format of self is not supported.</td>
      </tr>
      <tr>
        <td>The data types of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The dimension of self is not 4 or 5.</td>
      </tr>
      <tr>
        <td>Any dimension other than the first in the shape of self or out is not greater than 0.</td>
      </tr>
      <tr>
        <td>The length of outputSize is not 3.</td>
      </tr>
      <tr>
        <td>The NC dimensions of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The shape of out is inconsistent with the shape deduced from the shape of self and outputSize.</td>
      </tr>
    </tbody></table>

  <!-- npu="310p" id8 -->
  - <term>Atlas inference products</term>: For parameters `self` and `out`, a first dimension that is not greater than 0 is not supported.

  <!-- end id8 -->

## aclnnAdaptiveAvgPool3d

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
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
        <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnAdaptiveAvgPool3dGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op. Contains the computation flow of the operator.</td>
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

- Deterministic computation:
  - aclnnAdaptiveAvgPool3d defaults to a deterministic implementation.

- Shape description:
  - self.shape = (N, C, Din, Hin, Win) or (C, Din, Hin, Win)
  - outputSize = [Dout, Hout, Wout]
  - out.shape = (N, C, Dout, Hout, Wout) or (C, Dout, Hout, Wout)

## Example

The sample code is as follows and is for reference only. For details about the compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_avg_pool3d.h"

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
  // Boilerplate code, resource initialization.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate code) Device/stream initialization. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output. Custom construction is required based on the API.
  std::vector<int64_t> inputShape = {2, 1, 2, 2, 4};
  std::vector<int64_t> outShape = {2, 1, 2, 2, 2};
  std::vector<int64_t> outSize = {2, 2, 2};
  void* inputDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* out = nullptr;
  aclIntArray* outputSize = aclCreateIntArray(outSize.data(), 3);
  std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> outHostData(16, 0);

  // Create input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAdaptiveAvgPool3d.
  ret = aclnnAdaptiveAvgPool3dGetWorkspaceSize(input, outputSize, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveAvgPool3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAdaptiveAvgPool3d.
  ret = aclnnAdaptiveAvgPool3d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveAvgPool3d failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device-side memory to the host side. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify it according to the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(out);

  // 7. Release Device resources. Modify it according to the specific API definition.
  aclrtFree(inputDeviceAddr);
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
