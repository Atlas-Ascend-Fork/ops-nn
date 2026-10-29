# aclnnAdaptiveAvgPool3dBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:51:37.038Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

Backward computation of [aclnnAdaptiveAvgPool3d](../../adaptive_avg_pool3d/docs/aclnnAdaptiveAvgPool3d.md).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnAdaptiveAvgPool3dBackward" API to execute the computation.

```Cpp
aclnnStatus aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclTensor   *self,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAdaptiveAvgPool3dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1453px"><colgroup>
  <col style="width: 154px">
  <col style="width: 121px">
  <col style="width: 208px">
  <col style="width: 226px">
  <col style="width: 275px">
  <col style="width: 186px">
  <col style="width: 138px">
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient of the current node.</td>
      <td>Data type is the same as self, and NC dimensions are consistent with self.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW, ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Leaf node.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW, ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Corresponds to the gradient of the input leaf nodes.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW, ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
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
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
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
        <td>gradOutput, self, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data type and data format of gradOutput and self are not supported.</td>
      </tr>
      <tr>
        <td>The data types of gradOutput, self, and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The dims of gradOutput, self, and out are not 4 or 5.</td>
      </tr>
      <tr>
        <td>The shapes of gradOutput, self, and out do not match.</td>
      </tr>
      <tr>
        <td>Any dim in the shapes of gradOutput or self is not greater than 0.</td>
      </tr>
      <tr>
        <td>The data formats of gradOutput and self are inconsistent.</td>
      </tr>
      <tr>
        <td>The NC dims are inconsistent.</td>
      </tr>
    </tbody>
    </table>
    Ascend 950PR&950DT products: the value of the N dim in the shapes of gradOutput and self can be 0

## aclnnAdaptiveAvgPool3dBackward

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
      <td>Workspace memory address applied for on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied for on the device side, obtained by calling the first-phase API aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, including the specific computation process of an operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  <!-- npu="A3,910b,310p,910" id11 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: aclnnAdaptiveAvgPool3dBackward defaults to a non-deterministic implementation and supports enabling determinism via aclrtCtxSetSysParamOpt.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnAdaptiveAvgPool3dBackward defaults to a deterministic implementation.
  <!-- end id12 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "math.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_avg_pool3d_backward.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Initialize device/stream. For details, see the acl API manual.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output.
  std::vector<int64_t> yGradShape = {2, 2, 1, 1, 2};
  std::vector<int64_t> xShape = {2, 2, 1, 1, 4};
  std::vector<int64_t> xGradShape = {2, 2, 1, 1, 4};
  void* yGradDeviceAddr = nullptr;
  void* xDeviceAddr = nullptr;
  void* xGradDeviceAddr = nullptr;
  aclTensor* yGrad = nullptr;
  aclTensor* x = nullptr;
  aclTensor* xGrad = nullptr;
  std::vector<float> yGradHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> xHostData(GetShapeSize(xShape), 1);
  std::vector<float> xGradHostData(16, 0);
  // Create the yGrad aclTensor.
  ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT, &yGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create xGrad aclTensor.
  ret = CreateAclTensor(xGradHostData, xGradShape, &xGradDeviceAddr, aclDataType::ACL_FLOAT, &xGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAdaptiveAvgPool3dBackward.
  ret = aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(yGrad, x, xGrad, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAdaptiveAvgPool3dBackward.
  ret = aclnnAdaptiveAvgPool3dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveAvgPool3dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory.
  auto size = GetShapeSize(xGradShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xGradDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor.
  aclDestroyTensor(yGrad);
  aclDestroyTensor(x);
  aclDestroyTensor(xGrad);

  // 7. Release device resources. Modify based on the definition of the specific API.
  aclrtFree(yGradDeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(xGradDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
