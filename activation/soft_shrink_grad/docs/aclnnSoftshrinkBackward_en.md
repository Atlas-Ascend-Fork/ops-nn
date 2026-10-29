# aclnnSoftshrinkBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T14:19:12.667Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Computes the backward propagation gradient of [Softshrink](../../softshrink/docs/aclnnSoftshrink.md). `gradOutput` is the upstream gradient, and `self` is the input of the Softshrink forward computation.
- Formula:
  - Softshrink forward computation formula:

    $$
    Softshrink(x)=
    \begin{cases}
    x - {\lambda}, \ if \ x > {\lambda}\\
    x + {\lambda}, \ if \ x < -{\lambda}\\
    0, \ otherwise
    \end{cases}
    $$

  - Softshrink backward propagation formula:

    $$
    gradInput=
    \begin{cases}
    gradOutput, \ if \ \lvert self \rvert > \lambda \ or \ self \ is \ NaN \\
    0, \ otherwise
    \end{cases}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSoftshrinkBackwardGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the computation process of the operator, and then call the "aclnnSoftshrinkBackward" API to execute the computation.

```Cpp
aclnnStatus aclnnSoftshrinkBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclTensor* self,
  const aclScalar* lambd,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSoftshrinkBackward(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnSoftshrinkBackwardGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
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
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Upstream gradient for backward propagation, corresponding to gradOutput in the formula.</td>
      <td><ul><li>Must not be a null pointer; empty tensors are supported.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li><li>The data type can be different from that of self; if different, it is internally converted to FLOAT for computation.</li></ul></td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input of the Softshrink forward computation, corresponding to self in the formula.</td>
      <td><ul><li>Null pointers cannot be passed in; empty tensors are supported.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with gradOutput.</li><li>The data type can differ from that of gradOutput; if different, it is internally converted to FLOAT for computation.</li></ul></td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>lambd (aclScalar*)</td>
      <td>Input</td>
      <td>Threshold for the Softshrink computation, λ in the formula.</td>
      <td>Null pointers cannot be passed in; the value must be greater than or equal to 0.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Output gradient of Softshrink backward propagation, that is, gradInput in the formula.</td>
      <td><ul><li>Null pointers cannot be passed in; empty tensors are supported.</li><li>The shape must be the same as the broadcast result of self and gradOutput.</li></ul></td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="950,A3,910b" id7 -->
  - <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The data types of gradOutput, self, and gradInput support FLOAT16, FLOAT, and BFLOAT16.
  <!-- end id7 -->
  <!-- npu="310b,310p,910" id8 -->
  - <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: The data types of gradOutput, self, and gradInput support FLOAT16 and FLOAT.
  <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>gradOutput, self, lambd, gradInput, workspaceSize, or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of gradOutput, self, or gradInput is not within the supported range of the current product.</td>
    </tr>
    <tr>
      <td>The shapes of self and gradOutput do not comply with the broadcast rules.</td>
    </tr>
    <tr>
      <td>lambd is NaN or less than 0.</td>
    </tr>
    <tr>
      <td>The shape of gradInput is inconsistent with the broadcast result of self and gradOutput.</td>
    </tr>
    <tr>
      <td>gradOutput or self has more than 8 dimensions.</td>
    </tr>
  </tbody></table>

## aclnnSoftshrinkBackward

- **Parameters:**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
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
      <td>Workspace memory address allocated on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device side, obtained through the first-phase API aclnnSoftshrinkBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

   aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation: aclnnSoftshrinkBackward defaults to a deterministic implementation.

## Example

The following code shows the basic call flow. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_softshrink_backward.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Initialize runtime resources.
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

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Initialize the device and stream.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs.
  std::vector<int64_t> gradOutputShape = {4, 2};
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> gradInputShape = {4, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gradInput = nullptr;
  aclScalar* lambd = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> selfHostData = {1, 1, 1, 2, 1, 2, 3, 3};
  std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float lambdValue = 1.2f;
  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // lambd aclScalar
  lambd = aclCreateScalar(&lambdValue, aclDataType::ACL_FLOAT);
  CHECK_RET(lambd != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return -1);
  // Create the gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // Call the first-phase API of aclnnSoftshrinkBackward.
  ret = aclnnSoftshrinkBackwardGetWorkspaceSize(gradOutput, self, lambd, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftshrinkBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnSoftshrinkBackward.
  ret = aclnnSoftshrinkBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftshrinkBackward failed. ERROR: %d\n", ret); return ret);
  // 4. Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output and copy the device-side result to the host side.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyScalar(lambd);
  aclDestroyTensor(gradInput);

  // 7. Release device resources.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
