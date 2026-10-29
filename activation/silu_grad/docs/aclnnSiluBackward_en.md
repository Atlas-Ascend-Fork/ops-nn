# aclnnSiluBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T13:51:02.806Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: backward propagation of [aclnnSilu](../../swish/docs/aclnnSilu.md). Computes the gradient input of silu based on the silu backward propagation gradient and the forward output.
- Formula:

  $$
  \sigma(x) = {\frac{1} {1+{e}^{-x}}}
  $$

  $$
  s(x) = x\sigma(x)
  $$

  $$
  s^\prime(x) = \sigma(x)(1+x-x\sigma(x))
  $$

  $$
  gradInput = gradOutput * s^\prime(x)
  $$

where $\sigma(x)$ is the sigmoid function, $s(x)$ is the SiLU function, and $s^\prime(x)$ is the derivative of the SiLU function.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSiluBackwardGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnSiluBackward" API to execute the computation.

```Cpp
aclnnStatus aclnnSiluBackwardGetWorkspaceSize(
  const aclTensor*    gradOutput,
  const aclTensor*    self,
  aclTensor*          gradInput,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```Cpp
aclnnStatus aclnnSiluBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnSiluBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 280px">
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
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input gradient, which is gradOutput in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>gradOutput, self, and gradInput must have the same data type and shape.</li><li>gradOutput, self, and gradInput must have the same shape and satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input data, which is x in the formula and corresponds to the input parameter of forward propagation.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data types and shapes of gradOutput, self, and gradInput must be the same.</li><li>The shapes of gradOutput, self, and gradInput must be the same and comply with the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the input self. gradInput in the formula.</td>
      <td><ul><li>The data types and shapes of gradOutput, self, and gradInput must be the same.</li><li>The shapes of gradOutput, self, and gradInput must be the same and comply with the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to allocate on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
   </tbody>
  </table>

   <!-- npu="910" id7 -->
   - <term>Atlas training products</term>: Data types FLOAT16 and FLOAT are supported.
   <!-- end id7 -->

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
      <td>The input gradOutput, self, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of gradOutput, self, or gradInput is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, self, or gradInput are different or do not meet requirements.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, self, or gradInput are different or do not conform to the broadcast relationship.</td>
    </tr>
  </tbody></table>

## aclnnSiluBackward

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the device, obtained by calling the first-phase API aclnnSiluBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where tasks are executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSiluBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_silu_backward.h"

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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device.
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

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> gradOutputShape = {2, 3};
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> gradInputShape = {2, 3};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0};

  // Create gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutputShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace with the specific API name.
  int64_t dim = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnSiluBackward first-phase API.
  ret = aclnnSiluBackwardGetWorkspaceSize(gradOut, self, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSiluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnSiluBackward second-phase API.
  ret = aclnnSiluBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSiluBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result on the device side to the host side. Modify based on the specific API definition.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), gradInputDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify based on the specific API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(gradInput);

  // 7. Release device resources. Modify based on the specific API definition.
  aclrtFree(gradOutDeviceAddr);
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
