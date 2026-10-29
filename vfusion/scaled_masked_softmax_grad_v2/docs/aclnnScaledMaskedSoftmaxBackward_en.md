# aclnnScaledMaskedSoftmaxBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:00:13.132Z -->

## Applicable Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

- Description: backward propagation of softmax, with scaling and masking applied to the result.
- Formula:

  $$
  out = gradOutput \cdot output - sum(gradOutput \cdot output)\cdot output \\
  out = \begin{cases}
  out * scale, &\text { mask is 0 } \\
  0, &\text { mask is 1 }
  \end{cases}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the computation process of the operator, and then call "aclnnScaledMaskedSoftmaxBackward" to perform the computation.

```Cpp
aclnnStatus aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* y,
    const aclTensor* mask,
    double scale,
    bool fixTriuMask,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScaledMaskedSoftmaxBackward(
    void*             workspace,
    uint64_t          workspaceSize,
    aclOpExecutor*    executor,
    const aclrtStream stream)
```

## aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize

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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient value for backward propagation, that is, the output gradient of the previous layer, corresponding to gradOutput in the formula.</td>
      <td><ul><li>The data type and shape must be the same as those of y.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with mask.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,N,S1,S2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Output value of the softmax function, corresponding to y in the formula.</td>
      <td><ul><li>The data type and shape must be the same as those of gradOutput.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with mask.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,N,S1,S2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Masks the computation result. This is the mask in the formula.</td>
      <td>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with y.</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>[B,N,S1,S2], [B,1,S1,S2], [1,N,S1,S2], [1,1,S1,S2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scales the output, corresponding to scale in the formula.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fixTriuMask</td>
      <td>Input</td>
      <td>Indicates whether to generate the upper triangular mask tensor in the operator.</td>
      <td>Only false</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Gradient of y, i.e., out in the formula.</td>
      <td>The data type and shape must be the same as those of gradOutput and y.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,N,S1,S2]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be allocated on the Device.</td>
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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>gradOutput, y, mask, and the output out are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types, shapes, and data formats of gradOutput, y, mask, and out are out of the supported range.</td>
    </tr>
    <tr>
      <td>The shapes and dtypes of gradOutput, y, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>gradOutput and y do not satisfy the broadcast relationship with mask.</td>
    </tr>
    <tr>
      <td>fixTriuMask is not false.</td>
    </tr>
  </tbody>
  </table>

## aclnnScaledMaskedSoftmaxBackward

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
      <td>Address of the workspace memory requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory requested on the Device side, obtained by the first-phase API aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
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

- Deterministic computation
  - aclnnScaledMaskedSoftmaxBackward defaults to a non-deterministic implementation. You can enable deterministic computation by calling aclrtCtxSetSysParamOpt.

- The range of the last axis S2 is (0, 4096].
- The shape of mask can differ from gradOutput and y in the first two dimensions, but must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md).

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scaled_masked_softmax_backward.h"

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
  // Boilerplate, resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
  // input
  std::vector<float> gradOutputHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> yHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<char> maskHostData = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
  std::vector<float> outHostData(16, 0);
  std::vector<int64_t> gradOutputShape = {2, 2, 2, 2};
  std::vector<int64_t> yShape = {2, 2, 2, 2};
  std::vector<int64_t> maskShape = {2, 2, 2, 2};
  std::vector<int64_t> outShape = {2, 2, 2, 2};
  void *gradOutputDeviceAddr = nullptr;
  void *yDeviceAddr = nullptr;
  void *maskDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *gradOutput = nullptr;
  aclTensor *y = nullptr;
  aclTensor *mask = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  double scale = 1.0f;
  bool fixTriuMask = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  // aclnnScaledMaskedSoftmaxBackward
  ret = aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize(gradOutput, y, mask, scale, fixTriuMask, out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnScaledMaskedSoftmaxBackward
  ret = aclnnScaledMaskedSoftmaxBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnScaledMaskedSoftmaxBackward failed. ERROR: %d\n", ret);
            return ret);

  // 4. (boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result on device memory to host memory. Modify it based on the specific API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(y);
  aclDestroyTensor(mask);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(maskDeviceAddr);
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
