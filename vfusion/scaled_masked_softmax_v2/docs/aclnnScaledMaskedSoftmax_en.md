# aclnnScaledMaskedSoftmax

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:01:50.389Z -->

## Applicable Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

- Description: Scales the input x by scale, applies the mask, and then performs softmax to obtain the output.
- Formula:

  $$
  y = Softmax((scale * x) * mask, dim = -1)
  $$

  $$
  Softmax(X_i) ={e^{X_i - max(X, dim=-1)} \over \sum{e^{X_i - max(X, dim=-1)}}}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnScaledMaskedSoftmaxGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the "aclnnScaledMaskedSoftmax" API to perform the computation.

```Cpp
aclnnStatus aclnnScaledMaskedSoftmaxGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* mask,
    double scale,
    bool fixTriuMask,
    aclTensor*       y,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScaledMaskedSoftmax(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnScaledMaskedSoftmaxGetWorkspaceSize

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
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input data, which is x in the formula.</td>
      <td>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with mask.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,N,S1,S2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>The mask applied to x, which is mask in the formula.</td>
      <td>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with x.</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>[B,N,S1,S2], [B,1,S1,S2], [1,N,S1,S2], [1,1,S1,S2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scaling factor of the data, which is scale in the formula.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fixTriuMask</td>
      <td>Input</td>
      <td>Indicates whether an upper triangular mask Tensor needs to be generated in the operator.</td>
      <td>false only</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Softmax result of scaledMaskedX, which is y in the formula.</td>
      <td>The data type and shape must be the same as those of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[B,N,S1,S2]</td>
      <td>×</td>
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
      <td>Returns the op executor, which contains the operator computation process.</td>
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
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>x or y is a null pointer.</td>
    </tr>
    <tr>
      <td>When fixedTriuMask is false, mask is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of x, mask, and y are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x, mask, or y is not 4D.</td>
    </tr>
    <tr>
      <td>The fourth dimension of x is greater than 4096 or equal to 0.</td>
    </tr>
    <tr>
      <td>The shapes of x and y are different.</td>
    </tr>
    <tr>
      <td>mask cannot be broadcast to the shape of x.</td>
    </tr>
  </tbody>
  </table>

## aclnnScaledMaskedSoftmax

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnScaledMaskedSoftmaxGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnScaledMaskedSoftmax defaults to a deterministic implementation.

- The fourth dimension of x must be within the range of $(0,4096]$.
- The shape of mask can differ from x in the first two dimensions, but must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md).

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scaled_masked_softmax.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization.
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

  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  // input
  std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<char> maskHostData = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
  std::vector<float> yHostData(16, 0);
  std::vector<int64_t> xShape = {2, 2, 2, 2};
  std::vector<int64_t> maskShape = {2, 2, 2, 2};
  std::vector<int64_t> yShape = {2, 2, 2, 2};
  void *xDeviceAddr = nullptr;
  void *maskDeviceAddr = nullptr;
  void *yDeviceAddr = nullptr;
  aclTensor *x = nullptr;
  aclTensor *mask = nullptr;
  aclTensor *y = nullptr;

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // attr
  float scale = 1.0f;
  bool triuMask = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  // aclnnScaledMaskedSoftmax
  ret = aclnnScaledMaskedSoftmaxGetWorkspaceSize(x, mask, scale, triuMask, y, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnScaledMaskedSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // aclnnScaledMaskedSoftmax
  ret = aclnnScaledMaskedSoftmax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnScaledMaskedSoftmax failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Synchronize and wait until the task execution ends.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output values and copy the result on the device to the host. Modify the code based on the API definition of the specific API.
  PrintOutResult(yShape, &yDeviceAddr);

  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition of the specific API.
  aclDestroyTensor(x);
  aclDestroyTensor(mask);
  aclDestroyTensor(y);

  // 7. Release device resources.
  aclrtFree(xDeviceAddr);
  aclrtFree(maskDeviceAddr);
  aclrtFree(yDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
