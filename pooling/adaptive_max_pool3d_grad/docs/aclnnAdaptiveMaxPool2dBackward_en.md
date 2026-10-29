# aclnnAdaptiveMaxPool2dBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:54:05.408Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description:
  Backward propagation of the forward adaptive max pooling. Backfills the gradient to the coordinates of the maximum value in each adaptive window, with accumulation at identical coordinates.
- Forward formula:
  For an input self with dimensions $[N,C,H,W]$, where N (Batch) denotes the batch size, C (Channels) denotes the feature map channels, H (Height) denotes the feature map height, and W (Width) denotes the feature map width, and outputSize is $[H_o,W_o]$, the output has dimensions $[N,C,H_o,W_o]$ and the indices have dimensions $[N,C,H_o,W_o]$, where $m$ denotes the index in the output height dimension (value range $[0, H_o)$) and $n$ denotes the index in the output width dimension (value range $[0, W_o)$). The formula for each element $(m,n)$ in the corresponding tensor is as follows:
  $
  H_{left}^m = \lfloor(m*H)/H_o\rfloor \\
  H_{right}^m = \lceil(m*H)/H_o\rceil  \\
  W_{left}^n = \lfloor(n*W)/W_o\rfloor \\
  W_{right}^n = \lceil(n*W)/W_o\rceil  \\
  output(N,C,m,n) = \mathop{\max}\limits_{j\in[H_{left}^m,H_{right}^m],k\in[W_{left}^n,W_{right}^n]} input(N,C,j,k) \\
  indices(N,C,m,n) = \mathop{\arg\max}\limits_{j\in[H_{left}^m,H_{right}^m],k\in[W_{left}^n,W_{right}^n]} input(N,C,j,k)
  $

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnAdaptiveMaxPool2dBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclTensor   *self,
  const aclTensor   *indices,
  aclTensor         *gradInput,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAdaptiveMaxPool2dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize

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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient of the current node.</td>
      <td>Consistent with the forward output shape.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Forward input tensor.</td>
      <td>Data type consistent with gradOutput.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Index positions of the maximum elements in the forward input.</td>
      <td>shape consistent with gradOutput.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Output tensor of backward propagation.</td>
      <td>Shape is consistent with self.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Op executor that contains the operator computation flow.</td>
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
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>gradOutput, self, or indices is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of gradOutput, self, indices, or gradInput is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data format of gradOutput, self, indices, or gradInput is not in the supported range.</td>
    </tr>
    <tr>
      <td>The input or output shape is not 3D or 4D.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput and indices are inconsistent, and the shapes of self and gradInput are inconsistent.</td>
    </tr>
    <tr>
      <td>height * width > max int32, which exceeds the representation range of indices.</td>
    </tr>
  </tbody>
  </table>

## aclnnAdaptiveMaxPool2dBackward

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained through the first-phase API aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: aclnnAdaptiveMaxPool2dBackward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnAdaptiveMaxPool2dBackward defaults to a deterministic implementation.
  <!-- end id12 -->

- In non-divisible scenarios (where the dimension values of the last two dimensions of self modulo the dimension values of the last two dimensions of gradOutput are not zero), the shape must not exceed 2^24. In divisible scenarios, this restriction does not apply.
- Example of a divisible scenario: self = [a, b, c, d], gradOutput = [a, b, e, f]. The scenario is divisible when c % e is 0 and d % f is 0.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_max_pool2d_backward.h"

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
  // Call aclrtMalloc to apply for memory on the Device
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy Host data to the Device memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization. For details, see the acl API documentation
  // Set deviceId based on the actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API
  std::vector<int64_t> gradOutShape = {1, 1, 1, 1};
  std::vector<int64_t> selfShape = {1, 1, 2, 2};
  std::vector<int64_t> indicesShape = {1, 1, 1, 1};
  std::vector<int64_t> gradInShape = {1, 1, 2, 2};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* gradIn = nullptr;
  std::vector<float> gradOutHostData = {0.4757};
  std::vector<float> selfHostData = {0.0850, -0.5147, -0.0212, -0.5654};
  std::vector<int32_t> indicesHostData = {6};
  std::vector<float> gradInHostData = {0, 0, 0, 0};

  // Create gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create gradIn aclTensor
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnAdaptiveMaxPool2dBackward API call example
  // 3. Call the CANN operator library API, and change it to the specific API name
  // Call the first-phase API aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize
  ret = aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(gradOut, self, indices, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnAdaptiveMaxPool2dBackward
  ret = aclnnAdaptiveMaxPool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the actual API definition.
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradIn result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it based on the actual API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyTensor(gradIn);

  // 7. Release the device resources. Modify it based on the actual API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
