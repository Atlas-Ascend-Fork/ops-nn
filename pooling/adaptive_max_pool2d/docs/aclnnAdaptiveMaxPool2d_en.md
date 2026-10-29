# aclnnAdaptiveMaxPool2d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:51:24.638Z -->

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

Computes the kernel size for each operation based on the input outputSize, performs 2D max pooling on the input self, and outputs the pooled values out and indices indices. The difference between aclnnAdaptiveMaxPool2d and aclnnMaxPool2d is that you only need to specify the outputSize, and the pooling regions are divided according to the outputSize.

## Function Prototype

Each operator has a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnAdaptiveMaxPool2dGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call "aclnnAdaptiveMaxPool2d" to execute the computation.

```Cpp
aclnnStatus aclnnAdaptiveMaxPool2dGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *outputSize,
  aclTensor         *outputOut,
  aclTensor         *indicesOut,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAdaptiveMaxPool2d(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaptiveMaxPool2dGetWorkspaceSize

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
        <td>Tensor to be computed.</td>
        <td>Same data type as outputOut.</td>
        <td>FLOAT32, FLOAT16, BFLOAT16, DOUBLE</td>
        <td>NCHW, NCL, NHWC</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputSize</td>
        <td>Input</td>
        <td>Specifies the spatial size of the output in the H and W dimensions.</td>
        <td>An INT32 array of length 2. Each element must be greater than or equal to 0.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputOut</td>
        <td>Output</td>
        <td>Result after pooling.</td>
        <td>Same data type as self. Same shape as indicesOut.</td>
        <td>FLOAT32, FLOAT16, BFLOAT16, DOUBLE</td>
        <td>NCHW, NCL, NHWC</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indicesOut</td>
        <td>Output</td>
        <td>Index positions of outputOut elements in the input self.</td>
        <td>Same shape as outputOut.</td>
        <td>INT64</td>
        <td>NCHW, NCL, NHWC</td>
        <td>3-4</td>
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
        <td>Returns the op executor, which contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: returns a status code. For details, see [aclnn Error Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
    </colgroup>
    <thead>
      <tr>
        <th>Error Code</th>
        <th>error code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The passed-in self, outputSize, outputOut, or indicesOut is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>The data type of self is not in the supported range.</td>
      </tr>
      <tr>
        <td>The data types of self and outputOut are inconsistent.</td>
      </tr>
      <tr>
        <td>The data type of indicesOut is not int64.</td>
      </tr>
      <tr>
        <td>The shape of self is not 3D or 4D.</td>
      </tr>
      <tr>
        <td>The size of self on a dimension other than the first is less than 1.</td>
      </tr>
      <tr>
        <td>The shapes of outputOut and indicesOut are inconsistent.</td>
      </tr>
      <tr>
        <td>The size of outputSize is not 2.</td>
      </tr>
      <tr>
        <td>The value of an element in outputSize is less than or equal to 0.</td>
      </tr>
      <tr>
        <td>The shape of outputOut does not match the actual output shape.</td>
      </tr>
      <tr>
        <td>The format of self is not one of NCHW, NHWC, or NCL.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>Internal API validation error, usually caused by input data or attribute specifications out of the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnAdaptiveMaxPool2d

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
      <td>The workspace memory address on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size on the Device side, obtained by the first-phase API aclnnAdaptiveMaxPool2dGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
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

- deterministic computation:
  - aclnnAdaptiveMaxPool2d defaults to a deterministic implementation.

- Shape description:
  - self.shape = (N, C, Hin, Win), (C, Hin, Win), or (N, Hin, Win, C)
  - outputSize = [Hout, Wout]
  - outputOut.shape = (N, C, Hout, Wout), (C, Hout, Wout), or (N, Hout, Wout, C)
  - indicesOut.shape = (N, C, Hout, Wout), (C, Hout, Wout), or (N, Hout, Wout, C)

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_max_pool2d.h"

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
  // Call aclrtMalloc to allocate device-side memory.
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

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {1, 1, 4, 4};
  std::vector<int64_t> outShape = {1, 1, 2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* indDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData = {0, 0, 0, 0.0};
  std::vector<int64_t> indicesHostData = {0, 0, 0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, outShape, &indDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize = {2, 2};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize.data(), arraySize.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Replace with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAdaptiveMaxPool2d.
  ret = aclnnAdaptiveMaxPool2dGetWorkspaceSize(self, outputSize, out, indices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAdaptiveMaxPool2d.
  ret = aclnnAdaptiveMaxPool2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool2d failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result on the device side to the host side. Modify based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  std::vector<int64_t> indicesData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indDeviceAddr,
                    size * sizeof(indicesData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indices[%ld] is: %ld\n", i, indicesData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(indDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
