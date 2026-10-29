# aclnnRepeatInterleaveTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:37:58.498Z -->

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

- Description: For the i-th element of Tensor repeats, i is repeated repeats[i] times.

- Example: Assume that repeats is ([1, 3]). The resulting tensor is tensor([0, 1, 1, 1]).
  For the 0th element 1 of repeats, 0 is repeated once. For the 1st element 3 of repeats, 1 is repeated three times.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnRepeatInterleaveTensorGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnRepeatInterleaveTensor" API to perform the computation.

```Cpp
aclnnStatus aclnnRepeatInterleaveTensorGetWorkspaceSize(
  const aclTensor* repeats,
  int64_t          outputSize,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnRepeatInterleaveTensor(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnRepeatInterleaveTensorGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
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
        <th>Non-contiguous tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>repeats (aclTensor*)</td>
        <td>Input</td>
        <td>Number of repeats.</td>
        <td>Empty tensors are supported (including the shape=[0,] use case).</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputSize (int64_t)</td>
        <td>Input</td>
        <td>Final size of the tensor after repetition.</td>
        <td>outputSize must be equal to the sum of the values of repeats.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>The output tensor after data replication, as described in Function.</td>
        <td>The data type must be the same as that of repeats.<br>The shape dimensions are the same as those of repeats, and the number of elements equals the sum of the values of repeats.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
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

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 300px">
    <col style="width: 134px">
    <col style="width: 716px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Values</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>repeats or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>The data type of repeats is not within the supported range.</td>
      </tr>
      <tr>
        <td>The data types of repeats and out are different.</td>
      </tr>
      <tr>
        <td>repeats is not a 1D tensor or an empty tensor.</td>
      </tr>
      <tr>
        <td>The result of the repeatInterleave computation is inconsistent with the shape of out.</td>
      </tr>
    </tbody></table>

## aclnnRepeatInterleaveTensor

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device, obtained by the first-phase API aclnnRepeatInterleaveTensorGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream on which tasks are executed.</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRepeatInterleaveTensor defaults to a deterministic implementation.
- Input shape constraints: repeats must be a 1D tensor (including the case where shape=[0,]).
- Input value range constraints: The values in the repeats tensor must be natural numbers.
- Other constraints:
  - The value of outputSize must equal the sum of the values of repeats.
  - The value of outputSize cannot exceed 2^22.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat_interleave.h"

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
  // Boilerplate code for resource initialization.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output, which need to be custom-built based on the API.
  std::vector<int64_t> repeatsShape = {3};
  std::vector<int64_t> outShape = {6};
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* repeats = nullptr;
  int64_t output_size = 6;
  aclTensor* out = nullptr;
  std::vector<int64_t> repeatsHostData = {1, 2, 3};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0};

  // Create the repeats aclTensor.
  ret = CreateAclTensor(repeatsHostData, repeatsShape, &repeatsDeviceAddr, aclDataType::ACL_INT64, &repeats);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnRepeatInterleaveTensor first-phase API.
  ret = aclnnRepeatInterleaveTensorGetWorkspaceSize(repeats, output_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnRepeatInterleaveTensor second-phase API.
  ret = aclnnRepeatInterleaveTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it according to the specific API definition.
  aclrtFree(repeatsDeviceAddr);
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
