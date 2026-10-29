# aclnnRepeatInterleaveWithDim

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:37:56.531Z -->

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

- Description: Repeats each element in the tensor along the specified dimension according to the corresponding count in the Tensor repeats.

- Example: Assume the input tensor is [[a, b], [c, d], [e, f]], repeats is ([1, 2, 3]), and dim is 0.
  Then the resulting tensor is ([[a, b], [c, d], [c, d], [e, f], [e, f], [e, f]]).
  Along dim 0, a and b are repeated once, c and d are repeated twice, and e and f are repeated three times.

  Assume the input tensor is ([[a, b], [c, d], [e, f]]). repeats is ([2]), and dim is 0.
  Then the resulting tensor is [ [a, b], [a, b], [c, d], [c, d], [e, f], [e, f]].
  Along dim 0, a and b are repeated twice, c and d are repeated twice, and e and f are repeated twice.
  Note: This use case is equivalent to repeats (2).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnRepeatInterleaveWithDimGetWorkspaceSize` API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the `aclnnRepeatInterleaveWithDim` API to execute the computation.

```Cpp
aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* repeats,
  int64_t          dim,
  int64_t          outputSize,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnRepeatInterleaveWithDim(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnRepeatInterleaveWithDimGetWorkspaceSize

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
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Input tensor whose data is to be duplicated, as described in Function.</td>
        <td>Empty tensors are supported.</td>
        <td>UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, BOOL, FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>repeats (aclTensor*)</td>
        <td>Input</td>
        <td>Number of repetitions.</td>
        <td>Supports empty tensors.<br>Only 0D/1D tensors are supported.<br>If the tensor is 1D, the size of repeats must be 1 or the size of the dim dimension of self.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>Dimension along which elements are repeated.</td>
        <td>Range: [-self.dim(), self.dim() - 1].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputSize(int64_t)</td>
        <td>Input</td>
        <td>Final size of the dim dimension after repetition.</td>
        <td>If repeats contains multiple values, outputSize must be the sum of repeats.<br>If repeats contains only one element, outputSize must be repeats * the size of self along dim.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Output tensor holding the replicated data, as described in Function.</td>
        <td>The data type must be the same as self.<br>If repeats contains multiple values, the size of the out shape along dim equals the sum of all elements in repeats.<br>If repeats has only one element, the size of the out shape along dim equals repeats * the size of the self shape along dim.</td>
        <td>UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, BOOL, FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
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

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 300px">
    <col style="width: 134px">
    <col style="width: 716px">
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
        <td>Null pointer in self, repeats, or out.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data type of self or repeats is not in the supported range.</td>
      </tr>
      <tr>
        <td>The data types of self and out are different.</td>
      </tr>
      <tr>
        <td>repeats is not a 0D/1D tensor.</td>
      </tr>
      <tr>
        <td>The value of dim is not in the range [-number of dimensions of self, number of dimensions of self - 1].</td>
      </tr>
      <tr>
        <td>When repeats is a 1D tensor, the size of repeats is neither 1 nor the size of the dim dimension of self.</td>
      </tr>
      <tr>
        <td>The number of dimensions of self exceeds 8.</td>
      </tr>
      <tr>
        <td>dim is not supported when self is a 0-dimensional tensor.</td>
      </tr>
    </tbody></table>

## aclnnRepeatInterleaveWithDim

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
        <td>Workspace size allocated on the device, obtained from the first-phase API aclnnRepeatInterleaveWithDimGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, containing the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRepeatInterleaveWithDim defaults to a deterministic implementation.
- Input shape constraints: repeats can only be a 0D or 1D tensor. If it is a 1D tensor, the size of repeats must be 1 or the size of the dim dimension of self.
- Input value range constraints: The values in the repeats tensor must be natural numbers.
- Other restrictions: The value of outputSize must meet the following calculation: When repeats contains only one element, outputSize = size of the dim dimension of self * value of repeats; when repeats contains multiple values, outputSize = sum of the values of repeats.

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
  // Call aclrtMemcpy to copy host data to device memory.
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
  // 1. (boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> repeatsShape = {2};
  std::vector<int64_t> outShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* repeats = nullptr;
  aclTensor* out = nullptr;
  int64_t dim = 0;
  int64_t output_size = 3;
  std::vector<int64_t> selfHostData = {3, 4, 5, -3, -4, -5};
  std::vector<int64_t> repeatsHostData = {1, 2};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the repeats aclTensor.
  ret = CreateAclTensor(repeatsHostData, repeatsShape, &repeatsDeviceAddr, aclDataType::ACL_INT64, &repeats);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRepeatInterleaveWithDim.
  ret = aclnnRepeatInterleaveWithDimGetWorkspaceSize(self, repeats, dim, output_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveWithDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRepeatInterleaveWithDim.
  ret = aclnnRepeatInterleaveWithDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveWithDim failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result on device memory to the host. Modify the code according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Destroy the aclTensor and aclScalar. Modify the code according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code according to the specific API definition.
  aclrtFree(selfDeviceAddr);
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
