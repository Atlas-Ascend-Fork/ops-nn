# aclnnUniqueConsecutive

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:05:17.931Z -->

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

Description: Removes consecutive duplicate elements. If dim is specified, removes consecutive duplicate tensors along the corresponding dimension.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call `aclnnUniqueConsecutiveGetWorkspaceSize` to obtain the workspace size required for the computation and the executor containing the computation process. Then, call `aclnnUniqueConsecutive` to perform the computation.

```cpp
aclnnStatus aclnnUniqueConsecutiveGetWorkspaceSize(
  const aclTensor* self,
  bool             returnInverse,
  bool             returnCounts,
  int64_t          dim,
  aclTensor*       valueOut,
  aclTensor*       inverseOut,
  aclTensor*       countsOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnUniqueConsecutive(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnUniqueConsecutiveGetWorkspaceSize

* **Parameters**

  <table style="undefined;table-layout: fixed; width: 1391px"><colgroup>
  <col style="width: 128px">
  <col style="width: 122px">
  <col style="width: 300px">
  <col style="width: 156px">
  <col style="width: 294px">
  <col style="width: 106px">
  <col style="width: 136px">
  <col style="width: 149px">
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
      <td>Input tensor.</td>
      <td>Supports the empty tensor use case.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL, BFLOAT16</td>
      <td>ND</td>
      <td>≤8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>returnInverse (bool)</td>
      <td>Input</td>
      <td>Specifies whether to return the indices of elements in self relative to the corresponding elements in valueOut. Indices are returned when True and not returned when False.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnCounts (bool)</td>
      <td>Input</td>
      <td>Specifies whether to return the number of consecutive occurrences of each element in valueOut within self. Returned when True; not returned when False.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>Dimension along which deduplication is performed. Value range: [-self.dim(), self.dim()). A value of 1000 indicates that no dimension is specified (flatten mode, where the input is flattened before deduplication).</td>
      <td>-</td>
      <td>int64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueOut (aclTensor*)</td>
      <td>Output</td>
      <td>First output tensor. Returns the result of removing consecutive duplicate elements.</td>
      <td>Same data type as self</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL, BFLOAT16.</td>
      <td>ND</td>
      <td>Same as self</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inverseOut (aclTensor*)</td>
      <td>Output</td>
      <td>Second output tensor. Meaningful when returnInverse is True. Returns the indices of elements in valueOut corresponding to elements in self.</td>
      <td>The data types of inverseOut and countsOut must be the same.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>Same as valueOut</td>
      <td>-</td>
    </tr>
    <tr>
      <td>countsOut (aclTensor*)</td>
      <td>Output</td>
      <td>Third output tensor. Meaningful when returnCounts is True. Returns the number of consecutive occurrences of each element in valueOut in self.</td>
      <td>The data types of inverseOut and countsOut must be the same.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>Same as valueOut</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:

  Input `self` does not support DOUBLE, COMPLEX64, COMPLEX128, or BOOL types

  Input `returnInverse` supports only false

  Input `dim` supports only 1000.

<!-- end id7 -->

* **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1015px"><colgroup>
  <col style="width: 257px">
  <col style="width: 101px">
  <col style="width: 657px">
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
      <td>When self, valueOut, inverseOut, or countsOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of self or valueOut is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data type of self is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data type of inverseOut or countsOut is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of inverseOut and countsOut are inconsistent.</td>
    </tr>
  </tbody>
  </table>

## aclnnUniqueConsecutive

* **Parameters**

  <table style="undefined;table-layout: fixed; width: 950px"><colgroup>
  <col style="width: 122px">
  <col style="width: 105px">
  <col style="width: 723px">
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
      <td>Memory address of the workspace on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory on the device. The value is obtained by the first-phase API aclnnUniqueConsecutiveGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>The op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

* **Return Value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnUniqueConsecutive defaults to a deterministic implementation.
- Performance:
  - On A2, A3, and training products, the performance is poor or the execution times out when the dimension value of self on dim exceeds 200 million.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unique_consecutive.h"

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

  // Call aclCreateTensor to create an aclTensor.
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
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> valueShape = {8};
  std::vector<int64_t> inverseShape = {4, 2};
  std::vector<int64_t> countsShape = {8};
  void* selfDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* inverseDeviceAddr = nullptr;
  void* countsDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valueOut = nullptr;
  aclTensor* inverseOut = nullptr;
  aclTensor* countsOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 1, 3, 3, 1, 1, 3};
  std::vector<float> valueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> inverseHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> countsHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  bool returnInverse = false;
  bool returnCounts = false;
  int64_t dim = 0;

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the valueOut aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &valueOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the inverseOut aclTensor.
  ret = CreateAclTensor(inverseHostData, inverseShape, &inverseDeviceAddr, aclDataType::ACL_INT64, &inverseOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the countsOut aclTensor.
  ret = CreateAclTensor(countsHostData, countsShape, &countsDeviceAddr, aclDataType::ACL_INT64, &countsOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library APIs.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnUniqueConsecutive.
  ret = aclnnUniqueConsecutiveGetWorkspaceSize(self, returnInverse, returnCounts, dim, valueOut, inverseOut, countsOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueConsecutiveGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnUniqueConsecutive.
  ret = aclnnUniqueConsecutive(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueConsecutive failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronously wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values, copy the result from device memory to host memory, and modify the code based on the specific API definition.
  auto size = GetShapeSize(valueShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valueDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar, and modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valueOut);
  aclDestroyTensor(inverseOut);
  aclDestroyTensor(countsOut);

  // 7.Release device resources, and modify the code based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(inverseDeviceAddr);
  aclrtFree(countsDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
