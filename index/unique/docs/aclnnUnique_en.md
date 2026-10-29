# aclnnUnique

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:00:22.595Z -->

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

Function: Returns the unique elements in the input tensor.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call `aclnnUniqueGetWorkspaceSize` to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call `aclnnUnique` to perform the computation.

```cpp
aclnnStatus aclnnUniqueGetWorkspaceSize(
  const aclTensor* self,
  bool             sorted,
  bool             returnInverse,
  aclTensor*       valueOut,
  aclTensor*       inverseOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnUnique(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnUniqueGetWorkspaceSize

* **Parameters**

  <table style="undefined;table-layout: fixed; width: 1483px"><colgroup>
  <col style="width: 128px">
  <col style="width: 122px">
  <col style="width: 374px">
  <col style="width: 171px">
  <col style="width: 297px">
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
      <td>self</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>BOOL, FLOAT, FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16</td>
      <td>ND</td>
      <td>≤8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sorted</td>
      <td>Input</td>
      <td>Indicates whether to sort valueOut in ascending order. If true, the output is sorted. If false, the output is in random order.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnInverse</td>
      <td>Input</td>
      <td>Indicates whether to return the indices of each element in the input data in valueOut.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueOut</td>
      <td>Output</td>
      <td>First output tensor, containing the unique elements in the input tensor.</td>
      <td></td>
      <td>BOOL, FLOAT, FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>inverseOut</td>
      <td>Output</td>
      <td>Second output tensor, valid when returnInverse is True. Returns the indices at which elements in self appear in valueOut.</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
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
      <td>When self, valueOut, or inverseOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of self or valueOut is out of the supported range.</td>
    </tr>
    <tr>
      <td>self is a non-contiguous tensor.</td>
    </tr>
    <tr>
      <td>returnInverse is True, and the shape of inverseOut is inconsistent with that of self.</td>
    </tr>
    <tr>
      <td>returnInverse is True, and the data type of inverseOut is not INT64.</td>
    </tr>
  </tbody>
  </table>

## aclnnUnique

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnUniqueGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

* **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnUnique defaults to a deterministic implementation.
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
      - Due to differences in deduplication algorithm implementations, when all the following conditions are met, the operator ignores the value of the sorted parameter and always sorts the output in ascending order:
          - self is a 1D input.
          - The data type of self is one of the following: FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: If the input self contains 0, the operator output contains both positive 0 and negative 0 instead of a single 0.
  <!-- end id8 -->

## Example

The sample code is as follows for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unique.h"

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

  // Call aclrtMemcpy to copy data from the host side to the device-side memory.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> valueShape = {8};
  std::vector<int64_t> inverseShape = {8};
  void* selfDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* inverseDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valueOut = nullptr;
  aclTensor* inverseOut = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> valueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> inverseHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  bool sorted = true;
  bool returnInverse = true;

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the valueOut aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &valueOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the inverseOut aclTensor.
  ret = CreateAclTensor(inverseHostData, inverseShape, &inverseDeviceAddr, aclDataType::ACL_INT64, &inverseOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnUnique.
  ret = aclnnUniqueGetWorkspaceSize(self, sorted, returnInverse, valueOut, inverseOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnUnique.
  ret = aclnnUnique(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnique failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values, copy the result on the device memory to the host, and modify it according to the specific API definition.
  auto size = GetShapeSize(valueShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valueDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  std::vector<int64_t> resultData1(size, 0);
  ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), inverseDeviceAddr, size * sizeof(int64_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result1[%ld] is: %ld\n", i, resultData1[i]);
  }

  // 6. Release aclTensor and aclScalar, and modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valueOut);
  aclDestroyTensor(inverseOut);

  // 7. Release device resources, and modify it according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(inverseDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
