# aclnnUnique2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:34:33.132Z -->

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

Deduplicates the input tensor self and returns the unique elements in self. This is an enhancement of the unique function. The new return value countsOut indicates the number of occurrences of the elements in valueOut appearing in the input self, controlled by the returnCounts parameter.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnUnique2GetWorkspaceSize` first to obtain the workspace size required for the computation and the executor that contains the calculation process of the operator, and then call `aclnnUnique2` to perform the computation.

```cpp
aclnnStatus aclnnUnique2GetWorkspaceSize(
  const aclTensor* self,
  bool             sorted,
  bool             returnInverse,
  bool             returnCounts,
  aclTensor*       valueOut,
  aclTensor*       inverseOut,
  aclTensor*       countsOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnUnique2(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnUnique2GetWorkspaceSize

* **Parameters**

  <table style="undefined;table-layout: fixed; width: 1403px"><colgroup>
  <col style="width: 128px">
  <col style="width: 122px">
  <col style="width: 299px">
  <col style="width: 171px">
  <col style="width: 292px">
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
      <td>Target tensor to be deduplicated.</td>
      <td>-</td>
      <td>BOOL, FLOAT, FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sorted</td>
      <td>Input</td>
      <td>Indicates whether valueOut is sorted in ascending order.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnInverse</td>
      <td>Input</td>
      <td>Indicates whether to return the indices of elements in the input in valueOut.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnCounts</td>
      <td>Input</td>
      <td>Indicates whether to return the count of each unique element in valueOut appearing in the original input tensor.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueOut</td>
      <td>Output</td>
      <td>First output tensor, storing the unique elements in the input tensor. Only 1-D shape is supported, and the number of elements must be less than or equal to that of self.</td>
      <td>-</td>
      <td>BOOL, FLOAT, FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inverseOut</td>
      <td>Output</td>
      <td>Second output tensor. Meaningful only when returnInverse is True. Returns the position indices of elements in self appearing in valueOut, with the same shape as self.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>countsOut</td>
      <td>Output</td>
      <td>Third output tensor. Meaningful only when returnCounts is True. Returns the counts of elements in valueOut appearing in self, with the same shape as valueOut.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be requested by the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the calculation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

* **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).
    The first-phase API performs input parameter validation and reports an error in the following cases:

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
        <td>The data type of self or valueOut is not within the supported range.</td>
      </tr>
      <tr>
        <td>self is a non-contiguous tensor.</td>
      </tr>
      <tr>
        <td>returnInverse is True, and the shape of inverseOut is inconsistent with the shape of self.</td>
      </tr>
      <tr>
        <td>returnCounts is True, and the shape of countsOut is inconsistent with the shape of valueOut.</td>
      </tr>
    </tbody>
    </table>

## aclnnUnique2

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
      <td>Address of the workspace memory requested by the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested by the Device, obtained by the first-phase API aclnnUnique2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the calculation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

* **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnUnique2 defaults to a deterministic implementation.
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
      * Due to differences in the deduplication algorithm implementation, when all of the following conditions are met, the operator ignores the value of the sorted input parameter and always sorts the output results in ascending order:
          * self is a 1D input.
          * The data type of self is one of the following: FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16.
      * Due to differences in the deduplication algorithm implementation, the inverseOut output of the operator is meaningless when all of the following conditions are met:
          - returnInverse is set to false.
          - self is a 1D input.
          - The data type of self is one of the following: FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: If the input self contains 0, the operator output may contain both positive 0 and negative 0 instead of a single 0.
  <!-- end id8 -->
  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: If the amount of data in self exceeds 200 million, execution takes a long time and may time out.
  <!-- end id9 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unique2.h"

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

// Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
  shape.data(), shape.size(), *deviceAddr);
                            return 0;
  }

int main() {
// 1. (Boilerplate) device/stream initialization. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. The custom construction needs to be based on the interface definition of the API.
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> valueShape = {8};
  std::vector<int64_t> inverseShape = {8};
  std::vector<int64_t> countsShape = {8};
  void* selfDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* inverseDeviceAddr = nullptr;
  void* countsDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valueOut = nullptr;
  aclTensor* inverseOut = nullptr;
  aclTensor* countsOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> valueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> inverseHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> countsHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  bool sorted = false;
  bool returnInverse = false;
  bool returnCounts = false;

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

// 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnUnique2.
  ret = aclnnUnique2GetWorkspaceSize(self, sorted, returnInverse, returnCounts, valueOut, inverseOut, countsOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnique2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
  ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
  // Call the second-phase API of aclnnUnique2.
  ret = aclnnUnique2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnique2 failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values. Copy the result from device memory to the host. Modify the code based on the interface definition of the specific API.
  auto size = GetShapeSize(valueShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valueDeviceAddr, size * sizeof(float),
  ACL_MEMCPY_DEVICE_TO_HOST);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
  LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

// 6. Destroy the aclTensor and aclScalar. Modify the code based on the interface definition of the specific API.
  aclDestroyTensor(self);
  aclDestroyTensor(valueOut);
  aclDestroyTensor(inverseOut);
  aclDestroyTensor(countsOut);

// 7. Release device resources. Modify the code based on the interface definition of the specific API.
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
