# aclnnUniqueDim

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:07:41.079Z -->

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

- Function: Deduplicates the input tensor `self` along a specified dim.
- Example: Assume `self` is

  $$
  \begin{bmatrix}
  \begin{bmatrix}
   2 & 1 & 2
  \end{bmatrix}\\
   \begin{bmatrix}
   1 & 2 & 1
  \end{bmatrix}\\
   \begin{bmatrix}
   2 & 1 & 2
  \end{bmatrix}\\
  \end{bmatrix}
  $$

  If `dim` is 0, `valueOut` is:

  $$
  \begin{bmatrix}
  \begin{bmatrix}
   2 & 1 & 2
  \end{bmatrix}\\
   \begin{bmatrix}
   1 & 2 & 1
  \end{bmatrix}\\
  \end{bmatrix}
  $$

  `inverseOut` is:

  $$
  \begin{bmatrix}
   0 & 1 & 0
  \end{bmatrix}
  $$

  `countsOut` is:

  $$
  \begin{bmatrix}
   1 & 2
  \end{bmatrix}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call `aclnnUniqueDimGetWorkspaceSize` to obtain input parameters and calculate the required workspace size, then call `aclnnUniqueDim` to perform the computation.

```cpp
aclnnStatus aclnnUniqueDimGetWorkspaceSize(
  const aclTensor* self,
  bool             sorted,
  bool             returnInverse,
  int64_t          dim,
  aclTensor*       valueOut,
  aclTensor*       inverseOut,
  aclTensor*       countsOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnUniqueDim(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
  ```

## aclnnUniqueDimGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1568px"><colgroup>
  <col style="width: 126px">
  <col style="width: 121px">
  <col style="width: 327px">
  <col style="width: 283px">
  <col style="width: 327px">
  <col style="width: 104px">
  <col style="width: 134px">
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
      <td>The self in the example.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, DOUBLE, BOOL, BFLOAT16.</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sorted</td>
      <td>Input</td>
      <td>Indicates whether the returned output valueOut is sorted.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnInverse</td>
      <td>Input</td>
      <td>Indicates whether to return the position indices in valueOut of each element of self along the dim axis. Returned when True, not returned when False.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>The dim in the example.</td>
      <td>Specifies the dimension for the deduplication operation. The data type is INT64. The value range is [-self.dim(), self.dim()).</td>
      <td>INT64</td>
      <td>ND</td>
      <td></td>
      <td>√</td>
    </tr>
    <tr>
      <td>valueOut</td>
      <td>Output</td>
      <td>valueOut in the example, indicating the deduplication result.</td>
      <td>Data type is the same as that of self.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inverseOut</td>
      <td>Output</td>
      <td>inverseOut in the example indicates the position index of the corresponding element in valueOut for each element of self along dim.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>countsOut</td>
      <td>Output</td>
      <td>countsOut in the example indicates the number of occurrences of each element in valueOut in self.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1035px"><colgroup>
  <col style="width: 262px">
  <col style="width: 103px">
  <col style="width: 670px">
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
      <td>Any of self, valueOut, inverseOut, or countsOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of self is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of inverseOut and countsOut are not INT64.</td>
    </tr>
    <tr>
      <td>The data types of self and valueOut are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of dimensions of the shape of self is greater than 8.</td>
    </tr>
    <tr>
      <td>The value of dim is not in the range [-self.dim(), self.dim()).</td>
    </tr>
  </tbody>
  </table>

## aclnnUniqueDim

- **Parameters**

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnUniqueDimGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnUniqueDim defaults to a deterministic implementation.
- Performance:
  - On A2, A3, and training products, if the dimension size of self along dim exceeds 200 million, performance is poor or even the execution times out.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "math.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_unique_dim.h"

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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> valueShape = {3,2};
  std::vector<int64_t> inverseShape = {4, 2};
  std::vector<int64_t> countsShape = {3};
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
  bool sorted = false;
  bool returnInverse = false;
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

  // 3. Call the CANN operator library API. Replace it with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnUniqueDim.
  ret = aclnnUniqueDimGetWorkspaceSize(self, sorted, returnInverse, dim, valueOut, inverseOut, countsOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnUniqueDim.
  ret = aclnnUniqueDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueDim failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronously wait for the task to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(valueShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valueDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valueOut);
  aclDestroyTensor(inverseOut);
  aclDestroyTensor(countsOut);

   // 7. Release device resources. Modify it based on the specific API definition.
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
