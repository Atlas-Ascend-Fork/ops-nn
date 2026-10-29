# aclnnScatterDiv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:42:14.917Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Implements the function compatible with tf.compat.v1.scatter_div. The values in tensor updates are divided slice by slice into the slices of tensor varRef based on the specified index tensor indices. If more than one updates value is applied to the same slice of varRef, these values are successively divided on that slice. The rule is as follows:

$$
varRef[indices[i],...] = varRef[indices[i],...] / updates[i,...]
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnScatterDivGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call the "aclnnScatterDiv" API to execute the computation.

```cpp
aclnnStatus aclnnScatterDivGetWorkspaceSize(
  aclTensor       *varRef,
  const aclTensor *indices,
  const aclTensor *updates,
  bool             useLocking,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnScatterDiv(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnScatterDivGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1382px"><colgroup>
  <col style="width: 126px">
  <col style="width: 120px">
  <col style="width: 339px">
  <col style="width: 168px">
  <col style="width: 245px">
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
      <td>varRef</td>
      <td>Input/Output</td>
      <td>Input varRef in the formula, the initial tensor to be updated, updated in place.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Input indices in the formula, index positions to be updated.</td>
      <td>The value range of indices is [0, varRef.shape[0]) (varRef.shape[0] is the size of the first dimension of varRef).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Input updates in the formula, which are the update values to be divided into varRef.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>useLocking</td>
      <td>Input</td>
      <td>Whether to lock updates. This parameter does not take effect.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator calculation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

   The first-phase API validates input parameters. If any of the following error codes is returned, the cause is as follows:

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
      <td>The input varRef, indices, or updates is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of varRef, indices, and updates are not in the supported range.</td>
    </tr>
    <tr>
      <td>The dtypes of varRef and updates are inconsistent.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shape of updates does not meet the corresponding shape constraints.</td>
    </tr>
  </tbody>
  </table>

## aclnnScatterDiv

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnScatterDivGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The data types of varRef and updates must be the same.
- Shape constraint: updates.shape = indices.shape + varRef.shape[1:].
- The size of the first dimension of varRef cannot exceed INT32_MAX.
- Value range of updates: if the data type is an integer type (INT32, INT8, UINT8), the elements in updates cannot be 0.
- Deterministic computation:
  - aclnnScatterDiv defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_div.h"

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
  // Boilerplate: initialize resources.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Fill in the deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> varRefShape = {3, 2};
  std::vector<int64_t> indicesShape = {2};
  std::vector<int64_t> updatesShape = {2, 2};
  void* varRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* varRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> varRefHostData = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> indicesHostData = {0, 2};
  std::vector<float> updatesHostData = {10, 20, 30, 40};
  bool useLocking = false;
  // Create the varRef aclTensor.
  ret = CreateAclTensor(varRefHostData, varRefShape, &varRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScatterDiv.
  ret = aclnnScatterDivGetWorkspaceSize(varRef, indices, updates, useLocking, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterDivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnScatterDiv.
  ret = aclnnScatterDiv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterDiv failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device memory to the host memory, and modify the code based on the specific API definition.
  auto size = GetShapeSize(varRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor, and modify the code based on the specific API definition.
  aclDestroyTensor(varRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  // 7. Release device resources, and modify the code based on the parameters of the specific API definition.
  aclrtFree(varRefDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
