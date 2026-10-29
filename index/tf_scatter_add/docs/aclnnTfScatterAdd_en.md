# aclnnTfScatterAdd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:57:07.500Z -->

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

- Description: Implements functionality compatible with tf.compat.v1.scatter_add and tf.compat.v1.scatter_nd_add, adding values from the tensor updates to slices of the tensor varRef according to the specified index tensor indices. If more than one updates value is assigned to the same slice of varRef, these values are accumulated on that slice. The rule is as follows:

$$
varRef[indices[i,...,j],...] = varRef[indices[i,...,j],...] + updates
$$

or

$$
varRef[indices[i,:]] = varRef[indices[i,:]] + updates[i,...]
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnTfScatterAddGetWorkspaceSize" API to obtain the workspace size required for the computation and an executor containing the computation process of the operator, and then call the "aclnnTfScatterAdd" API to execute the computation.

```cpp
aclnnStatus aclnnTfScatterAddGetWorkspaceSize(
  aclTensor       *varRef,
  const aclTensor *indices,
  const aclTensor *updates,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnTfScatterAdd(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnTfScatterAddGetWorkspaceSize

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>varRef</td>
      <td>Input/Output</td>
      <td>Input varRef in the formula, the initial tensor to update.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT8, UINT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Input indices in the formula, the index positions to update.</td>
      <td>-</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Input updates in the formula, representing the update values to be added to varRef.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT8, UINT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the Device.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

   The first-phase API completes input parameter validation. If the following error codes are returned, the corresponding causes are:

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
      <td>varRef, indices, or updates is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of varRef, indices, or updates is out of the supported range.</td>
    </tr>
    <tr>
      <td>varRef and updates have inconsistent dtypes.</td>
    </tr>
    <tr>
      <td>varRef is empty and indices is not an empty tensor.</td>
    </tr>
    <tr>
      <td>The shape of updates does not meet the corresponding shape constraint.</td>
    </tr>
  </tbody>
  </table>

## aclnnTfScatterAdd

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
      <td>The workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size allocated on the device, obtained by the first-phase API aclnnTfScatterAddGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- One of the following constraints must be met:
  - updates.shape = indices.shape + varRef.shape[1:]
  - indices.shape[-1] <= varRef.shape.rank and updates.shape = indices.shape[:-1] + varRef.shape[indices.shape[-1]:]
- Deterministic computation:

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: defaults to a non-deterministic implementation. Deterministic computation can be enabled through `aclrtSetSysParamOpt`.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: default to a deterministic implementation.
  <!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_tf_scatter_add.h"

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
  // Boilerplate code. Initialize resources.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> varRefShape = {3, 2};
  std::vector<int64_t> indicesShape = {2, 1};
  std::vector<int64_t> updatesShape = {2, 1, 2};
  void* varRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* varRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> varRefHostData = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> indicesHostData = {0, 2};
  std::vector<float> updatesHostData = {10, 20, 30, 40};
  // Create varRef aclTensor.
  ret = CreateAclTensor(varRefHostData, varRefShape, &varRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace the API name with the specific one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnTfScatterAdd.
  ret = aclnnTfScatterAddGetWorkspaceSize(varRef, indices, updates, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTfScatterAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnTfScatterAdd.
  ret = aclnnTfScatterAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTfScatterAdd failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from device memory to the host. Modify it based on the specific API definition.
  auto size = GetShapeSize(varRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
  aclDestroyTensor(varRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  // 7. Release device resources. Modify the parameters based on the specific API definition.
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
