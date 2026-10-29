# aclnnScatterNdUpdate

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:56:18.771Z -->

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

Updates the values in the tensor varRef element by element with the values in the tensor updates according to the specified indices.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnScatterNdUpdateGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the computation process of the operator. Then call "aclnnScatterNdUpdate" to execute the computation.

```Cpp
aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(
  aclTensor*       varRef,
  const aclTensor* indices,
  const aclTensor* updates,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScatterNdUpdate(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnScatterNdUpdateGetWorkspaceSize

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
        <td>varRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Tensor to be updated.</td>
        <td>The data type must be the same as that of updates.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT64, BOOL, INT8, UINT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices (aclTensor*)</td>
        <td>Input</td>
        <td>Tensor of specified indices.</td>
        <td>The index data in indices cannot be out of bounds.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>updates (aclTensor*)</td>
        <td>Input</td>
        <td>The update data tensor.</td>
        <td>The data type must be the same as that of varRef.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT64, BOOL, INT8, UINT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
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
  <!-- npu="A3,910b" id8 -->
  <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: FLOAT8_E4M3FN, FLOAT8_E5M2, and HIFLOAT8 data types are not supported on this model.
  <!-- end id8 -->
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
        <td>varRef, indices, or updates is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002/561103</td>
        <td>The shapes of varRef, indices, and updates do not meet requirements.</td>
      </tr>
      <tr>
        <td>The data types of varRef, indices, and updates are not in the supported range.</td>
      </tr>
    </tbody></table>

## aclnnScatterNdUpdate

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
        <td>Workspace memory address requested on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnScatterNdUpdateGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor containing the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for task execution.</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Input shape constraints:
  - varRef supports up to 8 dimensions, and varRef.rank is [1,8].
  - indices has at least 2 dimensions. The rank of the last dimension of indices indicates the number of index dimensions. The rank value ranges from 1 to 7 and must satisfy rank <= varRef.rank, that is, the number of index dimensions cannot exceed the number of dimensions of varRef.
  - Assume the size of the last dimension of indices is a, then the shape of updates equals the shape of indices excluding the last dimension plus the shape of varRef excluding the first a dimensions. For example: if the shape of varRef is (4, 5, 6) and the shape of indices is (3, 2), then the shape of updates must be (3, 6).

- Non-contiguous tensor support:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: varRef supports the scenario where the first axis is non-contiguous (that is, the stride of dimension 0 is greater than the expected value under the contiguous layout, and all other dimensions are contiguous). The operator completes the update directly on the non-contiguous view, without first converting varRef to a contiguous tensor.

  <!-- end id7 -->

- Deterministic computation:

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: Defaults to a non-deterministic implementation. Deterministic computation can be enabled by calling `aclrtSetSysParamOpt`.
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: Defaults to a deterministic implementation.
  <!-- end id10 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_nd_update.h"

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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction according to the API.
  std::vector<int64_t> varRefShape = {2, 3, 7};
  std::vector<int64_t> indicesShape = {2, 2};
  std::vector<int64_t> updatesShape = {2, 7};
  void* varRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* varRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> varRefHostData = {0.3816, 0.3939, 0.8474, 0.1652, 0.6049, 0.3315, 0.4954,
                                     0.3284, 0.7060, 0.4359, 0.6514, 0.9476, 0.4708, 0.0656,
                                     0.9652, 0.9512, 0.6452, 0.1981, 0.4159, 0.9575, 0.1516,
                                     0.4987, 0.9107, 0.6635, 0.4119, 0.4845, 0.5558, 0.2749,
                                     0.6230, 0.1180, 0.2400, 0.9971, 0.4093, 0.5561, 0.4023,
                                     0.6612, 0.4109, 0.8470, 0.9733, 0.6947, 0.7980, 0.7957};
  std::vector<int64_t> indicesHostData = {5, 0, 1, 5};
  std::vector<float> updatesHostData = {0.7804, 0.3411, 0.6674, 0.8468, 0.6679, 0.5549, 0.9893,
                                    0.2086, 0.2473, 0.5110, 0.4549, 0.3113, 0.8490, 0.9217};
  // Create varRef aclTensor.
  ret = CreateAclTensor(varRefHostData, varRefShape, &varRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScatterNdUpdate.
  ret = aclnnScatterNdUpdateGetWorkspaceSize(varRef, indices, updates, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterNdUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnScatterNdUpdate.
  ret = aclnnScatterNdUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterNdUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device memory to the host memory, and modify it based on the specific API definition.
  auto size = GetShapeSize(varRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar, and modify it based on the specific API definition.
  aclDestroyTensor(varRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  // 7. Release device resources, and modify it based on the specific API definition parameters.
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
