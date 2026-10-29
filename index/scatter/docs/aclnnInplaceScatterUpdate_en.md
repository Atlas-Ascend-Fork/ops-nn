# aclnnInplaceScatterUpdate

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:40:13.774Z -->

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

Updates the values in tensor `data` one by one with the values in tensor `updates` based on the specified axis `axis` and indices `indices`. This operator uses custom operator semantics and has no corresponding TensorFlow or PyTorch API.

- Example:
  This operator has 3 inputs and one attribute: `data`, `updates`, `indices`, and `axis`. `data` is the tensor to be updated, `updates` is the tensor that stores the update data, `indices` indicates the update positions, and `axis` is the specified update dimension. Depending on whether `indices` is 1D or 2D, there are the following two use cases (when `indices` is 0D, it is equivalent to the 1D use case with shape (1), and axis 0 of `updates` must be 1 in this case):

  **Use case 1:** `indices` is 1D. The i-th element of `indices` corresponds to the i-th batch of `updates` (axis 0 of `updates`) and indicates the start offset on the `axis` dimension when that batch is written into the i-th batch of `data`.

  ```text
  Sample input:
  data:(a, b, c, d)
  if axis = -2: updates:(n, b, e, d), indices[i] + e <= c
  if axis = -1: updates:(n, b, c, e), indices[i] + e <= d
  indices:(n,), n <= a
  ```

      data[i][j][indices[i]+k][l] = updates[i][j][k][l] # if dim = -2
      data[i][j][k][indices[i]+l] = updates[i][j][k][l] # if dim = -1

  **Case 2:** `indices` is 2-dimensional, with size 2 along axis 1. The i-th row of `indices` corresponds to the i-th batch of `updates`: `indices[i][0]` specifies the position along axis 0 of `data` to which the batch is written (different i values can write to different batches of `data`), and `indices[i][1]` specifies the starting offset along the `axis` dimension. The size s of `updates` along the `axis` dimension is the length written each time, and `indices[i][1] + s` must not exceed the size of `data` along the `axis` dimension.

  ```text
  Sample input:
  data:(a, b, c, d)
  if axis = -2: updates:(n, b, e, d), indices[i][1] + e <= c
  if axis = -1: updates:(n, b, c, e), indices[i][1] + e <= d
  indices:(n, 2), indices[i][0] < a
  ```

      data[indices[i][0]][j][indices[i][1]+k][l] = updates[i][j][k][l] # if dim = -2
      data[indices[i][0]][j][k][indices[i][1]+l] = updates[i][j][k][l] # if dim = -1

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnInplaceScatterUpdateGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the computation process of the operator, and then call the "aclnnInplaceScatterUpdate" API to perform the computation.

```Cpp
aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(
    aclTensor*       data,
    const aclTensor* indices,
    const aclTensor* updates,
    int64_t          axis,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceScatterUpdate(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnInplaceScatterUpdateGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1477px"><colgroup>
    <col style="width: 147px">
    <col style="width: 120px">
    <col style="width: 233px">
    <col style="width: 277px">
    <col style="width: 270px">
    <col style="width: 121px">
    <col style="width: 164px">
    <col style="width: 145px">
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
        <td>data (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Tensor to be updated.</td>
        <td>The number of dimensions must be the same as updates. Empty tensors are not supported.</td>
        <td>INT8, UINT8, FLOAT16, FLOAT32, INT32, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
        <td>ND</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices (aclTensor*)</td>
        <td>Input</td>
        <td>Index of update positions. Only non-negative indices are supported, and indices must be within bounds.</td>
        <td>Empty tensors are not supported.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>updates (aclTensor*)</td>
        <td>Input</td>
        <td>Tensor that stores the update data.</td>
        <td>The data type must be the same as that of data, and the number of dimensions of the shape must be the same as that of the data shape. Empty tensors are not supported.</td>
        <td>INT8, UINT8, FLOAT16, FLOAT32, INT32, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
        <td>ND</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>axis (int64_t)</td>
        <td>Input</td>
        <td>Dimension used for scatter.</td>
        <td>Value range: (-data_rank, data_rank), and axis cannot be 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the computation flow of the operator.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="910" id7 -->
    - <term>Atlas training products</term>: The UINT8 and BFLOAT16 data types are not supported.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: Data types such as FLOAT8_E4M3FN, FLOAT8_E5M2, and HIFLOAT8 are only supported on this model.

    <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>data, indices, or updates is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of data, indices, or updates is not supported.</td>
      </tr>
      <tr>
      <td>data and updates have different data types.</td>
      </tr>
      <tr>
      <td>data and updates have different dimensions.</td>
      </tr>
      <tr>
      <td>The dimension of indices is not 0D, 1D, or 2D.</td>
      </tr>
      <tr>
      <td>When the dimension of indices is 0, the 0th axis of updates is not 1.</td>
      </tr>
      <tr>
      <td>data, indices, and updates are empty tensors.</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceScatterUpdate

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>The workspace memory address requested on the Device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory requested on the Device, obtained by the first-phase API aclnnInplaceScatterUpdateGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the computation process of the operator.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to specify the task to be executed.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The 0th dimension of the `updates` shape is the same as that of the `indices` shape.
- If `indices` is a 0-dimensional tensor, the 0th dimension of the `updates` shape is 1.
- The 0th dimension of the `updates` shape is less than or equal to that of the `data` shape.
- The shapes of `updates` and `data` must be the same on all axes except the `axis` axis and axis 0.
- If the shape of `indices` is two-dimensional, axis 1 of the shape must be 2.
- If the data type of `indices` is INT32, DtypeSize=4; if the data type is INT64, DtypeSize=8. IndicesShapeSize is the product of the shape of `indices`. The required UB = IndicesShapeSize * DtypeSize + 224. This API is not supported if the required UB is greater than the total UB size available for the corresponding AI processor version.
- If `indices` contains duplicates, the results at the duplicate positions are not guaranteed.
- Deterministic computation:

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: Defaults to non-deterministic. Deterministic computation can be enabled via `aclrtSetSysParamOpt`.
  <!-- end id9 -->
  <!-- npu="A3,910b,910" id10 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Atlas training products</term>: Defaults to deterministic.
  <!-- end id10 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_update.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  int64_t axis = -2;
  std::vector<int64_t> selfRefShape = {1, 1, 2, 8};
  std::vector<int64_t> indicesShape = {1};
  std::vector<int64_t> updatesShape = {1, 1, 1, 8};
  void* selfRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> selfRefHostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> indicesHostData = {1};
  std::vector<float> updatesHostData = {3, 3, 3, 3, 3, 3, 3, 3};

  // Create the selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceScatterUpdate.
  ret = aclnnInplaceScatterUpdateGetWorkspaceSize(selfRef, indices, updates, axis, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceScatterUpdate.
  ret = aclnnInplaceScatterUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result on the device to the host. Modify the code according to the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code according to the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
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
