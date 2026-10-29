# aclnnScatterList

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:52:38.883Z -->

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

- Description: Writes values in updates as a whole segment to the corresponding tensor in the var tensor list, starting at the position given by indice along the dimension specified by axis.

- Formula:

  Let B be the number of tensors in the varRef list, `i` the list index (`0 <= i < B`), `start` the start position for writing to the `i`-th tensor, and `len` the write length. Then:

  $$
  var[i][\dots, start + k, \dots] = updates[i][\dots, k, \dots] \quad , \quad 0 \le k < len
  $$

  In the formula, the ellipsis denotes the remaining dimensions other than the axis dimension, and writes are performed along the dimension specified by axis. `start` and `len` are given by indice:

  - When indice is 1-dimensional, `start = indice[i]`, and `len` is the size of updates along the axis dimension (tensors have the same write length).
  - When indice is 2-dimensional, `start = indice[i][0]`, and `len = indice[i][1]` (tensors can have different write lengths).

  The optional input maskOptional gates whether to perform the write for each list element. reduce currently supports only "update", that is, direct overwrite.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnScatterListGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnScatterList" API to perform the computation.

```Cpp
aclnnStatus aclnnScatterListGetWorkspaceSize(
  aclTensorList*       varRef,
  const aclTensor*     indice,
  const aclTensor*     updates,
  const aclTensor*     maskOptional,
  char*                reduceOptional,
  int64_t              axis,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnScatterList(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  aclrtStream       stream)
```

## aclnnScatterListGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1523px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 120px">
  <col style="width: 143px">
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
      <td>varRef (aclTensorList*)</td>
      <td>Input/Output</td>
      <td>Tensor list to be updated.</td>
      <td>The shape of each tensor in the list satisfies the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with updates.<br>The data type of each tensor must be the same as that of updates.</td>
      <td>INT8, INT16, INT32, UINT8, UINT16, UINT32, FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indice (aclTensor*)</td>
      <td>Input</td>
      <td>Position tensor that specifies the write start position (1-D) or the write start position and length (2-D) for each element of the varRef list.<br>1-D: indice[i] specifies the start position along axis for writing the i-th tensor, and the write length is the size of updates along axis.<br>2-D: indice[i][0] specifies the start position, and indice[i][1] specifies the write length.</td>
      <td>-</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>updates (aclTensor*)</td>
      <td>Input</td>
      <td>Value tensors to update var with.</td>
      <td>The data type must be the same as that of the tensors in var.<br>Shapes must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with the shapes of the tensors in var.</td>
      <td>INT8, INT16, INT32, UINT8, UINT16, UINT32, FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>maskOptional(aclTensor*)</td>
      <td>Optional parameter</td>
      <td>Mask tensor.</td>
      <td>Only 0 and 1 are supported: 0 indicates that the list element is not written, and 1 indicates that the write is performed. The behavior is undefined for other values.<br>The value is on the device and cannot be verified by the first-phase API; the caller must ensure its validity.</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>reduceOptional (char*)</td>
      <td>Attribute</td>
      <td>Type of the reduction operation to apply.</td>
      <td>Currently only "update" is supported.</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Attribute</td>
      <td>The dimension to scatter along.</td>
      <td>The default value is -2.</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the device.</td>
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
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input tensor or tensorList is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of tensors in varRef is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data type of indice is out of the supported range.</td>
    </tr>
    <tr>
      <td>The value of reduceOptional is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data type of tensors in varRef is inconsistent with that of updates.</td>
    </tr>
  </tbody></table>

## aclnnScatterList

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnScatterListGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnScatterList defaults to a non-deterministic implementation. You can enable deterministic computation through aclrtCtxSetSysParamOpt.

- <term>Ascend 950PR&950DT products</term>: The shapes of all inputs must meet the following requirements. Otherwise, the first-phase API returns 561002. Let B be the number of tensors in the varRef list.
  - varRef: Each tensor in the list must have the same shape, and the number of dimensions of each tensor must be greater than or equal to 1.
  - updates: The number of dimensions equals that of a single tensor in varRef plus 1; the size of the first dimension equals B; the size of the axis dimension is not greater than the corresponding dimension of varRef; the remaining dimensions are the same as those of varRef.
  - indice: The shape is 1D or 2D. For a 1D shape of (B,), only the write start position is given; for a 2D shape of (B, 2), both the write start position and length are given.
  - maskOptional: The shape is 1D, and the size of the first dimension equals B.
  - axis: After normalization (negative values are converted based on the number of dimensions of updates), the value must fall in the open interval (0, number of dimensions of updates), that is, it cannot point to dimension 0 or be out of bounds. The default value -2 requires that updates have at least 3 dimensions.

- indice semantics and value range (values are guaranteed by the caller and not validated by the operator): indice specifies the **write start position**, not the scatter target index. The write covers a contiguous range starting from this position, and the entire range must stay within the axis dimension.
  - 1D: `indice[i]` is the write start position of the i-th tensor, and the write length is the size of updates along axis.
  - 2D: `indice[i][0]` is the start position, and `indice[i][1]` is the write length, which should not exceed the size of updates along axis.

- maskOptional value range (guaranteed by the caller and not validated by the operator): only 0 and 1 are supported. 0 indicates that no write is performed on the list element, 1 indicates that the write is performed, and the behavior for other values is undefined.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_list.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: initialization.
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
  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct custom inputs and outputs based on the API.
  std::vector<int64_t> varShape = {5, 3, 4};
  std::vector<int64_t> indicesShape = {1, 2};
  std::vector<int64_t> updatesShape = {1, 5, 3, 4};

  void* varDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;

  aclTensor* varTensor = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;

  std::vector<float> varHostData(5 * 3 * 4, 0);
  std::vector<int64_t> indicesHostData = {0, 1};
  std::vector<float> updatesHostData(1 * 5 * 3 * 4, 10);

  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &varTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* varArray[] = {varTensor};
  auto varTensorList = aclCreateTensorList(varArray, 1);

  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the aclnnScatterList first-phase API.
  ret = aclnnScatterListGetWorkspaceSize(varTensorList, indices, updates, nullptr, "update", -2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterListGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnScatterList.
  ret = aclnnScatterList(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterList failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result from device memory to host memory.
  PrintOutResult(varShape, &varDeviceAddr);

  // 6. Release the aclTensor.
  aclDestroyTensorList(varTensorList);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);

  // 7. Release the device resources.
  aclrtFree(varDeviceAddr);
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
