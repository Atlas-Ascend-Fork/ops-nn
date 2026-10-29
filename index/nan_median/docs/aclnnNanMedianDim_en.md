# aclnnNanMedianDim

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:31:36.825Z -->

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

  - Description: Ignores NaN and returns the median of the Tensor along the specified dimension and its position.

  - Example:
    - Example 1:

      ```text
      If keepDim is True, the size of the corresponding dimension is set to 1; if False, the corresponding dimension is removed.
      If the shape of self is [2, 3, 4], dim = 1, and keepDim is true, the output shape is [2, 1, 4].
      If the shape of self is [2, 3, 4], dim = 1, and keepDim is false, the output shape is [2, 4].
      ```

    - Example 2:

      ```text
      Example of the output shape
      If the input is
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) with shape [2, 4]
      dim = 0
      keepDim = true
      Then the output is
      valuesOut = tensor([[-1., float('nan'),  3.,  2.]]) with shape [1, 4]
      indicesOut = tensor([[1, 0, 0, 0]]) with shape [1, 4]
      ```

    - Example 3:

      ```text
      If the input is
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) shape is [2, 4]
      dim = 0
      keepDim = false
      then the output is
      valuesOut = tensor([-1., float('nan'),  3.,  2.]) shape is [4]
      indicesOut = tensor([1, 0, 0, 0]) shape is [4]
      ```

    - Example 4:

      ```text
      If the input is
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) shape is [2, 4]
      dim = 1
      keepDim = false
      then the output is
      valuesOut = tensor([2, 2]) shape is [2]
      indicesOut = tensor([3, 3]) shape is [2]
      ```

## Function Prototype

  Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnNanMedianDimGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call the "aclnnNanMedianDim" API to execute the computation.

  ```cpp
  aclnnStatus aclnnNanMedianDimGetWorkspaceSize(
    const aclTensor* self,
    int64_t          dim,
    bool             keepDim,
    aclTensor*       valuesOut,
    aclTensor*       indicesOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
  ```

  ```cpp
  aclnnStatus aclnnNanMedianDim(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
  ```

## aclnnNanMedianDimGetWorkspaceSize

  - **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1800px"><colgroup>
    <col style="width: 140px">
    <col style="width: 127px">
    <col style="width: 268px">
    <col style="width: 418px">
    <col style="width: 387px">
    <col style="width: 134px">
    <col style="width: 171px">
    <col style="width: 155px">
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
        <td></td>
        <td>-</td>
        <td>FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Dimension to reduce.</td>
        <td>Value range: [-self.dim(), self.dim() - 1]</td>
        <td></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepDim</td>
        <td>Input</td>
        <td>Whether to retain the dimensions of the input tensor in the output tensor.</td>
        <td>If true, the size of the corresponding dimension is set to 1. If false, the corresponding dimension is removed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>valuesOut</td>
        <td>Output</td>
        <td>Median values.</td>
        <td>If keepDim is true, the shape must be consistent with the shape of self in all dimensions except dim, and the size on dim must be 1. If keepDim is false, the shape must be consistent with the shape of self except dim. Non-contiguous tensors are supported. The supported data format is ND.</td>
        <td>FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, BFLOAT16.</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indicesOut</td>
        <td>Output</td>
        <td>Indices of the median values.</td>
        <td></td>
        <td>INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="910" id7 -->
    - <term>Atlas training products</term>: The data type BFLOAT16 is not supported.
    <!-- end id7 -->

  - **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1415px"><colgroup>
    <col style="width: 314px">
    <col style="width: 161px">
    <col style="width: 940px">
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
        <td>Returned when self, valuesOut, or indicesOut is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>The data type of self, valuesOut, or indicesOut is not supported.</td>
      </tr>
      <tr>
        <td>The data types of self and valuesOut are different.</td>
      </tr>
      <tr>
        <td>The value of dim is out of the range [-self.dim(), self.dim() - 1].</td>
      </tr>
      <tr>
        <td>The number of dimensions of self, valuesOut, or indicesOut exceeds 8.</td>
      </tr>
      <tr>
        <td>The size of self along dim cannot be 0.</td>
      </tr>
      <tr>
        <td>When keepDim is true, the number of dimensions of valuesOut or indicesOut is not consistent with that of self.</td>
      </tr>
      <tr>
        <td>When keepDim is false, the number of dimensions of valuesOut or indicesOut is not 1 less than that of self.</td>
      </tr>
      <tr>
        <td>When keepDim is true, the shape of valuesOut or indicesOut is not consistent with the shape of self in size along dimensions other than dim.</td>
      </tr>
      <tr>
        <td>When keepDim is true, the size of the shape of valuesOut or indicesOut along dim is not 1.</td>
      </tr>
      <tr>
        <td>When keepDim is false, the shape of valuesOut or indicesOut is not consistent with the shape of self excluding dim.</td>
      </tr>
    </tbody></table>

## aclnnNanMedianDim

  - **Parameters**

  <table style="undefined;table-layout: fixed; width: 1042px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 791px">
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
      <td>Address of the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device, obtained by calling the first-phase API aclnnNanMedianDimGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
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
  - aclnnNanMedianDim defaults to a deterministic implementation.

  - If the data type of self is not FLOAT, FLOAT16, or BFLOAT16, an excessively large tensor size may cause the operator execution to time out (reported as an aicpu error, with reason=[aicpu timeout]). The maximum size limits for specific types (strongly related to the remaining memory of the machine) are as follows:
    - INT64: 150000000
    - UINT8, INT8, INT16, INT32: 725000000
  - When self is of the BFLOAT16 data type, the dimension where dim is located being 1 is not supported, and scalars are not supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_nan_median.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Process the check according to your needs.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output, which need to be custom-built based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> valuesOutShape = {2};
  std::vector<int64_t> indicesOutShape = {2};
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valuesOut = nullptr;
  aclTensor* indicesOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, NAN};
  std::vector<float> valuesOutHostData = {0, 0};
  std::vector<int64_t> indicesOutHostData = {0, 0};
  int64_t dim = 0;
  bool keepDim = false;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the valuesOut aclTensor.
  ret = CreateAclTensor(valuesOutHostData, valuesOutShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &valuesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indicesOut aclTensor.
  ret = CreateAclTensor(indicesOutHostData, indicesOutShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indicesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API (replace it with the actual API).
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNanMedianDim.
  ret = aclnnNanMedianDimGetWorkspaceSize(self, dim, keepDim, valuesOut, indicesOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanMedianDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnNanMedianDim.
  ret = aclnnNanMedianDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanMedianDim failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values and copy the results from the device side to the host side. Modify the code based on the actual API definition.
  auto size = GetShapeSize(valuesOutShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valuesOutDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("valuesOut[%ld] is: %f\n", i, resultData[i]);
  }

  std::vector<int64_t> indicesData(size, 0);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indicesOutDeviceAddr,
                    size * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indicesOut[%ld] is: %ld\n", i, indicesData[i]);
  }

  // 6. Destroy the aclTensor and aclScalar. Modify the code based on the actual API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valuesOut);
  aclDestroyTensor(indicesOut);

  // 7. Release device resources. Modify the code based on the actual API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(valuesOutDeviceAddr);
  aclrtFree(indicesOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
