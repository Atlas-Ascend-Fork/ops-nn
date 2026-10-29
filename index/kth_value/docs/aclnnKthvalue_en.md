# aclnnKthvalue

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:26:34.319Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

Returns the k-th smallest value and its index along the specified dimension of the input tensor.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnKthvalueGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call the "aclnnKthvalue" API to perform the computation.

```Cpp
aclnnStatus aclnnKthvalueGetWorkspaceSize(
 const aclTensor*     self,
 int64_t              k,
 int64_t              dim,
 bool                 keepdim,
 aclTensor*           valuesOut,
 aclTensor*           indicesOut,
 uint64_t*            workspaceSize,
 aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnKthvalue(
 void*             workspace,
 uint64_t          workspaceSize,
 aclOpExecutor*    executor,
 const aclrtStream stream)
```

## aclnnKthvalueGetWorkspaceSize

* **Parameters**

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
        <td>self</td>
        <td>Input</td>
        <td>-</td>
        <td>-</td>
        <td>BFLOAT16, FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>k</td>
        <td>Input</td>
        <td>Integer on the host, indicating the k-th minimum values to obtain along a specified dimension.</td>
        <td><ul><li>The value range is [0, self.size(dim)].</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Host-side integer that specifies the dimension of the input tensor.</td>
        <td><ul><li>The value range is [-self.dim(), self.dim()).</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepdim</td>
        <td>Input</td>
        <td>bool data, indicating whether the output tensor keeps dim.</td>
        <td><ul><li>True indicates that the sizes of valuesOut and indicesOut tensors are the same as that of self;</li><li>False indicates that dim is reduced, and the resulting tensor has one fewer dimension than input.</li></ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>valuesOut</td>
        <td>Output</td>
        <td>aclTensor on the device, with the data type consistent with self.</td>
        <td>The size of the sorting axis in shape is 1, and the non-sorting axes are consistent with self.</td>
        <td>Consistent with self</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indicesOut</td>
        <td>Output</td>
        <td>Device-side aclTensor, representing the index of the k-th minimum value along dim in the original input tensor.</td>
        <td>The shape has a size of 1 on the sorting axis and is consistent with self on non-sorting axes.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be requested on the device.</td>
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

    <!-- npu="910,310p" id7 -->
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, and UINT8 data types are supported.
    <!-- end id7 -->
    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: BFLOAT16, FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, and UINT8 data types are supported.
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: Data type support: BFLOAT16, FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64.
    <!-- end id9 -->

* **Return Value**

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
      <td>The self, valuesOut, or indicesOut is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of self, valuesOut, or indicesOut is not within the supported range, or the shapes do not match.</td>
      </tr>
      <tr>
      <td>The value of dim is not within the dimension range of the input tensor self.</td>
      </tr>
      <tr>
      <td>k is less than 0 or greater than the size of input self along the dim dimension.</td>
      </tr>
    </tbody>
    </table>

## aclnnKthvalue

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
      <td>Workspace memory address requested on the Device side.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by calling the first-phase API aclnnKthvalueGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation process.</td>
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

- Deterministic computation:
  - aclnnKthvalue defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kthvalue.h"

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
  // Boilerplate for resource initialization
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

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {2, 4};
  std::vector<int64_t> outShape = {2, 1};
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  int64_t dim = -1;
  int64_t k = 2;
  bool keepdim = true;
  aclTensor* valuesOut = nullptr;
  aclTensor* indicesOut = nullptr;
  std::vector<float> selfHostData = {0.0, 1.1, 2, 3, 4, 5, 6, 7};
  std::vector<float> valuesOutHostData = {0.0, 0};
  std::vector<int64_t> indicesOutHostData = {0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the valuesOut aclTensor.
  ret = CreateAclTensor(valuesOutHostData, outShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &valuesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indicesOut aclTensor.
  ret = CreateAclTensor(indicesOutHostData, outShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indicesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be changed to a specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnKthvalue.
  ret = aclnnKthvalueGetWorkspaceSize(self, k, dim, keepdim, valuesOut, indicesOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKthvalueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnKthvalue.
  ret = aclnnKthvalue(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKthvalue failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values and copy the results on the device memory to the host. Change the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> valuesData(size, 0);
  ret = aclrtMemcpy(valuesData.data(), valuesData.size() * sizeof(valuesData[0]), valuesOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("valuesResult[%ld] is: %f\n", i, valuesData[i]);
  }
  std::vector<int64_t> indicesData(size, 0);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indicesOutDeviceAddr, size * sizeof(int64_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indicesResult[%ld] is: %ld\n", i, indicesData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Change the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valuesOut);
  aclDestroyTensor(indicesOut);

  // 7. Release the device resources. Change the code based on the API definition.
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
