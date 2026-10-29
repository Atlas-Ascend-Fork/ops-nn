# aclnnScatterValue&aclnnInplaceScatterValue

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:52:41.795Z -->

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

- Description: Fills values from scalar value into tensor self one by one according to the specified axis, direction, and corresponding position relationship. value is broadcast to tensor src with the same shape as tensor index for the Scatter computation.

- Example:
  For a 3D tensor, self is updated according to the following rules:

  ```text
  self[index[i][j][k]][j][k] = value # if dim == 0
  self[i][index[i][j][k]][k] = value # if dim == 1
  self[i][j][index[i][j][k]] = value # if dim == 2
  ```

  The following requirements must be met during computation:
  - self and index must have the same number of dimensions.
  - For each dimension d, if d != dim, index.size(d) <= self.size(d) is required.
  - The value of dim must be in the range [-number of dimensions of self, number of dimensions of self - 1].
  - The number of dimensions of self must be less than or equal to 8.
  - The values of index in the dimension dim must be in the range [0, self.size(dim)-1].

## Function Prototype

- aclnnScatterValue and aclnnInplaceScatterValue implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnScatterValue: A new output tensor object needs to be created to store the computation result.
  - aclnnInplaceScatterValue: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnScatterValueGetWorkspaceSize" or "aclnnInplaceScatterValueGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call "aclnnScatterValue" or "aclnnInplaceScatterValue" to perform the computation.

  ```cpp
  aclnnStatus aclnnScatterValueGetWorkspaceSize(
    const aclTensor *self,
    int64_t          dim,
    const aclTensor *index,
    const aclScalar *value,
    int64_t          reduce,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
  ```

  ```cpp
  aclnnStatus aclnnScatterValue(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
  ```

  ```cpp
  aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(
    aclTensor       *selfRef,
    int64_t          dim,
    const aclTensor *index,
    const aclScalar *value,
    int64_t          reduce,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
  ```

  ```cpp
  aclnnStatus aclnnInplaceScatterValue(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
  ```

## aclnnScatterValueGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1697px"><colgroup>
  <col style="width: 133px">
  <col style="width: 129px">
  <col style="width: 286px">
  <col style="width: 329px">
  <col style="width: 410px">
  <col style="width: 111px">
  <col style="width: 143px">
  <col style="width: 156px">
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
      <td>self in the formula.</td>
      <td>Target tensor of scatter. The number of dimensions of self must be the same as that of index.</td>
      <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>Dimension along which to scatter.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>The index in the formula.</td>
      <td>Index tensor. The number of dimensions of index must be the same as that of self.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td></td>
      <td>If value is COMPLEX, self must also be a COMPLEX tensor. No other data type restrictions apply.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="4">reduce</td>
      <td rowspan="4">Input</td>
      <td rowspan="4">Specifies the reduction operation to apply.</td>
      <td rowspan="4">Operation options and their corresponding int values are (add, 1), (mul, 2), and (none, 0). The specific operations are as follows:<br>0: replacement operation. Replaces the corresponding positions in out with value according to index.<br>1: addition operation. Adds value to the corresponding positions in out according to index.<br>2: multiplication operation. Multiplies the corresponding positions in out by value according to index.<br></td>
      <td rowspan="4">-</td>
      <td rowspan="4">-</td>
      <td rowspan="4">-</td>
      <td rowspan="4">-</td>
    </tr>
    <tr>
    </tr>
    <tr>
    </tr>
    <tr>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The scatter output result will be stored in this tensor.</td>
      <td>Data format, data type, and shape must be the same as those of self.</td>
      <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
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
      <td>Passed self, index, value, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type of self, index, value, or out is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and out are different.</td>
    </tr>
    <tr>
      <td>The numbers of dimensions of self and index are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td rowspan="2">The shapes of self and index do not meet the following restriction:<br>For each dimension d, if d != dim, index.size(d) <= self.size(d).</td>
    </tr>
    <tr>
    </tr>
    <tr>
      <td>The value of dim is not in [-number of dimensions of self, number of dimensions of self - 1].</td>
    </tr>
    <tr>
      <td>The number of dimensions of self exceeds 8.</td>
    </tr>
    <tr>
      <td>If value is COMPLEX, the data type of self is not COMPLEX.</td>
    </tr>
  </tbody>
  </table>

## aclnnScatterValue

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
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnScatterValueGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, including the operator computation flow.</td>
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

## aclnnInplaceScatterValueGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1721px"><colgroup>
    <col style="width: 130px">
    <col style="width: 126px">
    <col style="width: 259px">
    <col style="width: 407px">
    <col style="width: 400px">
    <col style="width: 108px">
    <col style="width: 139px">
    <col style="width: 152px">
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
        <td>selfRef</td>
        <td>Input/Output</td>
        <td>The self in the formula.</td>
        <td>Target tensor of the scatter operation. The number of dimensions of selfRef must be the same as that of index. Empty tensors are supported.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>The dimension along which to scatter.</td>
        <td>The value range is [-the number of dimensions in selfRef, the number of dimensions in selfRef-1].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Input</td>
        <td>The index in the formula.</td>
        <td>Index tensor. The number of dimensions in index must be the same as that in selfRef. For each dimension d, if d != dim, index.size(d) &lt;= selfRef.size(d) must be met. Empty tensors are supported.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td></td>
        <td>When value is COMPLEX, selfRef must also be a COMPLEX tensor. No other data type restrictions apply.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td rowspan="4">reduce</td>
        <td rowspan="4">Input</td>
        <td rowspan="4">Selects the reduction operation to apply.</td>
        <td rowspan="4">Available operation options and corresponding int values are (add, 1), (mul, 2), and (none, 0). The operations are described as follows:<br>0: replacement. value replaces the element at the corresponding position in selfRef according to index.<br>1: addition. value is added to the element at the corresponding position in selfRef according to index.<br>2: multiplication. value is multiplied into the element at the corresponding position in selfRef according to index.</td>
        <td rowspan="4">-</td>
        <td rowspan="4">-</td>
        <td rowspan="4">-</td>
        <td rowspan="4">-</td>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
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
      <td>Passed selfRef, index, or value is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of selfRef, index, or value is out of the supported range.</td>
    </tr>
    <tr>
      <td>The dimension counts of selfRef and index are inconsistent.</td>
    </tr>
    <tr>
      <td rowspan="2">The shapes of self and index do not meet the following restrictions:<br>For each dimension d, if d != dim, index.size(d) <= selfRef.size(d) must be satisfied.<br></td>
    </tr>
    <tr>
    </tr>
    <tr>
      <td>The value of dim is not in the range [-selfRef dimension count, selfRef dimension count-1].</td>
    </tr>
    <tr>
      <td>The dimension count of selfRef exceeds 8.</td>
    </tr>
    <tr>
      <td>When value is COMPLEX, the data type of selfRef is not COMPLEX.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceScatterValue

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnInplaceScatterValueGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

Deterministic computation description:

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: Non-deterministic by default. Determinism can be enabled through aclrtCtxSetSysParamOpt (when reduction is product, enabling determinism does not take effect and the computation remains non-deterministic).
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: Deterministic by default.

<!-- end id8 -->

## Example

The sample code is as follows and is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**aclnnScatterValue sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter.h"

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
  // Call aclrtMalloc to apply for memory on the device
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual
  // Set deviceId based on the actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. The construction must be customized based on the API
  int64_t dim = 1;
  int64_t reduce = 1;
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> indexShape = {2, 3};
  std::vector<int64_t> outShape = {3, 4};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float Value = 1.2f;

  // Create the self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create value aclScalar.
  value = aclCreateScalar(&Value, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScatterValue.
  ret = aclnnScatterValueGetWorkspaceSize(self, dim, index, value, reduce, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterValueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnScatterValue.
  ret = aclnnScatterValue(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterValue failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyScalar(value);
  aclDestroyTensor(out);

  // 7. Release device resources, which must be modified based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

**aclnnInplaceScatterValue sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter.h"

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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output, which must be customized based on the API definition.
  int64_t dim = 1;
  int64_t reduce = 1;
  std::vector<int64_t> selfRefShape = {3, 4};
  std::vector<int64_t> indexShape = {2, 3};
  void* selfRefDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* index = nullptr;
  aclScalar* value = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  float Value = 1.2f;

  // Create the selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the value aclScalar.
  value = aclCreateScalar(&Value, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Change it to the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceScatterValue.
  ret = aclnnInplaceScatterValueGetWorkspaceSize(selfRef, dim, index, value, reduce, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterValueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceScatterValue.
  ret = aclnnInplaceScatterValue(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterValue failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from device memory to host memory. Modify the code according to the actual API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code according to the actual API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(index);
  aclDestroyScalar(value);

  // 7. Release device resources. Modify the code according to the actual API definition.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indexDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
