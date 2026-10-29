# aclnnScatter&aclnnInplaceScatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:48:35.144Z -->

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

- Description: Replaces/accumulates/multiplies values from the tensor src into the tensor self element by element according to the specified axis, direction, and corresponding position relationship.

- Example:
  For a 3D tensor, self is updated according to the following rules:

  ```text
  self[index[i][j][k]][j][k] += src[i][j][k] # If dim == 0 && reduction == 1
  self[i][index[i][j][k]][k] *= src[i][j][k] # If dim == 1 && reduction == 2
  self[i][j][index[i][j][k]] = src[i][j][k]  # If dim == 2 && reduction == 0
  ```

- The following requirements must be met during computation:
  - self, index, and src must have the same number of dimensions.
  - For each dimension d, index.size(d) must be less than or equal to src.size(d).
  - For each dimension d, if d != dim, index.size(d) must be less than or equal to self.size(d).
  - The value of dim must be in the range [-number of dimensions of self, number of dimensions of self - 1].
  - The number of dimensions of self must be less than or equal to 8.
  - The values of index in the dimension dim must be in the range [0, self.size(dim)-1].

## Function Prototype

aclnnScatter and aclnnInplaceScatter implement the same functionality. The differences are as follows. Select the appropriate operator based on your actual scenario.

- aclnnScatter: Requires creating a new output tensor object to store the computation result.
- aclnnInplaceScatter: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnScatterGetWorkspaceSize" or "aclnnInplaceScatterGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call the "aclnnScatter" or "aclnnInplaceScatter" API to execute the computation.

```Cpp
aclnnStatus aclnnScatterGetWorkspaceSize(
  const aclTensor* self,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  int64_t          reduce,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScatter(
  void*               workspace,
  uint64_t            workspaceSize,
  aclOpExecutor*      executor,
  const aclrtStream   stream)
```

```Cpp
aclnnStatus aclnnInplaceScatterGetWorkspaceSize(
  aclTensor*       selfRef,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  int64_t          reduce,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceScatter(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnScatterGetWorkspaceSize

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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>The `self` in the formula.</td>
        <td>The data type must be the same as that of src.<br>The dimension count must be the same as index and src.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>The dimension along which to scatter.</td>
        <td>The range is [-the dimension count of self, the dimension count of self - 1].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index (aclTensor*)</td>
        <td>Input</td>
        <td>Index tensor.</td>
        <td>The dimension count must be the same as that of self and src.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src (aclTensor*)</td>
        <td>Input</td>
        <td>`src` in the formula.</td>
        <td>The data type must be the same as that of self.<br>The dimension count must be the same as that of self and index.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduce (int64_t)</td>
        <td>Input</td>
        <td>Selects the reduction operation to apply.</td>
        <td>Optional operations and their corresponding int values are (add, 1), (mul, 2), and (none, 0).<br>The meanings are as follows: 0 indicates replacement, 1 indicates addition, and 2 indicates multiplication.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Calculation result.</td>
        <td>The data type must be the same as that of self.<br>The shape must be the same as that of self.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device.</td>
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
        <td>self, index, src, or out contains a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data type of self, index, src, or out is not in the supported range.</td>
      </tr>
      <tr>
        <td>The data types of self, src, and out are different.</td>
      </tr>
      <tr>
        <td>The number of dimensions of self, index, and src is inconsistent.</td>
      </tr>
      <tr>
        <td>The shapes of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The value of dim is not in [-self dimension count, self dimension count - 1].</td>
      </tr>
      <tr>
        <td>The number of dimensions of self exceeds 8.</td>
      </tr>
      <tr>
        <td>The shapes of self, index, and src do not meet the constraints: for each dimension d, index.size(d) <= src.size(d); and when d != dim, index.size(d) <= self.size(d).</td>
      </tr>
    </tbody></table>

## aclnnScatter

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
        <td>Workspace size requested on the Device, obtained by the first-phase API aclnnScatterGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor. It contains the computation flow of the operator.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceScatterGetWorkspaceSize

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
        <td>selfRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Target tensor of the scatter operation.</td>
        <td>The shape must have the same dimension count as index and src.<br>The data type must be the same as that of src.<br>Empty tensors are supported.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>Specifies the dimension along which the scatter operation is performed.</td>
        <td>Range: [-dimension count of selfRef, dimension count of selfRef-1].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index (aclTensor*)</td>
        <td>Input</td>
        <td>Index tensor.</td>
        <td>Specifies the positions in the src tensor to scatter into the self tensor.<br>Empty tensors are supported.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src(aclTensor*)</td>
        <td>Input</td>
        <td>Source tensor.</td>
        <td>The data type must be the same as that of selfRef.<br>Empty tensors are supported.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduce(int64_t)</td>
        <td>Input</td>
        <td>Reduction operation to apply.</td>
        <td>Optional operations: 0 indicates replacement, 1 indicates addition, and 2 indicates multiplication.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t*)</td>
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
        <td>selfRef, index, or src contains a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>The data type of selfRef, index, or src is not within the supported range.</td>
      </tr>
      <tr>
        <td>The data types of selfRef and src are different.</td>
      </tr>
      <tr>
        <td>The number of dimensions of selfRef, index, and src is inconsistent.</td>
      </tr>
      <tr>
        <td>The value of dim is not in the range [-selfRef's dimension count, selfRef's dimension count-1].</td>
      </tr>
      <tr>
        <td>The number of dimensions of selfRef exceeds 8.</td>
      </tr>
      <tr>
        <td>The shape does not meet the restrictions: for each dimension d, index.size(d) <= src.size(d); and when d != dim, index.size(d) <= selfRef.size(d).</td>
      </tr>
    </tbody></table>

## aclnnInplaceScatter

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
        <td>Workspace memory address allocated on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnInplaceScatterGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, containing the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

Deterministic computation description:

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: Default to a nondeterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt (When reduction is the product operation, enabling determinism does not take effect and the result remains nondeterministic).
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: Default to a deterministic implementation.

<!-- end id8 -->

## Example

The following example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**aclnnScatter Sample Code:**

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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. You need to customize the construction based on the API.
  int64_t dim = 1;
  int64_t reduce = 1;
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> indexShape = {2, 3};
  std::vector<int64_t> srcShape = {2, 3};
  std::vector<int64_t> outShape = {3, 4};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* src = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  std::vector<float> srcHostData = {-1, -2, -3, -4, -5, -6};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the src aclTensor.
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScatter.
  ret = aclnnScatterGetWorkspaceSize(self, dim, index, src, reduce, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnScatter.
  ret = aclnnScatter(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatter failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from device memory to host memory. Modify based on the interface definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify based on the interface definition of the specific API.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(src);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(srcDeviceAddr);
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

**aclnnInplaceScatter sample code:**

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

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize based on the API definition.
  int64_t dim = 1;
  int64_t reduce = 1;
  std::vector<int64_t> selfRefShape = {3, 4};
  std::vector<int64_t> indexShape = {2, 3};
  std::vector<int64_t> srcShape = {2, 3};
  void* selfRefDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* index = nullptr;
  aclTensor* src = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  std::vector<float> srcHostData = {-1, -2, -3, -4, -5, -6};

  // Create the selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the src aclTensor.
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceScatter.
  ret = aclnnInplaceScatterGetWorkspaceSize(selfRef, dim, index, src, reduce, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceScatter.
  ret = aclnnInplaceScatter(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatter failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Change it based on the specific API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Change it based on the specific API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(index);
  aclDestroyTensor(src);

  // 7. Release device resources. Change it based on the specific API definition.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(srcDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
