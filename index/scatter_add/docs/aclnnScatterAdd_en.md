# aclnnScatterAdd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:50:59.270Z -->

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

- Description: Fills values from the src tensor into the self tensor element by element according to the specified dimension and the positional relationships in the index tensor. If more than one src value is filled into the same position of self, these values are accumulated at that position.

  Example:
  For a 3D tensor, self is updated according to the following rules:

  ```text
  self[index[i][j][k]][j][k] += src[i][j][k] # if dim == 0
  self[i][index[i][j][k]][k] += src[i][j][k] # if dim == 1
  self[i][j][index[i][j][k]] += src[i][j][k] # if dim == 2
  ```

  The following requirements must be met during computation:
  - self, index, and src must have the same number of dimensions.
  - For each dimension d, index.size(d) <= src.size(d).
  - For each dimension d where d != dim, index.size(d) <= self.size(d).
  - dim ranges from [-self.dim(), self.dim() - 1].

- Example:

  Input tensor $self = \begin{bmatrix} [1&2&3] \\ [4&5&6] \\ [7&8&9] \end{bmatrix}$,
  index tensor $index = \begin{bmatrix} [0&2&1] \\ [0&0&1] \end{bmatrix}$, dim = 1,
  source tensor $src = \begin{bmatrix} [10&11&12] \\ [13&14&15] \end{bmatrix}$,
  output tensor $output = \begin{bmatrix} [11&14&14] \\ [31&20&6] \\ [7&8&9] \end{bmatrix}$

  dim = 1 indicates that scatter_add accumulates values along the columns of the tensor based on $index$.

  $output[0][0] = self[0][0] + src[0][0]$ = 1 + 10,

  $output[0][1] = self[0][1] + src[0][2]$ = 2 + 12,

  $output[0][2] = self[0][2] + src[0][1]$ = 3 + 11,

  $output[1][0] = self[1][0] + src[1][0] + src[1][1]$ = 4 + 13 + 14,

  $output[1][1] = self[1][1] + src[1][2]$ = 5 + 15,

  $output[1][2] = self[1][2]$ = 6,

  $output[2][0] = self[2][0]$ = 7,

  $output[2][1] = self[2][1]$ = 8,

  $output[2][2] = self[2][2]$ = 9.

  Here, the Number of Dimensions of $self$, $index$, and $src$ are all 2. Each dimension size {2, 3} of $index$ is not greater than the corresponding dimension size {2, 3} of $src$. In the dimension where dim != 1 (dim = 0), the dimension size {2} of $index$ is not greater than the corresponding dimension size {3} of $self$. The maximum value {2} in $index$ is less than the size {3} of $self$ in the dim = 1 dimension.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnScatterAddGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnScatterAdd" to perform the computation.

```Cpp
aclnnStatus aclnnScatterAddGetWorkspaceSize(
  const aclTensor* self,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScatterAdd(
  void*               workspace,
  uint64_t            workspaceSize,
  aclOpExecutor*      executor,
  const aclrtStream   stream)
```

## aclnnScatterAddGetWorkspaceSize

- **Parameters**

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
        <td>Input self in the formula. The target tensor of the scatter operation.</td>
        <td>The number of dimensions must be the same as that of index and src.<br>The data type must be the same as that of src.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>Input dim in the formula.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index (aclTensor*)</td>
        <td>Input</td>
        <td>Index input in the formula.</td>
        <td>The number of dimensions must be the same as that of src.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src (aclTensor*)</td>
        <td>Input</td>
        <td>Input src in the formula, which is the source tensor.</td>
        <td>The number of dimensions must be the same as that of index.<br>The data type must be the same as that of self.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
        <td>ND</td>
        <td>-</td>
        <td>×</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>The output in the formula.</td>
        <td>The shape must be the same as that of self.<br>The data type must be the same as the data type of self.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
        <td>ND</td>
        <td>-</td>
        <td>×</td>
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
        <td>Returns the op executor, which contains the operator computation process.</td>
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
        <td>self, index, src, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of self, index, src, or out is not supported.</td>
      </tr>
      <tr>
        <td>The shapes of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The shape of src or index is invalid.</td>
      </tr>
    </tbody></table>

## aclnnScatterAdd

- **Parameters**

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
        <td>Workspace memory address applied for on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size applied for on the Device, obtained by the first-phase API aclnnScatterAddGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, containing the operator computation process.</td>
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

Determinism description:

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: defaults to a non-deterministic implementation.
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: defaults to a deterministic implementation.

<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_add.h"

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
  // Call aclrtMemcpy to copy host data to the device memory.
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
// 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

// 2. Construct the input and output. Perform custom construction based on the API definition.
  std::vector<int64_t> selfShape = {4, 4};
  std::vector<int64_t> indexShape = {3, 4};
  std::vector<int64_t> srcShape = {4, 4};
  std::vector<int64_t> outShape = {4, 4};
  int64_t dim = 0;
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* src = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData(16, 0);
  std::vector<int64_t> indexHostData = {0, 1, 2, 1, 0, 1, 2, 0, 2, 2, 1, 0};
  std::vector<float> srcHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData(16, 0);
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

// 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScatterAdd.
  ret = aclnnScatterAddGetWorkspaceSize(self, dim, index, src, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
  ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
  // Call the second-phase API of aclnnScatterAdd.
  ret = aclnnScatterAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterAdd failed. ERROR: %d\n", ret); return ret);

// 4. (Boilerplate) Synchronize and wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

// 5. Obtain the output values. Copy the result from device memory to host memory. Modify it as required by the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
  size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
  LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

// 6. Release the aclTensor and aclScalar. Modify it as required by the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(src);
  aclDestroyTensor(out);
  // 7. Release device resources. Modify it as required by the parameters in the API definition.
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
