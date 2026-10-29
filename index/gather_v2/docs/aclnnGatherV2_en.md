# aclnnGatherV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:19:51.636Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Unsupported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Extracts elements from the input tensor along the specified dimension dim based on the index numbers in index, and saves them to the out tensor.
- Example:
  For example, the input tensor $x=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ and the index tensor idx=[1, 0].
  - Result when dim=0: $y=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$

  - Result when dim=1: $y=\begin{bmatrix}2 & 1\\ 5 & 4\\ 8 & 7\end{bmatrix}$

  The specific computation process is as follows:
  Taking a 3D tensor as an example, a tensor x with shape (3,2,2) = $\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$, idx = [1, 0], the subscripts corresponding to dim=0, 1, 2 of tensor x are $l, m, n$ respectively, and idx is one-dimensional (the zero-dimensional case is treated as one-dimensional with size 1).
  - When dim is 0: I = index[i]; &nbsp;&nbsp; y$[i][m][n]$ = x$[I][m][n]$

  - When dim is 1: J = index[j]; &nbsp;&nbsp;&nbsp; y$[l][j][n]$ = x$[l][J][n]$

  - When dim is 2: K = index[k]; &nbsp; y$[l][m][k]$ = x$[l][m][K]$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGatherV2GetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call "aclnnGatherV2" to execute the computation.

```Cpp
aclnnStatus aclnnGatherV2GetWorkspaceSize(
 const aclTensor *self,
 int64_t          dim,
 const aclTensor *index,
 aclTensor       *out,
 uint64_t        *workspaceSize,
 aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGatherV2(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

## aclnnGatherV2GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1479px"><colgroup>
    <col style="width: 183px">
    <col style="width: 120px">
    <col style="width: 265px">
    <col style="width: 299px">
    <col style="width: 197px">
    <col style="width: 114px">
    <col style="width: 156px">
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
        <td>Input aclTensor.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, INT64, INT32, INT16, INT8, UINT8, BOOL, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Input integer.</td>
        <td>The value range is [-self.dim(), self.dim()-1].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>Supports 1-8 dimensions. The value range is 0-self.shape[dim] (including 0 but excluding self.shape[dim]).</td>
        <td>INT64, INT32</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output aclTensor.</td>
        <td>The data type must be the same as that of self. 1 to 8 dimensions are supported. The number of dimensions equals the sum of the dimension counts of self and index minus 1. All dimensions except dim have the same length as the corresponding dimensions of self. The dim dimension is extended to the same shape as index.</td>
        <td>Same as self</td>
        <td>ND</td>
        <td>1-8</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be applied for on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="910,310p" id7 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The BFLOAT16 data type is not supported.

    <!-- end id7 -->

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
      <td>self, index, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of self or index is out of the supported range.</td>
      </tr>
      <tr>
      <td>When self.dim()=0, the value range of dim is not [-1, 0]; when self.dim()>0, the value range of dim is not [-self.dim(), self.dim()-1].</td>
      </tr>
      <tr>
      <td>The number of dimensions of self or index is greater than 8.</td>
      </tr>
      <tr>
      <td>The data types of self and out are inconsistent.</td>
      </tr>
      <tr>
      <td>The shape of out does not meet the following requirement: the dim dimension is expanded to match the shape of index, and all other dimensions have the same length as the corresponding dimensions of self.</td>
      </tr>
    </tbody>
    </table>

## aclnnGatherV2

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
      <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnGatherV2GetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation process.</td>
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

- deterministic computation:
  - aclnnGatherV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_v2.h"

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
  // Call aclrtMemcpy to copy host data to device memory.
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
  // 1. (boilerplate) device/stream initialization. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> indexShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2, 2};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<int64_t> indexHostData = {0, 1, 0, 1};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t dim = 0;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGatherV2.
  ret = aclnnGatherV2GetWorkspaceSize(self, dim, index, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory for the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnGatherV2.
  ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait for the task execution to complete synchronously.
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
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it according to the specific API definition.
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
