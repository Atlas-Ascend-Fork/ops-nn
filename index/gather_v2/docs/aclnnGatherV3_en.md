# aclnnGatherV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:19:59.179Z -->

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

- Description: Extracts elements from the input tensor along the specified dimension dim based on the index values in index, where batchDims represents the number of batch dimensions. The result is stored in the out tensor.
- Example:

  For example, when batchDims is 0, for the input tensor $self=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ and the index tensor index=[1, 0].
  - Result when dim=0: $out=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$

  - Result when dim=1: $out=\begin{bmatrix}2 & 1\\ 5 & 4\\ 8 & 7\end{bmatrix}$

  The specific computation process is as follows:
  Taking a 3D tensor as an example, a tensor self with shape (3,2,2) = $\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$, index = [1, 0], the subscripts corresponding to dim=0, 1, 2 of the self tensor are $l$, $m$, $n$ respectively, and index is one-dimensional (for the zero-dimensional case: it is treated as a one-dimensional tensor of size 1).
  - When dim is 0: I = index[i]; &nbsp;&nbsp; out$[i][m][n]$ = self$[I][m][n]$

  - When dim is 1: J = index[j]; &nbsp;&nbsp; out$[l][j][n]$ = self$[l][J][n]$

  - When dim is 2: K = index[k]; &nbsp;&nbsp; out$[l][m][k]$ = self$[l][m][K]$

  When batchDims is 1: taking a four-dimensional tensor as an example, the operation on a tensor self with shape (3,3,2,2) and a tensor index with shape (3,2) is equivalent to performing three gather operations with batchDims being 0 and dim being dim-batchDims.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGatherV3GetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnGatherV3" to perform the computation.

```Cpp
aclnnStatus aclnnGatherV3GetWorkspaceSize(
 const aclTensor *self,
 int64_t          dim,
 const aclTensor *index,
 int64_t          batchDims,
 int64_t          mode,
 aclTensor       *out,
 uint64_t        *workspaceSize,
 aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGatherV3(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

## aclnnGatherV3GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1473px"><colgroup>
    <col style="width: 143px">
    <col style="width: 120px">
    <col style="width: 227px">
    <col style="width: 307px">
    <col style="width: 254px">
    <col style="width: 117px">
    <col style="width: 160px">
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
        <td>Data to gather.</td>
        <td>-</td>
        <td>FLOAT,FLOAT16,BFLOAT16,INT64,INT32,INT16,INT8,UINT8,BOOL,DOUBLE,COMPLEX64</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Axis to gather along.</td>
        <td>The value range is [-self.dim(), self.dim()-1].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Input</td>
        <td>Index for collecting data.</td>
        <td>The value range is 0-self.shape[dim] (including 0, excluding self.shape[dim]). When batchDims = N and N != 0, the first N dimensions of index must be the same as those of self.</td>
        <td>INT64, INT32</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>batchDims</td>
        <td>Input</td>
        <td>Operation batch.</td>
        <td>The value range is [0, dim] and must be less than or equal to rank(index).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>mode</td>
        <td>Input</td>
        <td>Calculation mode.</td>
        <td>Value range: [0, 2].<ul><li>0: Performance optimization for the scattered-index use case;</li><li>1: Performance optimization for the gathered-index use case.</li><li>2: Supports out-of-bound indices. Currently, only 1 is supported.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output aclTensor.</td>
        <td>When batchDims = N and N == 0, the number of dimensions equals the sum of the dimensions of self and index minus one. Except for the dim dimension, which is expanded to the same shape as index, the lengths of the other dimensions are the same as the corresponding dimensions of self; <br> When N != 0, the dimensions of output are self[:dim]+index[N+1:]+self[dim+1:].</td>
        <td>Same as self</td>
        <td>ND</td>
        <td>1-8</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace that needs to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator calculation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Currently, only 0 is supported for dim and batchDims.

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
      <td>Parameters self, index, and out are null pointers.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of self and index are not in the supported range.</td>
      </tr>
      <tr>
      <td>The number of dimensions of self and index is greater than 8.</td>
      </tr>
      <tr>
      <td>The data types of self and out are inconsistent.</td>
      </tr>
      <tr>
      <td>The shape of out does not meet the requirement that, except for dimension 0 which is expanded to match the shape of index, the lengths of the other dimensions are consistent with the corresponding dimensions of self.</td>
      </tr>
    </tbody>
    </table>

## aclnnGatherV3

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
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnGatherV3GetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator calculation process.</td>
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
  - aclnnGatherV3 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_v3.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to request device-side memory.
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
  // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> indexShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2, 2};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* out = nullptr;
  int64_t dim = 0;
  int64_t batchDims = 0;
  int64_t mode = 1;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<int64_t> indexHostData = {1, 0, 0, 1};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGatherV3
  ret = aclnnGatherV3GetWorkspaceSize(self, dim, index, batchDims, mode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnGatherV3
  ret = aclnnGatherV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherV3 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to finish
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify the code based on the definition of the specific API
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy the aclTensor and aclScalar. Modify the code based on the definition of the specific API
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the definition of the specific API
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
