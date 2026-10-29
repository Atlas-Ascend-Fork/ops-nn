# aclnnIndexSelect

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:19:54.061Z -->

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

- Description: Extracts elements from the input tensor along the specified dimension dim based on the index numbers in index, and saves them to the out tensor.
    For example, for the input tensor $self=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ and the index tensor index=[1, 0],
    the result of self.index_select(0, index) is: $y=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$;

    the result of self.index_select(1, index) is: $y=\begin{bmatrix}2 & 1\\ 5 & 4\\8 & 7\end{bmatrix}$;

- Formula:

    Taking a three-dimensional tensor as an example, the tensor **self** with shape (3,2,2) =$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$, **index**=[1, 0], the subscripts corresponding to dim=0, 1, and 2 of the self tensor are $l$, $m$, and $n$ respectively, and index is one-dimensional.

    dim is 0, index_select(0, index):   I=index[i];  &nbsp;&nbsp;   out$[i][m][n]$ = self$[I][m][n]$

    dim is 1, index_select(1, index):   J=index[j];  &nbsp;&nbsp;&nbsp;    out$[l][j][n]$ = self$[l][J][n]$

    dim is 2, index_select(2, index):   K=index[k]; &nbsp;  out$[l][m][k]$ = self$[l][m][K]$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnIndexSelectGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the computation flow of the operator, and then call the "aclnnIndexSelect" API to perform the computation.

```Cpp
aclnnStatus aclnnIndexSelectGetWorkspaceSize(
 const aclTensor *self,
 int64_t          dim,
 const aclTensor *index,
 aclTensor       *out,
 uint64_t        *workspaceSize,
 aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnIndexSelect(
 void             *workspace,
 uint64_t          workspaceSize,
 aclOpExecutor    *executor,
 const aclrtStream stream)
```

## aclnnIndexSelectGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1519px"><colgroup>
    <col style="width: 147px">
    <col style="width: 120px">
    <col style="width: 233px">
    <col style="width: 257px">
    <col style="width: 270px">
    <col style="width: 183px">
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
        <td>Input tensor.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, BFLOAT16, INT64, INT32, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128</td>
        <td>ND, NCHW, NHWC, HWCN, NDHWC, NCDHW</td>
        <td>Not greater than 8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Specified dimension.</td>
        <td>Range: [-self.dim(), self.dim() - 1].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Input</td>
        <td>Index. Only 0D or 1D (a 0D tensor is treated as a 1D tensor of size 1).</td>
        <td>The index values in index are in the range of 0 to self.shape[dim] (including 0, excluding self.shape[dim]).</td>
        <td>INT64, INT32</td>
        <td>ND, NCHW, NHWC, HWCN, NDHWC, NCDHW</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output tensor.</td>
        <td>The dim length equals the length of index, and other dimension lengths are the same as the corresponding dimensions of self.</td>
        <td>Same as self</td>
        <td>ND, NCHW, NHWC, HWCN, NDHWC, NCDHW</td>
        <td>Same as self</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be allocated on the Device side.</td>
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
      <td>Parameters self, index, and out are null pointers.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of parameters self and index are not within the supported range.</td>
      </tr>
      <tr>
      <td>dim >= self.dim() or dim < -self.dim().</td>
      </tr>
      <tr>
      <td>The dimension of index is greater than 1.</td>
      </tr>
      <tr>
      <td>The dimension of self is greater than 8.</td>
      </tr>
    </tbody>
    </table>

## aclnnIndexSelect

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
      <td>Workspace memory address allocated on the Device side.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnIndexSelectGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes the task.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When the shape of self is [], the shape of index can only be [1].

- Deterministic computation:
  - aclnnIndexSelect defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_select.h"

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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host side to the device side memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Perform custom construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> indexShape = {2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  int64_t dim = 0;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> indexHostData = {1, 0};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnIndexSelect first-phase API.
  ret = aclnnIndexSelectGetWorkspaceSize(self, dim, index, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexSelectGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnIndexSelect second-phase API.
  ret = aclnnIndexSelect(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexSelect failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the Device side to the host side. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the specific API definition.
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
