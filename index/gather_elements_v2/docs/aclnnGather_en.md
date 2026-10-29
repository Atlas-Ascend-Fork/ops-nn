# aclnnGather

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:17:26.339Z -->

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

- Description: Gathers data along a specified dimension dim of the input tensor.

- Formula:
  Given a tensor $self$, a dimension $d$, and an index tensor $index$, let $n$ be the number of dimensions of $self$, $i_d$ denote the index along dimension $d$, and $index_{i_d}$ denote the $i_d$-th index value of the index tensor $index$ along dimension $d$. The gather operation along the specified dimension $d$ can be expressed by the following mathematical formula:

  $$
  gather(X,index,d)_{i_0,i_1,\cdots,i_{d-1},i_{d+1},\cdots,i_{n-1}} = self_{i_0,i_1,\cdots,i_{d-1},index_{i_d},i_{d+1},\cdots,i_{n-1}}
  $$

- Example:
  - Example 1:
    Assume the input tensor $self=\begin{bmatrix}1 & 2 & 3\\ 4 & 5 & 6\\ 7 & 8 & 9\end{bmatrix}$, the index tensor $index=\begin{bmatrix}0 & 2\\ 1 & 0\end{bmatrix}$, and $dim = 0$. Then the output tensor $out=\begin{bmatrix}1 & 8\\ 4 & 2\end{bmatrix}$. The specific computation process is as follows:

    $$
    \begin{aligned} out_{0,0}&=self_{index_{0,0}, 0}=self_{0,0}=1 \\
    out_{0,1}&=self_{index_{0,1}, 1}=self_{2,1}=8 \\
    out_{1,0}&=self_{index_{1,0}, 0}=self_{1,0}=4 \\
    out_{1,1}&=self_{index_{1,1}, 1}=self_{0,1}=2 \end{aligned}
    $$

  - Example 2:
    Assume the input tensor $self=\begin{bmatrix}1 & 2 & 3\\ 4 & 5 & 6\\ 7 & 8 & 9\end{bmatrix}$, the index tensor $index=\begin{bmatrix}0 & 2\\ 1 & 0\end{bmatrix}$, and $dim = 1$. Then the output tensor $out=\begin{bmatrix}1 & 3\\ 5 & 4\end{bmatrix}$. The specific computation process is as follows:

    $$
    \begin{aligned} out_{0,0}&=self_{0, index_{0,0}}=self_{0,0}=1 \\
    out_{0,1}&=self_{0, index_{0,1}}=self_{0,2}=3 \\
    out_{1,0}&=self_{1, index_{1,0}}=self_{1,1}=5 \\
    out_{1,1}&=self_{1, index_{1,1}}=self_{1,0}=4 \end{aligned}
    $$

## Function Prototype

  Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnGatherGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnGather" API to execute the computation.

```Cpp
aclnnStatus aclnnGatherGetWorkspaceSize(
 const aclTensor*  self,
 const int64_t     dim,
 const aclTensor*  index,
 aclTensor*        out,
 uint64_t*         workspaceSize,
 aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnGather(
 void*             workspace,
 uint64_t          workspaceSize,
 aclOpExecutor*    executor,
 const aclrtStream stream)
```

## aclnnGatherGetWorkspaceSize

  - **Parameters**

    <table style="undefined;table-layout: fixed; width: 1438px"><colgroup>
    <col style="width: 162px">
    <col style="width: 120px">
    <col style="width: 245px">
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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>self in the formula.</td>
        <td>The data type must be consistent with out, and the number of dimensions must be consistent with index.</td>
        <td>DOUBLE, FLOAT16, BFLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
        <td>-</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>d in the formula.</td>
        <td>Value range [-self.dim(), self.dim() - 1]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index (aclTensor*)</td>
        <td>Input</td>
        <td>The index in the formula.</td>
        <td>The number of dimensions must be consistent with self, and the shape must be consistent with out. Except for the specified dimension dim, the size of each dimension must be less than or equal to the size of the corresponding dimension of self.<br>The values in index represent the indices of self on the dim axis, with a value range of [-self.shape[dim], self.shape[dim] - 1]. Out-of-bound indices in index are not supported.</td>
        <td>INT32, INT64</td>
        <td>-</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Output aclTensor*.</td>
        <td>The data type must be consistent with self, and the shape must be consistent with index.</td>
        <td>Consistent with self</td>
        <td>-</td>
        <td>Consistent with index</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>: The data type of self cannot be FLOAT8_E5M2, FLOAT8_E4M3FN, or FLOAT8_E8M0.
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
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of self, index, or out is not supported.</td>
      </tr>
      <tr>
      <td>The data type of out is not consistent with that of self.</td>
      </tr>
      <tr>
      <td>self and index have different dimensions.</td>
      </tr>
      <tr>
      <td>The dimension of self, index, or out is greater than 8.</td>
      </tr>
      <tr>
      <td>The shapes of out and index are inconsistent.</td>
      </tr>
      <tr>
      <td>The dimension specified by dim exceeds the dimension range of self, [-self.dim(), self.dim() - 1].</td>
      </tr>
      <tr>
      <td>Except for the dimension specified by dim, the size of index along other dimensions is greater than that of self.</td>
      </tr>
      <tr>
      <td>index is a non-empty tensor and the size of self along the dimension specified by dim is 0.</td>
      </tr>
    </tbody>
    </table>

## aclnnGather

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
      <td>Size of the workspace requested on the Device side, obtained by the first-phase API aclnnGatherGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
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
  - aclnnGather defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather.h"

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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to device memory.
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
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> indexShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> indexHostData = {0, 1, 2, 3, 3, 2, 1, 0};
  std::vector<float> outHostData(8, 0);
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
  // Call the first-phase API of aclnnGather.
  ret = aclnnGatherGetWorkspaceSize(self, dim, index, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnGather.
  ret = aclnnGather(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value, and copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(out);

  // 7. Release device resources.
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
