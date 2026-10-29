# aclnnNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:22:45.619Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Returns the matrix norm or vector norm of a given tensor.

- Formula: Supports computing 0/1/2-norms, the infinity norm, and other norms where `p` is of the float type.
  - 0-norm:

    $$
    \Vert x \Vert = \sum_{i=1}^{N}{(x_i ≠ 0)}
    $$

  - 1-norm:

    $$
    \Vert x \Vert = \sum_{i=1}^{N}{\vert x_i \vert}
    $$

  - 2-norm (default):

    $$
    \Vert x \Vert_2 = (\sum_{i=1}^{N}{\vert x_i \vert^2})^{\frac{1}{2}}
    $$

  - infinity norm:

    $$
    \Vert x \Vert_\infty = \max\limits_{i}{\vert x_i \vert}
    $$

    $$
    \Vert x \Vert_{-\infty} = \min\limits_{i}{\vert x_i \vert}
    $$

  - p-norm:

    $$
    \Vert x \Vert_p = (\sum_{i=1}^{N}{\vert x_i \vert^p})^{\frac{1}{p}}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnNormGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnNorm" to perform the computation.

```Cpp
aclnnStatus aclnnNormGetWorkspaceSize(
  const aclTensor     *self,
  const aclScalar     *pScalar,
  const aclIntArray   *dim,
  bool                 keepdim,
  aclTensor           *out,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnNorm(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnNormGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>x in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pScalar (aclScalar*)</td>
      <td>Input</td>
      <td>Represents the norm type, p in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim (aclIntArray*)</td>
      <td>Input</td>
      <td>Dimensions for calculating the norm of self.</td>
      <td>dim cannot be a null pointer. Elements in dim support [-N, N-1] and cannot be duplicated, where N is the dimension of self and negative values are supported. When dim is an empty array (array length is 0), the norm is calculated over all dimensions of self.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim (bool)</td>
      <td>Input</td>
      <td>Determines whether the output tensor retains the axis specified by dim.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out(aclTensor*)</td>
      <td>Output</td>
      <td>Output in the formula.</td>
      <td><ul><li>If keepdim is true, the shape must be the same as that of self except that the size on the dimension specified by dim is 1; when dim is an empty array, each dimension size of out is 1.</li><li>If keepdim is false, the reduced dimension is not retained, and the shapes of other dimensions must be the same as that of self; when dim is an empty array, out is a 0-dimension tensor.</li></ul></td>
      <td>        FLOAT32, FLOAT16, BFLOAT16
      </td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>The self, pScalar, dim, or out parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data format of self is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types and data formats of dim and keepdim do not meet the API input requirements.</td>
    </tr>
    <tr>
      <td>dim exceeds [-N, N-1] (N indicates the dimension of self).</td>
    </tr>
    <tr>
      <td>The values in dim are duplicated.</td>
    </tr>
    <tr>
      <td>The shape of self or out exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>The shape of out is not equal to the shape derived from self, dim, and keepdim.</td>
    </tr>
  </tbody></table>

## aclnnNorm

- **Parameters**

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
      <td>Memory address of the workspace allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor. It contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation

  <!-- npu="A3,910b,910,310p,310b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: aclnnNorm defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnNorm defaults to a deterministic implementation.
  <!-- end id8 -->

<!-- npu="950" id9 -->
- Batch consistency:
  - <term>Ascend 950PR&950DT products</term>: Defaults to a non-Batch-consistency implementation. Batch consistency can be enabled through aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3). After it is enabled, the computation results of non-reduction axes are independent of the batch size and position; reduction axes do not support Batch consistency. Enabling Batch consistency may degrade performance.
<!-- end id9 -->

## Example

The sample code is as follows and is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_norm.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate for resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId); aclFinalize(); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API reference.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Customize them based on the API definition.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> outShape = {1, 2};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* pScalar = nullptr;
    aclIntArray* dim = nullptr;

    std::vector<float> selfHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
    std::vector<float> outHostData = {0.0, 0.0};
    std::vector<int64_t> dimData = {0};
    float pValue = 0.0f;
    bool keepdim = true;

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the p aclTensor.
    pScalar = aclCreateScalar(&pValue, aclDataType::ACL_FLOAT);
    CHECK_RET(pScalar != nullptr, return ret);
    // Create the dim aclIntArray.
    dim = aclCreateIntArray(dimData.data(), 1);
    CHECK_RET(dim != nullptr, return ret);

    // 3. Call the CANN operator library API. Modify it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnNorm.
    ret = aclnnNormGetWorkspaceSize(self, pScalar, dim, keepdim, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnNorm.
    ret = aclnnNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, copy the result from device memory to the host side, and modify the code based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar, and modify the code based on the specific API definition.
    aclDestroyTensor(self);
    aclDestroyIntArray(dim);
    aclDestroyScalar(pScalar);
    aclDestroyTensor(out);

    // 7. Release device resources, and modify the code based on the specific API definition.
    aclrtFree(selfDeviceAddr);
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
