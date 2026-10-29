# aclnnGeGluBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T09:40:47.804Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs the backward propagation of [aclnnGeGlu](../../ge_glu_v2/docs/aclnnGeGlu.md).
- Formula:

  $$
  \text{gradOutput} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_N \end{bmatrix}, \quad \text{self} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1N} \\ a_{21} & a_{22} & \cdots & a_{2N} \\ \vdots & \vdots & \ddots & \vdots \\ a_{M1} & a_{M2} & \cdots & a_{MN} \end{bmatrix}, \quad gelu = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1N/2} \\ a_{21} & a_{22} & \cdots & a_{2N/2} \end{bmatrix}
  $$

  $$
  \text{dim} = \text{the dimension along which chunk is performed}, \quad \text{approximate = 'approximate'}
  $$

  Output:

  $$
  \text{gradInput} = \begin{bmatrix} b_1 \cdot y\_gelu_1 & b_2 \cdot y\_gelu_2 & \cdots & b_N \cdot y\_gelu_N \end{bmatrix}

  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnGeGluBackwardGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call the "aclnnGeGluBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnGeGluBackwardGetWorkspaceSize(
  const aclTensor *gradOutput,
  const aclTensor *self,
  const aclTensor *gelu,
  int64_t          dim,
  int64_t          approximate,
  aclTensor       *gradInput,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGeGluBackward(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnGeGluBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 280px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <th>Dim(shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the GeGluBackward computation, corresponding to gradOutput in the formula.</td>
      <td><ul><li>The data type is the same as that of self.</li><li>All dims of shape have the same size as self except for the dim dimension, whose size is half of that of self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the GeGluBackward computation, corresponding to self in the formula.</td>
      <td><ul><li>Except for the dim dimension, the sizes of all other dimensions are the same as those of gradOutput, and the size of the dim dimension is twice that of gradOutput.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gelu (aclTensor*)</td>
      <td>Input</td>
      <td>Input parameter for the GeGluBackward computation, corresponding to gelu in the formula.</td>
      <td><ul><li>The data type must be the same as that of self.</li><li>The shape must be the same as that of gradOutput.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>-</td>
      <td>Currently supported values are [-len(input shape), len(input shape)].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>approximate (int64_t)</td>
      <td>Input</td>
      <td>-</td>
      <td>Value range: 0('none'), 1('tanh'). Atlas inference products support only 1('tanh').</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Output parameter of the GeGluBackward computation, that is, gradInput in the formula.</td>
      <td><ul><li>Data type is the same as self.</li><li>Shape must be the same as self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
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
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The supported data types are FLOAT32 and FLOAT16.
  <!-- end id7 -->

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>Parameters gradOutput, self, gelu, and gradInput are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of parameters gradOutput, self, gelu, and gradInput are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of parameters gradOutput, self, gelu, and gradInput do not meet requirements.</td>
    </tr>
    <tr>
      <td>When self.dim()=0, the value of parameter dim is not in the range of [-1, 0]; when self.dim()>0, the value of parameter dim is not in the range of [-self.dim(), self.dim()-1].</td>
    </tr>
    <tr>
      <td>The value of parameter approximate is not 0 or 1.</td>
    </tr>
  </tbody></table>

## aclnnGeGluBackward

- **Parameters:**

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnGeGluBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGeGluBackward defaults to a deterministic implementation.

## Example

The sample code below is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_geglu_backward.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** selfOrResult)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous selfOrResult.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *selfOrResult = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs as required by the API.
    std::vector<int64_t> gradOutputShape = {2, 1};
    std::vector<int64_t> selfShape = {2, 2};
    std::vector<int64_t> geluShape = {2, 1};
    std::vector<int64_t> gradInputShape = {2, 2};
    void* gradOutputDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* geluDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* self = nullptr;
    aclTensor* gelu = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<aclFloat16> gradOutputHostData = {aclFloatToFloat16(-2), aclFloatToFloat16(-1)};
    std::vector<aclFloat16> selfHostData = {aclFloatToFloat16(-2), aclFloatToFloat16(-1), aclFloatToFloat16(0),
                                            aclFloatToFloat16(1)};
    std::vector<aclFloat16> geluHostData = {aclFloatToFloat16(-2), aclFloatToFloat16(-1)};
    int64_t dim = -1;
    int64_t approximate = 1;
    std::vector<aclFloat16> gradInputHostData(4, aclFloatToFloat16(0));
    // Create the gradOutput aclTensor.
    ret = CreateAclTensor(
        gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT16, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gelu aclTensor.
    ret = CreateAclTensor(geluHostData, geluShape, &geluDeviceAddr, aclDataType::ACL_FLOAT16, &gelu);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradInput aclTensor.
    ret =
        CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT16, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnGeGluBackward.
    ret = aclnnGeGluBackwardGetWorkspaceSize(
        gradOutput, self, gelu, dim, approximate, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeGluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGeGluBackward.
    ret = aclnnGeGluBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeGluBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(self);
    aclDestroyTensor(gelu);
    aclDestroyTensor(gradInput);

    // 7. Destroy device resources. Modify the code based on the API definition.
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(selfDeviceAddr);
    aclrtFree(geluDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
