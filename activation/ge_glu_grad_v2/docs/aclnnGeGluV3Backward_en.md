# aclnnGeGluV3Backward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T09:46:33.850Z -->

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

Description: Implements backward propagation of [aclnnGeGluV3](../../ge_glu_v2/docs/aclnnGeGluV3.md).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnGeGluV3BackwardGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the computation process of the operator. Then, call the "aclnnGeGluV3Backward" API to execute the computation.

```Cpp
aclnnStatus aclnnGeGluV3BackwardGetWorkspaceSize(
  const aclTensor *gradOutput,
  const aclTensor *self,
  const aclTensor *gelu,
  int64_t          dim,
  int64_t          approximate,
  bool             activateLeft,
  aclTensor       *gradInput,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGeGluV3Backward(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnGeGluV3BackwardGetWorkspaceSize

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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the GeGluV3Backward computation, corresponding to gradOutput in the formula.</td>
      <td><ul><li>Except for the dim dimension, the shape is the same as that of self. The size of the dim dimension is half of that of self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the GeGluV3Backward computation, corresponding to self in the formula.</td>
      <td><ul><li>Except for the dim dimension, the size of other dimensions in shape is the same as that of gradOutput, and the size of the dim dimension is twice that of gradOutput.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gelu (aclTensor*)</td>
      <td>Input</td>
      <td>Input of the GeGluV3Backward computation, corresponding to gelu in the formula.</td>
      <td><ul><li>shape must be the same as gradOutput.</li><li>The data type must be the same as that of self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>-</td>
      <td>Value range: [-self.dim(), self.dim()-1].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>approximate (int64_t)</td>
      <td>Input</td>
      <td>-</td>
      <td>Value range: 0 ('none'), 1 ('tanh'). Atlas inference products support only 1 ('tanh').</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Specifies the direction of the data block on which the activation function operates.</td>
      <td>false indicates that activation is performed on the right.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput(aclTensor*)</td>
      <td>Output</td>
      <td>Output parameter of the GeGluV3Backward computation.</td>
      <td><ul><li>Data type is the same as that of self.</li><li>Shape must be the same as that of self.</li></ul></td>
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
      <td>Returns the op executor that contains the operator computation process.</td>
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
      <td>The data types of gradOutput, self, gelu, and gradInput are not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, self, gelu, and gradInput do not meet the requirements.</td>
    </tr>
    <tr>
      <td>When self.dim()=0, dim is not in the range [-1, 0]; when self.dim()>0, dim is not in the range [-self.dim(), self.dim()-1].</td>
    </tr>
    <tr>
      <td>The value of approximate is not 0 or 1.</td>
    </tr>
  </tbody></table>

## aclnnGeGluV3Backward

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
      <td>Memory address of the workspace allocated on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device side, obtained by the first-phase API aclnnGeGluV3BackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGeGluV3Backward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** selfOrResult)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of contiguous selfOrResult.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs, customized according to the API.
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
    bool activateLeft = false;
    // Call the first-phase API of aclnnGeGluV3Backward.
    ret = aclnnGeGluV3BackwardGetWorkspaceSize(
        gradOutput, self, gelu, dim, approximate, activateLeft, gradInput, &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnGeGluV3BackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGeGluV3Backward.
    ret = aclnnGeGluV3Backward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeGluV3Backward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, and copy the result from the device to the host. Modify it according to the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }

    // 6. Release aclTensor and aclScalar. Modify them according to the specific API definition.
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(self);
    aclDestroyTensor(gelu);
    aclDestroyTensor(gradInput);

    // 7. Release device resources. Modify them according to the specific API definition.
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
