# aclnnSwishBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T16:17:22.288Z -->

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

- Description: aclnnSwishBackward is the backward propagation of the [aclnnSwish](../../swish/docs/aclnnSwish.md) activation function, used to compute the gradient of the Swish activation function.
- Formula:

  - Swish function formula

  $$
  s(x) = x*\sigma(\beta x)
  $$

  - Derivative implementation of the Swish function formula

  $$

  s^\prime(x)= \beta s(x)+\sigma(\beta x)(1-\beta s(x))= \sigma(\beta x)*(1+\beta x(1-\sigma(\beta x)))

  $$

  $$
  gradInput = gradOutput * s^\prime(x)
  $$

  $$
  \sigma(x) = {\frac{1} {1+{e}^{-x}}}
  $$

  Where $\sigma(x)$ is the Sigmoid function, $s(x)$ is the Swish function, and $s^\prime(x)$ is the derivative of the Swish function.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSwishBackwardGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnSwishBackward" to execute the computation.

```Cpp
aclnnStatus aclnnSwishBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclTensor* self,
  const aclScalar* betaOptional,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSwishBackward(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnSwishBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1380px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 230px">
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
      <td>Gradient of the forward output of the Swish activation function, corresponding to gradOutput in the formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>gradOutput, self, and gradInput must have the same shape.</li><li>gradOutput, self, and gradInput must have the same data type.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Tensor used to compute the activation function, corresponding to x in the formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>gradOutput, self, and gradInput must have the same shape.</li><li>gradOutput, self, and gradInput must have the same data type.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>betaOptional (aclScalar*)</td>
      <td>Input</td>
      <td>Tunable scalar for adjusting the shape and slope of the Swish function, that is, β in the formula.</td>
      <td><ul><li>The data type must be convertible to FLOAT (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</li><li>If betaOptional is a null pointer, the API uses 1.0 for calculation.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput(aclTensor*)</td>
      <td>Output</td>
      <td><ul><li>Output of the backward computation. It is the gradient of the Swish forward input, that is, the derivative of the input.</li><li>gradInput in the formula.</li></ul></td>
      <td><ul><li>Supports null tensors.</li><li>The shapes of gradOutput, self, and gradInput are consistent.</li><li>The data types of gradOutput, self, and gradInput are consistent.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
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
      <td>Returns the op executor, which includes the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

   <!-- npu="910,310p" id7 -->
   - <term>Atlas inference products</term> and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
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
      <td>gradOutput, self, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of gradOutput, self, betaOptional, or gradInput is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, self, and gradInput are different.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, self, and gradInput are different.</td>
    </tr>
  </tbody></table>

## aclnnSwishBackward

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
      <td>Address of the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnSwishBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation flow of the operator.</td>
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
  - aclnnSwishBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swish_backward.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
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
    *tensor = aclCreateTensor(
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

    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> gradOutputShape = {2, 3};
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> gradInputShape = {2, 3};
    void* gradOutDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* self = nullptr;
    aclScalar* betaOptional = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutHostData = {1, 1, 1, 1, 1, 1};
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0};
    float betaValue = 1.1f;

    // Create the gradOut aclTensor.
    ret = CreateAclTensor(gradOutHostData, gradOutputShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the betaOptional aclScalar.
    betaOptional = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
    CHECK_RET(betaOptional != nullptr, return ret);

    // Create the gradInput aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace the API name with the actual one.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnSwishBackward.
    ret = aclnnSwishBackwardGetWorkspaceSize(gradOut, self, betaOptional, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwishBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnSwishBackward.
    ret = aclnnSwishBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwishBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> outData(size, 0);
    ret = aclrtMemcpy(
        outData.data(), outData.size() * sizeof(outData[0]), gradInputDeviceAddr, size * sizeof(outData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify it according to the specific API definition.
    aclDestroyTensor(gradOut);
    aclDestroyTensor(self);
    aclDestroyScalar(betaOptional);
    aclDestroyTensor(gradInput);

    // 7. Release device resources. Modify it according to the specific API definition.
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(selfDeviceAddr);
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
