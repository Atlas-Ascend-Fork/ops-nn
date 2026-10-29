# aclnnEluBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T09:15:03.780Z -->

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

- Description: Backward calculation of the [aclnnElu](../../elu/docs/aclnnElu&aclnnInplaceElu.md) activation function. Outputs the gradient of the forward input of the ELU activation function.

- Formula: $x$ is an element in selfOrResult.

  - When isResult is True:

    $$
    gradInput = gradOutput *
    \begin{cases}
    scale, \quad x > 0\\
    inputScale \ast (x + \alpha \ast scale),  \quad x \leq 0
    \end{cases}
    $$

  - When isResult is False:

    $$
    gradInput = gradOutput *
    \begin{cases}
    scale, \quad x > 0\\
    inputScale \ast \alpha \ast scale \ast exp(x \ast inputScale), \quad x \leq 0
    \end{cases}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnEluBackwardGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call "aclnnEluBackward" to execute the computation.

```Cpp
aclnnStatus aclnnEluBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclScalar* alpha,
  const aclScalar* scale,
  const aclScalar* inputScale,
  bool             isResult,
  const aclTensor* selfOrResult,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnEluBackward(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnEluBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1370px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
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
      <td>Indicates the gradient of the forward output of the ELU activation function, that is, gradInput in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>alpha (aclScalar*)</td>
      <td>Input</td>
      <td>Indicates the activation coefficient of the ELU activation function, that is, α in the formula.</td>
      <td><ul><li>If isResult is true, α must be greater than or equal to 0.</li><li>The data type must be convertible to FLOAT (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>scale (aclScalar*)</td>
      <td>Input</td>
      <td>Indicates the scaling coefficient of the ELU activation function, scale in the formula.</td>
      <td>The data type must be convertible to FLOAT (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>inputScale (aclScalar*)</td>
      <td>Input</td>
      <td>Indicates the scaling coefficient of the ELU activation function input, inputScale in the formula.</td>
      <td>The data type must be convertible to FLOAT (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isResult (bool)</td>
      <td>Input</td>
      <td>Indicates whether the input passed to the ELU backward computation is the output of the ELU forward computation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>selfOrResult (aclTensor*)</td>
      <td>Input</td>
      <td><ul><li>When isResult is True, this parameter indicates the forward output of the ELU activation function.</li><li>When isResult is False, this parameter indicates the forward input of the ELU activation function.</li></ul></td>
      <td><ul><li>The data type must be consistent with that of gradOutput.</li><li>The shape must be consistent with the shape of gradOutput.</li><li>Supports empty tensors.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
       <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the gradient of the forward input of the ELU activation function, that is, the result of differentiating the input, that is, gradOutput in the formula.</td>
      <td><ul><li>The data type must be convertible to the data type of gradOutput after deduction (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li><li>The shape must be consistent with the shape of gradOutput.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that must be applied for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
   <!-- npu="910,310p" id7 -->
   - <term>Atlas inference products</term>, <term>Atlas training products</term>: The supported data types are FLOAT and FLOAT16.
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
      <td>gradOutput, alpha, scale, inputScale, selfOrResult, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types of gradOutput and selfOrResult are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput and selfOrResult are not consistent.</td>
    </tr>
    <tr>
      <td>The data types of alpha, scale, and inputScale cannot be converted to FLOAT.</td>
    </tr>
    <tr>
      <td>The data type of gradInput is not one that gradOutput can be converted to.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, selfOrResult, and gradInput are not consistent.</td>
    </tr>
     <tr>
      <td>The dimensions of gradOutput, selfOrResult, and gradInput are greater than 8.</td>
    </tr>
    <tr>
      <td>When isResult is True, alpha is less than 0.</td>
    </tr>
  </tbody></table>

## aclnnEluBackward

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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by the first-phase API aclnnEluBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnEluBackward defaults to a deterministic implementation.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_elu_backward.h"

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
                    aclDataType dataType, aclTensor** selfOrResult) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of contiguous selfOrResult.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *selfOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the ACL API.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct custom inputs and outputs based on the API.
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfOrResultShape = {2, 2};
  std::vector<int64_t> gradInputShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfOrResultDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* scale = nullptr;
  aclScalar* inputScale = nullptr;
  aclTensor* selfOrResult = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutputHostData = {-2, -1, 0, 1};
  std::vector<float> selfOrResultHostData = {-2, -1, 0, 1};
  std::vector<float> gradInputHostData = {0, 0, 0, 0};
  float alphaValue = 1.0f;
  float scaleValue = 1.0f;
  float inputScaleValue = 1.0f;
  bool isResult = true;
  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
                        &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create the scale aclScalar.
  scale = aclCreateScalar(&scaleValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scale != nullptr, return ret);
  // Create the inputScale aclScalar.
  inputScale = aclCreateScalar(&inputScaleValue, aclDataType::ACL_FLOAT);
  CHECK_RET(inputScale != nullptr, return ret);
  // Create the selfOrResult aclTensor.
  ret = CreateAclTensor(selfOrResultHostData, selfOrResultShape, &selfOrResultDeviceAddr, aclDataType::ACL_FLOAT,
                        &selfOrResult);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnEluBackward API call example
  // 3. Call the CANN operator library API. Replace it with the specific API name.
  // Call the first-phase API of aclnnEluBackward.
  ret = aclnnEluBackwardGetWorkspaceSize(gradOutput, alpha, scale, inputScale, isResult, selfOrResult, gradInput,
                                         &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnEluBackward.
  ret = aclnnEluBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEluBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify it according to the definition of the specific API.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the actual API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyScalar(alpha);
  aclDestroyScalar(scale);
  aclDestroyScalar(inputScale);
  aclDestroyTensor(selfOrResult);
  aclDestroyTensor(gradInput);

  // 7. Release device resources. Modify the code based on the actual API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfOrResultDeviceAddr);
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
