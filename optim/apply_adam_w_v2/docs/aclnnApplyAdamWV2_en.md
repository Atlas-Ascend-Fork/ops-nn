# aclnnApplyAdamWV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:45:14.434Z -->

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

- Description: Implements the AdamW optimizer function.

- Formula:

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$

  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$

  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnApplyAdamWV2GetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnApplyAdamWV2" to perform the computation.

```cpp
aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(
    aclTensor       *varRef,
    aclTensor       *mRef,
    aclTensor       *vRef,
    aclTensor       *maxGradNormOptionalRef,
    const aclTensor *grad,
    const aclTensor *step,
    float            lr,
    float            beta1,
    float            beta2,
    float            weightDecay,
    float            eps,
    bool             amsgrad,
    bool             maximize,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnApplyAdamWV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnApplyAdamWV2GetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1520px"><colgroup>
    <col style="width: 230px">
    <col style="width: 120px">
    <col style="width: 330px">
    <col style="width: 220px">
    <col style="width: 230px">
    <col style="width: 115px">
    <col style="width: 130px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data type</th>
        <th>Data format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>varRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Weight to be calculated, which is also the output. θ in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>The m parameter in the AdamW optimizer, m in the formula.</td>
        <td>-</td>
        <td>Consistent with the "varRef" parameter.</td>
        <td>ND</td>
        <td>Consistent with the "varRef" parameter.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>vRef(aclTensor*)</td>
        <td>Input/Output</td>
        <td>v parameter in the AdamW optimizer, that is, v in the formula. It cannot be negative.</td>
        <td>-</td>
        <td>Consistent with the "varRef" parameter.</td>
        <td>ND</td>
        <td>Consistent with the "varRef" parameter.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxGradNormOptionalRef(aclTensor*)</td>
        <td>Input/Output</td>
        <td>The input maxGradNormOptionalRef is compared with the updated vRef, and the maximum value is output to maxGradNormOptionalRef.</td>
        <td>This parameter is required when amsgrad is true, and optional when amsgrad is false.</td>
        <td>Consistent with the "varRef" parameter.</td>
        <td>ND</td>
        <td>Consistent with the "varRef" parameter.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>grad (aclTensor*)</td>
        <td>Input</td>
        <td>Gradient data, g<sub>t</sub> in the formula.</td>
        <td>-</td>
        <td>Consistent with the "varRef" parameter.</td>
        <td>ND</td>
        <td>Consistent with the "varRef" parameter.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>step (aclTensor*)</td>
        <td>Input</td>
        <td>Number of iterations, t in the formula. The value must be greater than 0.</td>
        <td>The number of elements is 1.</td>
        <td>INT64, FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>x</td>
      </tr>
      <tr>
        <td>lr (float)</td>
        <td>Input</td>
        <td>Learning rate, η in the formula.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta1 (float)</td>
        <td>Input</td>
        <td>β<sub>1</sub> parameter.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta2 (float)</td>
        <td>Input</td>
        <td>β<sub>2</sub> parameter.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightDecay (float)</td>
        <td>Input</td>
        <td>Weight decay coefficient, λ in the formula.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps (float)</td>
        <td>Input</td>
        <td>Prevents division by zero.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>amsgrad (bool)</td>
        <td>Input</td>
        <td>Whether to use the AMSGrad variant of the algorithm.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maximize (bool)</td>
        <td>Input</td>
        <td>Whether to maximize the parameters.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be applied on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation flow.</td>
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

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Returned when varRef, mRef, vRef, maxGradNormOptionalRef, grad, or step is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>Returned when the data types of varRef, mRef, vRef, maxGradNormOptionalRef, grad, or step are not within the supported range.</td>
      </tr>
      <tr>
      <td>Returned when the data formats of varRef, mRef, vRef, maxGradNormOptionalRef, grad, or step are not within the supported range.</td>
      </tr>
      <tr>
      <td>The shapes of mRef, vRef, grad, and varRef are inconsistent.</td>
      </tr>
      <tr>
      <td>When amsgrad is true, the shapes of maxGradNormOptionalRef and varRef are inconsistent.</td>
      </tr>
      <tr>
      <td>The shape size of step is not 1.</td>
      </tr>
    </tbody>
    </table>

## aclnnApplyAdamWV2

- **Parameters:**

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
      <td>Workspace memory address applied on the side of the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the side of the Device, obtained by the first-phase API aclnnApplyAdamWV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>The stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The data types of the input tensors varRef, mRef, vRef, and grad must be consistent, and the supported data types are FLOAT16, BFLOAT16, and FLOAT32.
- The shapes of the input tensors varRef, mRef, vRef, and grad must remain consistent.
- Deterministic computation:
  - aclnnApplyAdamWV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_w_v2.h"

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
  // Call aclrtMalloc to apply for memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate strides of the contiguous tensor.
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
  // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs as required by the API.
  std::vector<int64_t> varShape = {2, 2};
  std::vector<int64_t> mShape = {2, 2};
  std::vector<int64_t> vShape = {2, 2};
  std::vector<int64_t> gradShape = {2, 2};
  std::vector<int64_t> maxgradShape = {2, 2};
  std::vector<int64_t> stepShape = {1};
  void* varDeviceAddr = nullptr;
  void* mDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  void* maxgradDeviceAddr = nullptr;
  void* stepDeviceAddr = nullptr;
  aclTensor* var = nullptr;
  aclTensor* m = nullptr;
  aclTensor* v = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* maxgrad = nullptr;
  aclTensor* step = nullptr;
  std::vector<float> varHostData = {0, 1, 2, 3};
  std::vector<float> mHostData = {0, 1, 2, 3};
  std::vector<float> vHostData = {0, 1, 2, 3};
  std::vector<float> gradHostData = {0, 1, 2, 3};
  std::vector<float> maxgradHostData = {0, 1, 2, 3};
  std::vector<float> stepHostData = {1};
  bool amsgrad = true;
  bool maximize = true;
  float lr = 1e-3;
  float beta1 = 0.9;
  float beta2 = 0.999;
  float weightDecay = 1e-2;
  float eps = 1e-8;
  // Create the var aclTensor.
  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the m aclTensor.
  ret = CreateAclTensor(mHostData, mShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create v aclTensor.
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create grad aclTensor.
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create maxgrad aclTensor.
  ret = CreateAclTensor(maxgradHostData, maxgradShape, &maxgradDeviceAddr, aclDataType::ACL_FLOAT, &maxgrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create step aclTensor.
  ret = CreateAclTensor(stepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_FLOAT, &step);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnApplyAdamWV2.
  ret = aclnnApplyAdamWV2GetWorkspaceSize(var, m, v, maxgrad, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnApplyAdamWV2.
  ret = aclnnApplyAdamWV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device side to the host side, and modify it based on the specific API definition.
  auto size = GetShapeSize(varShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor, which needs to be modified based on the specific API definition.
  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(grad);
  aclDestroyTensor(maxgrad);
  aclDestroyTensor(step);

  // 7. Release device resources.
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(gradDeviceAddr);
  aclrtFree(maxgradDeviceAddr);
  aclrtFree(stepDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
