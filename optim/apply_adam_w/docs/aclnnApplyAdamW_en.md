# aclnnApplyAdamW

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:49:35.445Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- **Description:** Implements the AdamW optimizer function.

- **Formula:**

  $$
  g_t=\begin{cases}-g_t
  & \text{ if } maximize= true\\
  g_t  & \text{ if } maximize=false
  \end{cases}
  $$

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \beta_{1}^{t}=\beta_{1}^{t-1}\times\beta_{1}
  $$

  $$
  \beta_{2}^{t}=\beta_{2}^{t-1}\times\beta_{2}
  $$

  $$
  v_t=\begin{cases}\max(maxGradNorm, v_t)
  & \text{ if } amsgrad = true\\
  v_t  & \text{ if } amsgrad = false
  \end{cases}
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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnApplyAdamWGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnApplyAdamW" API to perform the computation.

```Cpp
aclnnStatus aclnnApplyAdamWGetWorkspaceSize(
    aclTensor*       varRef,
    aclTensor*       mRef,
    aclTensor*       vRef,
    const aclTensor* beta1Power,
    const aclTensor* beta2Power,
    const aclTensor* lr,
    const aclTensor* weightDecay,
    const aclTensor* beta1,
    const aclTensor* beta2,
    const aclTensor* eps,
    const aclTensor* grad,
    const aclTensor* maxGradNormOptional,
    bool             amsgrad,
    bool             maximize,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnApplyAdamW(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnApplyAdamWGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1428px"><colgroup>
  <col style="width: 230px">
  <col style="width: 120px">
  <col style="width: 330px">
  <col style="width: 230px">
  <col style="width: 138px">
  <col style="width: 115px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">data type</th>
      <th class="tg-0pky">data format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0lax">varRef (aclTensor*)</td>
      <td class="tg-0lax">Input/Output</td>
      <td class="tg-0lax">Weight input to be computed and also output; θ in the formula.</td>
      <td class="tg-0lax"></td>
      <td class="tg-0lax">FLOAT16, BFLOAT16, FLOAT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">mRef (aclTensor*)</td>
      <td class="tg-0lax">Input/Output</td>
      <td class="tg-0lax">Parameter m in the AdamW optimizer; m in the formula.</td>
      <td class="tg-0lax">The shape is required to remain consistent with the input varRef.</td>
      <td class="tg-0lax">Remain consistent with varRef</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">vRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">v parameter in the AdamW optimizer, corresponding to v in the formula.</td>
      <td class="tg-0pky">The shape must remain consistent with that of the input varRef.</td>
      <td class="tg-0pky">Remain consistent with varRef</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1Power (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The beta1^(t-1) parameter.</td>
      <td class="tg-0pky">The shape is required to be [1].</td>
      <td class="tg-0pky">Remain consistent with varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2Power (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta2^(t-1) parameter.</td>
      <td class="tg-0pky">shape required to be [1].</td>
      <td class="tg-0pky">Remain consistent with varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">lr (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Learning rate, η in the formula, typically 1e-3, 1e-6, or 1e-9.</td>
      <td class="tg-0pky">The shape is required to be [1].</td>
      <td class="tg-0pky">Remain consistent with varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightDecay (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight decay coefficient.</td>
      <td class="tg-0pky">The shape is required to be [1].</td>
      <td class="tg-0pky">Remain consistent with varRef</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1 (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta1 parameter.</td>
      <td class="tg-0pky">The shape is required to be [1].</td>
      <td class="tg-0pky">Remain consistent with varRef</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2 (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta2 parameter.</td>
      <td class="tg-0pky">The shape is required to be [1].</td>
      <td class="tg-0pky">Remain consistent with varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">eps (aclTensor*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Prevents division by zero.</td>
      <td class="tg-0lax">The shape is required to be [1].</td>
      <td class="tg-0lax">Must remain consistent with varRef.</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">grad (aclTensor*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Gradient data, that is, g in the formula.</td>
      <td class="tg-0lax">The shape must remain consistent with input varRef.</td>
      <td class="tg-0lax">Remains consistent with varRef.</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">maxGradNormOptional(aclTensor*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">If amsgrad=true, saves the maximum of maxGradNorm and v in vRef. If amsgrad=false, no additional computation is performed on the value in vRef. This parameter is required when amsgrad is true and optional when amsgrad is false.</td>
      <td class="tg-0lax">The shape must remain consistent with input varRef.</td>
      <td class="tg-0lax">Remain consistent with varRef.</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">amsgrad (bool)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Whether to use the maxGradNormOptional variable.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">bool</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">maximize (bool)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Specifies whether to negate the gradient grad, applying gradient ascent to optimize the weights for maximizing the loss function.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">bool</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the workspace to be applied by the user on the Device side.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the op executor, which contains the computation flow of the operator.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 270px">
  <col style="width: 130px">
  <col style="width: 750px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky" rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky" rowspan="2">161001</td>
      <td class="tg-0pky">When the incoming computation input parameter is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky">When amsgrad is true and maxGradNormOptional is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="5">161002</td>
      <td class="tg-0pky">When the data type of the incoming computation input is not in the supported range.</td>
    </tr>
    <tr>
      <td class="tg-0pky">When the data types of the incoming computation inputs are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">When the shapes of the incoming computation inputs are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">When amsgrad is true and the data type or shape of maxGradNormOptional and varRef is inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shape size of beta1Power, beta2Power, lr, weightDecay, beta1, beta2, or eps is not 1.</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyAdamW

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
      <td>Workspace size requested on the Device side, obtained by the first-phase API aclnnApplyAdamWGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnApplyAdamW defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_w.h"

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
  // Boilerplate: AscendCL initialization
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
  // Call aclrtMalloc to apply for device-side memory.
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
  // 1. (Boilerplate) device/stream initialization, refer to the AscendCL external API list.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Custom construction based on the API is required.
  std::vector<int64_t> varShape = {2, 2};
  std::vector<int64_t> mShape = {2, 2};
  std::vector<int64_t> vShape = {2, 2};
  std::vector<int64_t> beta1PowerShape = {1};
  std::vector<int64_t> beta2PowerShape = {1};
  std::vector<int64_t> lrShape = {1};
  std::vector<int64_t> weightDecayShape = {1};
  std::vector<int64_t> beta1Shape = {1};
  std::vector<int64_t> beta2Shape = {1};
  std::vector<int64_t> epsShape = {1};
  std::vector<int64_t> gradShape = {2, 2};
  std::vector<int64_t> maxgradShape = {2, 2};
  void* varDeviceAddr = nullptr;
  void* mDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* beta1PowerDeviceAddr = nullptr;
  void* beta2PowerDeviceAddr = nullptr;
  void* lrDeviceAddr = nullptr;
  void* weightDecayDeviceAddr = nullptr;
  void* beta1DeviceAddr = nullptr;
  void* beta2DeviceAddr = nullptr;
  void* epsDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  void* maxgradDeviceAddr = nullptr;
  aclTensor* var = nullptr;
  aclTensor* m = nullptr;
  aclTensor* v = nullptr;
  aclTensor* beta1Power = nullptr;
  aclTensor* beta2Power = nullptr;
  aclTensor* lr = nullptr;
  aclTensor* weightDecay = nullptr;
  aclTensor* beta1 = nullptr;
  aclTensor* beta2 = nullptr;
  aclTensor* eps = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* maxgrad = nullptr;
  std::vector<float> varHostData = {0, 1, 2, 3};
  std::vector<float> mHostData = {0, 1, 2, 3};
  std::vector<float> vHostData = {0, 1, 2, 3};
  std::vector<float> beta1PowerHostData = {0.431};
  std::vector<float> beta2PowerHostData = {0.992};
  std::vector<float> lrHostData = {0.001};
  std::vector<float> weightDecayHostData = {0.01};
  std::vector<float> beta1HostData = {0.9};
  std::vector<float> beta2HostData = {0.999};
  std::vector<float> epsHostData = {1e-8};
  std::vector<float> gradHostData = {0, 1, 2, 3};
  std::vector<float> maxgradHostData = {0, 1, 2, 3};
  bool amsgrad = true;
  bool maximize = true;
  // Create the var aclTensor.
  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the m aclTensor.
  ret = CreateAclTensor(mHostData, mShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create v aclTensor.
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create beta1Power aclTensor.
  ret = CreateAclTensor(beta1PowerHostData, beta1PowerShape, &beta1PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta1Power);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create beta2Power aclTensor.
  ret = CreateAclTensor(beta2PowerHostData, beta2PowerShape, &beta2PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta2Power);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create lr aclTensor.
  ret = CreateAclTensor(lrHostData, lrShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create weightDecay aclTensor.
  ret = CreateAclTensor(weightDecayHostData, weightDecayShape, &weightDecayDeviceAddr, aclDataType::ACL_FLOAT, &weightDecay);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create beta1 aclTensor.
  ret = CreateAclTensor(beta1HostData, beta1Shape, &beta1DeviceAddr, aclDataType::ACL_FLOAT, &beta1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create beta2 aclTensor.
  ret = CreateAclTensor(beta2HostData, beta2Shape, &beta2DeviceAddr, aclDataType::ACL_FLOAT, &beta2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create eps aclTensor.
  ret = CreateAclTensor(epsHostData, epsShape, &epsDeviceAddr, aclDataType::ACL_FLOAT, &eps);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create grad aclTensor.
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create maxgrad aclTensor.
  ret = CreateAclTensor(maxgradHostData, maxgradShape, &maxgradDeviceAddr, aclDataType::ACL_FLOAT, &maxgrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and replace it with the actual API name
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnApplyAdamW
  ret = aclnnApplyAdamWGetWorkspaceSize(var, m, v, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad, maxgrad, amsgrad, maximize, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnApplyAdamW
  ret = aclnnApplyAdamW(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamW failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait for the task to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result on device memory to the host side, and modify it based on the specific API definition
  auto size = GetShapeSize(varShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor, and modify it based on the specific API definition
  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(beta1Power);
  aclDestroyTensor(beta2Power);
  aclDestroyTensor(lr);
  aclDestroyTensor(weightDecay);
  aclDestroyTensor(beta1);
  aclDestroyTensor(beta2);
  aclDestroyTensor(eps);
  aclDestroyTensor(grad);
  aclDestroyTensor(maxgrad);

  // 7. Release device resources
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(beta1PowerDeviceAddr);
  aclrtFree(beta2PowerDeviceAddr);
  aclrtFree(lrDeviceAddr);
  aclrtFree(weightDecayDeviceAddr);
  aclrtFree(beta1DeviceAddr);
  aclrtFree(beta2DeviceAddr);
  aclrtFree(epsDeviceAddr);
  aclrtFree(gradDeviceAddr);
  aclrtFree(maxgradDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
