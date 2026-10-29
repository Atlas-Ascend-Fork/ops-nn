# aclnnApplyAdam

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:42:07.372Z -->

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

- **Description:** Implements the Adam optimizer functionality.

- **Formula:**

  $$
  lr_t = lr \times \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
  $$

  $$
  m_t = \beta_1 \times m_{t-1} + (1 - \beta_1) \times g_t
  $$

  $$
  v_t = \beta_2 \times v_{t-1} + (1 - \beta_2) \times g_t^2
  $$

  If use_nesterov = true:
  $$
  var_t = var_{t-1} - lr_t \times \frac{\beta_1 \times m_t + (1 - \beta_1) \times g_t}{\sqrt{v_t} + \epsilon}
  $$

  If use_nesterov = false:
  $$
  var_t = var_{t-1} - lr_t \times \frac{m_t}{\sqrt{v_t} + \epsilon}
  $$

## Function Prototype

Each operator has a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnApplyAdamGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation flow, and then call the "aclnnApplyAdam" API to perform the computation.

```Cpp
aclnnStatus aclnnApplyAdamGetWorkspaceSize(
    aclTensor *varRef,
    const aclTensor *m,
    const aclTensor *v,
    const aclTensor *beta1Power,
    const aclTensor *beta2Power,
    const aclTensor *lr,
    const aclTensor *beta1,
    const aclTensor *beta2,
    const aclTensor *epsilon,
    const aclTensor *grad,
    bool useLocking,
    bool useNesterov,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnApplyAdam(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnApplyAdamGetWorkspaceSize

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
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimensions (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0lax">varRef (aclTensor*)</td>
      <td class="tg-0lax">Input/Output</td>
      <td class="tg-0lax">The weight to be computed is both the input and output, corresponding to var in the formula.</td>
      <td class="tg-0lax"></td>
      <td class="tg-0lax">FLOAT16, BFLOAT16, FLOAT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">m (aclTensor*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">The m parameter in the Adam optimizer, corresponding to m in the formula.</td>
      <td class="tg-0lax">The shape requirements and input varRef must remain consistent.</td>
      <td class="tg-0lax">Remains consistent with varRef</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">v(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The v parameter in the Adam optimizer, that is, v in the formula.</td>
      <td class="tg-0pky">The shape requirements and input varRef remain consistent.</td>
      <td class="tg-0pky">Remains consistent with varRef</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1Power (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The t-th power of beta1.</td>
      <td class="tg-0pky">The shape is required to be [1].</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2Power (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The t-th power of beta2.</td>
      <td class="tg-0pky">The shape is required to be [1].</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">lr (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Learning rate, which is lr in the formula.</td>
      <td class="tg-0pky">Shape required to be [1].</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1 (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta1 parameter.</td>
      <td class="tg-0pky">Shape required to be [1].</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2 (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta2 parameter.</td>
      <td class="tg-0pky">shape is required to be [1].</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">epsilon (aclTensor*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Prevents division by zero.</td>
      <td class="tg-0lax">Shape is required to be [1].</td>
      <td class="tg-0lax">FLOAT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">grad (aclTensor*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Gradient data, g in the formula.</td>
      <td class="tg-0lax">shape requirements and input varRef remain consistent.</td>
      <td class="tg-0lax">Remain consistent with varRef</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">useLocking (bool)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Whether to use the lock mechanism.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">bool</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">useNesterov (bool)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Whether to use Nesterov momentum.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">bool</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the workspace size that needs to be applied on the Device side.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the op executor, which contains the operator computation flow.</td>
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
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">When the incoming computation input parameter is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="5">161002</td>
      <td class="tg-0pky">When the data type of the incoming computation input is not within the supported range.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The data types of the inputs of the incoming computation are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shapes of the inputs of the incoming computation are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The data type of beta1Power, beta2Power, lr, beta1, beta2, or epsilon is not FLOAT32.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shape size of beta1Power, beta2Power, lr, beta1, beta2, or epsilon is not 1.</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyAdam

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
      <td>Memory address of the workspace applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnApplyAdamGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
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
  - aclnnApplyAdam defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam.h"

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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> varShape = {2, 2};
  std::vector<int64_t> mShape = {2, 2};
  std::vector<int64_t> vShape = {2, 2};
  std::vector<int64_t> beta1PowerShape = {1};
  std::vector<int64_t> beta2PowerShape = {1};
  std::vector<int64_t> lrShape = {1};
  std::vector<int64_t> beta1Shape = {1};
  std::vector<int64_t> beta2Shape = {1};
  std::vector<int64_t> epsilonShape = {1};
  std::vector<int64_t> gradShape = {2, 2};
  void* varDeviceAddr = nullptr;
  void* mDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* beta1PowerDeviceAddr = nullptr;
  void* beta2PowerDeviceAddr = nullptr;
  void* lrDeviceAddr = nullptr;
  void* beta1DeviceAddr = nullptr;
  void* beta2DeviceAddr = nullptr;
  void* epsilonDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  aclTensor* var = nullptr;
  aclTensor* m = nullptr;
  aclTensor* v = nullptr;
  aclTensor* beta1Power = nullptr;
  aclTensor* beta2Power = nullptr;
  aclTensor* lr = nullptr;
  aclTensor* beta1 = nullptr;
  aclTensor* beta2 = nullptr;
  aclTensor* epsilon = nullptr;
  aclTensor* grad = nullptr;
  std::vector<float> varHostData = {0, 1, 2, 3};
  std::vector<float> mHostData = {0, 1, 2, 3};
  std::vector<float> vHostData = {0, 1, 2, 3};
  std::vector<float> beta1PowerHostData = {0.431};
  std::vector<float> beta2PowerHostData = {0.992};
  std::vector<float> lrHostData = {0.001};
  std::vector<float> beta1HostData = {0.9};
  std::vector<float> beta2HostData = {0.999};
  std::vector<float> epsilonHostData = {1e-8};
  std::vector<float> gradHostData = {0, 1, 2, 3};
  bool useLocking = false;
  bool useNesterov = false;

  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(mHostData, mShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(beta1PowerHostData, beta1PowerShape, &beta1PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta1Power);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(beta2PowerHostData, beta2PowerShape, &beta2PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta2Power);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lrHostData, lrShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(beta1HostData, beta1Shape, &beta1DeviceAddr, aclDataType::ACL_FLOAT, &beta1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(beta2HostData, beta2Shape, &beta2DeviceAddr, aclDataType::ACL_FLOAT, &beta2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(epsilonHostData, epsilonShape, &epsilonDeviceAddr, aclDataType::ACL_FLOAT, &epsilon);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnApplyAdamGetWorkspaceSize(var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, useLocking, useNesterov, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnApplyAdam(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdam failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(varShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(beta1Power);
  aclDestroyTensor(beta2Power);
  aclDestroyTensor(lr);
  aclDestroyTensor(beta1);
  aclDestroyTensor(beta2);
  aclDestroyTensor(epsilon);
  aclDestroyTensor(grad);

  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(beta1PowerDeviceAddr);
  aclrtFree(beta2PowerDeviceAddr);
  aclrtFree(lrDeviceAddr);
  aclrtFree(beta1DeviceAddr);
  aclrtFree(beta2DeviceAddr);
  aclrtFree(epsilonDeviceAddr);
  aclrtFree(gradDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
