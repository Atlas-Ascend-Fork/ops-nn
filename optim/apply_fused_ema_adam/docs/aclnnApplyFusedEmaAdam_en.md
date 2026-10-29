# aclnnApplyFusedEmaAdam

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:44:52.027Z -->

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

- **Description**: Implements the FusedEmaAdam fused optimizer function.
- **Formula**:

  $$
  (correction_{\beta_1},correction_{\beta_2})=\begin{cases}
  (1,1),&biasCorrection=False\\
  (1-\beta_1^{step},1-\beta_2^{step}),&biasCorrection=True
  \end{cases}
  $$

  $$
  grad=\begin{cases}
  grad+weightDecay*var,&mode=0\\
  grad,&mode=1
  \end{cases}
  $$

  $$
  m_{out}=\beta_1*m+(1-\beta_1)*grad
  $$

  $$
  v_{out}=\beta_2*v+(1-\beta_2)*grad^2
  $$

  $$
  m_{next}=m_{out}/correction_{\beta_1}
  $$

  $$
  v_{next}=v_{out}/correction_{\beta_2}
  $$

  $$
  denom=\sqrt{v_{next}}+eps
  $$

  $$
  update=\begin{cases}
  m_{next}/denom,&mode=0\\
  m_{next}/denom+weightDecay*var,&mode=1
  \end{cases}
  $$

  $$
  var_{out}=var-lr*update
  $$

  $$
  s_{out}=emaDecay*s+(1-emaDecay)*var_{out}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnApplyFusedEmaAdamGetWorkspaceSize" API first to obtain the input parameters and, based on the computation process, calculate the required workspace size and the executor containing the computation process of the operator. Then call the "aclnnApplyFusedEmaAdam" API to perform the computation.

```cpp
aclnnStatus aclnnApplyFusedEmaAdamGetWorkspaceSize(
    const aclTensor *grad,
    aclTensor       *varRef,
    aclTensor       *mRef,
    aclTensor       *vRef,
    aclTensor       *sRef,
    const aclTensor *step,
    double           lr,
    double           emaDecay,
    double           beta1,
    double           beta2,
    double           eps,
    int64_t          mode,
    bool             biasCorrection,
    double           weightDecay,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnApplyFusedEmaAdam(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnApplyFusedEmaAdamGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 380px">
  <col style="width: 220px">
  <col style="width: 200px">
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
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">grad (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Gradient of the parameters to be updated, corresponding to grad in the formula.</td>
      <td class="tg-0pky"></td>
      <td class="tg-0pky">FLOAT16, BFLOAT16, FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">varRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Parameters to be updated, corresponding to var in the formula.</td>
      <td class="tg-0pky">shape must remain consistent with the input grad.</td>
      <td class="tg-0pky">Remain consistent with grad</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">mRef(aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">First-order moment corresponding to the parameters to be updated, which is m in the corresponding formula.</td>
      <td class="tg-0pky">The shape must remain consistent with that of input grad.</td>
      <td class="tg-0pky">Remain consistent with grad</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">vRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Second moment of the parameters to be updated, which is v in the corresponding formula.</td>
      <td class="tg-0pky">Shape must remain consistent with the input grad.</td>
      <td class="tg-0pky">Remain consistent with grad</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">sRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">EMA weights corresponding to the parameters to be updated, corresponding to s in the formula.</td>
      <td class="tg-0pky">The shape must remain consistent with input grad.</td>
      <td class="tg-0pky">Remain consistent with grad</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">step (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Current number of updates of the optimizer, corresponding to step in the formula.</td>
      <td class="tg-0pky">The shape must be [1,].</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">lr (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Learning rate, η in the formula. 1e-3, 1e-5, and 1e-8 are recommended. The scope is 0-1.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">emaDecay (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Decay rate of the exponential moving average (EMA), corresponding to emaDecay in the formula.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1 (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Coefficient for computing the first momentum (beta1 in the formula). The recommended value is 0.9, and the value range is 0-1.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2 (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Coefficient for computing the second-order momentum, corresponding to the beta2 parameter in the formula. Recommended value: 0.9. Scope: 0-1.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">eps (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Term added to the denominator for numerical stability, corresponding to eps in the formula.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">mode (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Controls whether to apply L2 regularization or weight decay, corresponding to mode in the formula. 1 for adamw, 0 for L2.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">biasCorrection (bool)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Controls whether to perform bias correction, corresponding to biasCorrection in the formula. true indicates that the correction is performed, and false indicates no correction.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">weightDecay (double)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Weight decay, which is weightDecay in the corresponding formula.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">DOUBLE</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation flow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
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
      <td class="tg-0pky">Input and output tensors are null pointers.</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky">161002</td>
      <td class="tg-0pky">The data types and data formats of input and output tensors are not within the supported scope.</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyFusedEmaAdam

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnApplyFusedEmaAdamGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation process.</td>
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

The data types and shapes of the input parameters grad, var, m, v, and s must be consistent.

- Deterministic computation:
  - aclnnApplyFusedEmaAdam defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_fused_ema_adam.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate for resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the inputs and outputs, customized as required by the API.
  // input
  std::vector<float> gradHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> varHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> mHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> vHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> sHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> stepHostData = {10};
  std::vector<int64_t> inputShape = {2, 2, 2};
  std::vector<int64_t> stepShape = {1,};
  void *gradDeviceAddr = nullptr;
  void *varDeviceAddr = nullptr;
  void *mDeviceAddr = nullptr;
  void *vDeviceAddr = nullptr;
  void *sDeviceAddr = nullptr;
  void *stepDeviceAddr = nullptr;
  aclTensor *grad = nullptr;
  aclTensor *var = nullptr;
  aclTensor *m = nullptr;
  aclTensor *v = nullptr;
  aclTensor *s = nullptr;
  aclTensor *step = nullptr;
  ret = CreateAclTensor(gradHostData, inputShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(varHostData, inputShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(mHostData, inputShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, inputShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sHostData, inputShape, &sDeviceAddr, aclDataType::ACL_FLOAT, &s);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(stepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // out, inplace
  std::vector<int64_t> outShape = {2, 2, 2};

  // attr
  float lr = 0.001f;
  float emaDecay = 0.5f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  int64_t mode = 1;
  bool bias = true;
  float weightDecay = 0.5f;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  // Call the aclnnApplyFusedEmaAdam first-phase API.
  ret = aclnnApplyFusedEmaAdamGetWorkspaceSize(grad, var, m, v, s, step, lr, emaDecay, beta1, beta2, eps,
                                               mode, bias, weightDecay, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnApplyFusedEmaAdamGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnApplyFusedEmaAdam.
  ret = aclnnApplyFusedEmaAdam(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnApplyFusedEmaAdam failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output values and copy the results from the device memory to the host memory. Modify the code according to the specific API definition.
  PrintOutResult(outShape, &varDeviceAddr);
  PrintOutResult(outShape, &mDeviceAddr);
  PrintOutResult(outShape, &vDeviceAddr);
  PrintOutResult(outShape, &sDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
  aclDestroyTensor(grad);
  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(s);
  aclDestroyTensor(step);

  // 7. Release device resources.
  aclrtFree(gradDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sDeviceAddr);
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
