# aclnnCrossEntropyLossGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:14:15.059Z -->

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

- Description: Backward propagation of aclnnCrossEntropyLoss.
- Formula:

  $$
  ignoreMask_{target(t)}=\begin{cases}
  1, &target(t) ≠ ignoreIndex \\
  0, &target(t) = ignoreIndex
  \end{cases}
  $$

  $$
  smoothLossGrad=\begin{cases}
  grad / sum(weight_{target}* ignoreMask) * labelSmoothing / C, &reduction = mean \\
  grad * labelSmoothing / C, &reduction = sum \\
  grad * labelSmoothing / C, &reduction = none
  \end{cases}
  $$

  $$
  lossOutGrad=\begin{cases}
  grad * (1-labelSmoothing) / sum(weight_{target}* ignoreMask) * ignoreMask, &reduction = mean \\
  grad * (1-labelSmoothing) * ignoreMask, &reduction = sum \\
  grad * (1-labelSmoothing) * ignoreMask, &reduction = none
  \end{cases}
  $$

  $$
  nllLossGrad = lossOutGrad * weight_{target}
  $$

  $$
  logSoftmaxGradLossOutSubPart = exp(logProb) * nllLossGrad
  $$

  $$
  predictionsGradLossOut_{ij}=\begin{cases}
  nllLossGrad_i, & j=target(i)  \\
  0, & j ≠ target(i)
  \end{cases}
  $$

  $$
  predictionsGradLossOut = logSoftmaxGradLossOutSubPart - predictionsGradLossOut
  $$

  $$
  smoothLossGrad = smoothLossGrad * ignoreMask
  $$

  $$
  logSoftmaxGradSmoothLoss = smoothLossGrad * weight
  $$

  $$
  predictionsGradSmoothLoss = exp(logProb) * sum(logSoftmaxGradSmoothLoss) - logSoftmaxGradSmoothLoss
  $$

  Output when the zloss scenario is not involved:

  $$
  xGrad_{out} = predictionsGradLossOut + predictionsGradSmoothLoss
  $$

  zloss scenario:

  $
  gradZ=\begin{cases}
  grad + gradZloss, & \text{gradZloss is passed}  \\
  grad, & \text{gradZloss is not passed}
  \end{cases}
  $

  $$
  zlossGrad=\begin{cases}
  gradZ / sum(ignoreMask), & &reduction = mean  \\
  gradZ, & &reduction = sum \\
  gradZ, & &reduction = none
  \end{cases}
  $$

  $$
  lseGrad = 2 * lseSquareScaleForZloss * lseForZloss * ignoreMask * zlossGrad
  $$

  $$
  zlossOutputGrad = exp(logProb) * lseGrad
  $$

  Output in the zloss scenario:

  $$
  xGrad_{out} = xGrad_{out} + zlossOutputGrad
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnCrossEntropyLossGradGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call the "aclnnCrossEntropyLossGrad" API to perform the computation.

```Cpp
aclnnStatus aclnnCrossEntropyLossGradGetWorkspaceSize(
    const aclTensor *gradLoss,
    const aclTensor *logProb,
    const aclTensor *target,
    const aclTensor *weightOptional,
    const aclTensor *gradZlossOptional,
    const aclTensor *lseForZlossOptional,
    char            *reductionOptional,
    int64_t          ignoreIndex,
    double           labelSmoothing,
    double           lseSquareScaleForZloss,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCrossEntropyLossGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnCrossEntropyLossGradGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1378px"><colgroup>
    <col style="width: 200px">
    <col style="width: 120px">
    <col style="width: 272px">
    <col style="width: 227px">
    <col style="width: 149px">
    <col style="width: 112px">
    <col style="width: 153px">
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
        <td>gradLoss (aclTensor*)</td>
        <td>Input</td>
        <td>Gradient of the forward output loss. This parameter corresponds to grad in the formula.</td>
        <td><ul><li>When reductionOptional is none, this parameter must be a 1D tensor.</li><li>When reductionOptional is mean/sum, this parameter must be a 0D tensor.</li></ul></td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N,)<br>N is the batch size.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logProb (aclTensor*)</td>
        <td>Input</td>
        <td>logSoftmax computation result of the forward input. This parameter must be a 2D tensor.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N, C)<br>C is the number of labels and must be greater than 0.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>target (aclTensor*)</td>
        <td>Input</td>
        <td>Class index. Must be a 1D tensor.</td>
        <td>N equals the 0th dimension of input. The value range is [0, C). If ignoreIndex is specified, the value of target can also be equal to ignoreIndex.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightOptional (aclTensor*)</td>
        <td>Optional input</td>
        <td>The shape must be a 1D tensor.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>(C,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gradZlossOptional (aclTensor*)</td>
        <td>Optional input</td>
        <td>Parameter corresponding to gradZloss in the formula. Zloss-related input: if the forward pass has an extra zloss output, the backward pass requires a grad_zloss input.</td>
        <td><ul><li>If reductionOptional is none, a 1D tensor is required.</li><li>If reductionOptional is mean/sum, a 0D tensor is required.</li><li>Currently not supported.</li></ul></td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>lseForZlossOptional (aclTensor*)</td>
        <td>Optional input</td>
        <td>Input related to zloss. If lse_square_scale_for_zloss is non-zero, the lse_for_zloss intermediate result output by forward propagation is passed to backward propagation for lse computation.</td>
        <td><ul><li>Must be a 1D tensor.</li><li>Currently not supported.</li></ul></td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>reductionOptional (char*)</td>
        <td>Input</td>
        <td>Specifies the reduction to apply to the output.</td>
        <td><ul><li>none: No reduction is applied.</li><li>mean: Takes the weighted average of the output.</li><li>sum: Sums the output.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex (int64_t)</td>
        <td>Input</td>
        <td>Specifies the label value to be ignored.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>labelSmoothing (double)</td>
        <td>Input</td>
        <td>Amount of smoothing applied when computing the loss. A floating-point number in the range [0.0, 1.0], where 0.0 indicates no smoothing.</td>
        <td>Currently, only 0.0 is supported.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>lseSquareScaleForZloss (double)</td>
        <td>Input</td>
        <td>zloss-related attribute. 0.0 indicates the native PyTorch branch, and a non-zero value indicates the new zloss branch.</td>
        <td>Currently not supported.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Gradient computation result. It must be a 2D tensor.</td>
        <td>-</td>
        <td>Same as gradLoss</td>
        <td>ND</td>
        <td>(N,C)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Size of the workspace to be applied for on the Device.</td>
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
    </tbody></table>

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
      <td>gradLoss, logProb, target, or out is a null pointer.</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types of gradLoss, logProb, target, weightOptional, gradZlossOptional, or lseForZlossOptional are not in the supported range.</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shape of the input logProb, target, or weightOptional is not in the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnCrossEntropyLossGrad

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
      <td>Workspace memory address allocated on the Device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnCrossEntropyLossGradGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - target supports only class label indices, not probability inputs.
  - gradLoss, logProb, gradZlossOptional, lseForZlossOptional, and xGradOut must have the same data type.
  - The zloss function is currently not supported. Passing related inputs, i.e., gradZlossOptional, lseForZlossOptional, and lseSquareScaleForZloss, will not take effect.

  - Deterministic computation:
    - aclnnCrossEntropyLossGrad defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_loss_grad.h"

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
  // Boilerplate, resource initialization.
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
  // Call aclrtMalloc to allocate memory on the Device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the Host to Device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> gradLossShape = {};
  std::vector<int64_t> logProbShape = {2, 3};
  std::vector<int64_t> targetShape = {2,};
  std::vector<int64_t> weightShape = {3,};
  std::vector<int64_t> xGradShape = {2, 3};
  void* gradLossDeviceAddr = nullptr;
  void* logProbDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* xGradOutDeviceAddr = nullptr;
  aclTensor* gradLoss = nullptr;
  aclTensor* logProb = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradZloss = nullptr;
  aclTensor* lseForZloss = nullptr;
  aclTensor* xGradOut = nullptr;
  std::vector<float> gradLossHostData = {0.1};
  std::vector<float> logProbHostData = {-0.2, -0.2, -0.2, -0.2, -0.2, -0.2};
  std::vector<int64_t> targetHostData = {0, 0};
  std::vector<float> weightHostData = {1.0, 1.0, 1.0};
  std::vector<float> xGradOutHostData = {-0.0091, 0.0409, 0.0409, -0.0091, 0.0409, 0.0409};
  int64_t ignoreIndex = -100;
  double labelSmoothing = 0.0;
  double lseSquareScaleForZloss = 0.0;

  // Create the gradLoss aclTensor.
  ret = CreateAclTensor(gradLossHostData, gradLossShape, &gradLossDeviceAddr, aclDataType::ACL_FLOAT, &gradLoss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the logProb aclTensor.
  ret = CreateAclTensor(logProbHostData, logProbShape, &logProbDeviceAddr, aclDataType::ACL_FLOAT, &logProb);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the xGradOut aclTensor.
  ret = CreateAclTensor(xGradOutHostData, xGradShape, &xGradOutDeviceAddr, aclDataType::ACL_FLOAT, &xGradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. Call the CANN operator library API. Change it to the specific API name.
  // Call the first-phase API of aclnnCrossEntropyLossGrad
  ret = aclnnCrossEntropyLossGradGetWorkspaceSize(gradLoss, logProb, target, weight, gradZloss, lseForZloss, "mean", ignoreIndex, labelSmoothing, lseSquareScaleForZloss, xGradOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrossEntropyLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnCrossEntropyLossGrad
  ret = aclnnCrossEntropyLossGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrossEntropyLossGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait until the task execution completes.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(xGradShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xGradOutDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(gradLoss);
  aclDestroyTensor(logProb);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(xGradOut);

  // 7. Release device resources. Modify the code according to the specific API definition.
  aclrtFree(gradLossDeviceAddr);
  aclrtFree(logProbDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(xGradOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
