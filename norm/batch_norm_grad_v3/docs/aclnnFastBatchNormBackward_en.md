# aclnnFastBatchNormBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:56:30.353Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

- Description: The backward propagation (high-performance version) of [aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm.md). Computes the gradients of the input tensors for updating model parameters during backward propagation.
- Formula:

  - When training is true:

    $$
    gradInput = \frac{weight}{ n{\sqrt{saveVar + eps}} }(n * gradOut - \sum^m_{i=0}{gradOut} - \frac{x-saveMean}{ \sqrt{saveVar + eps} }\sum^m_{i=0}({gradOut} *\frac{x-saveMean}{ \sqrt{saveVar + eps} } ))
    $$

    $$
    gradWeight = \sum^m_{i=0}[{gradOut} * (x - saveMean)] * \frac{1}{ \sqrt{saveVar + eps} }
    $$

    $$
    gradBias = \sum^m_{i=0}{gradOut}
    $$

  - When training is false:

    $$
    gradInput = gradOut * \frac{1}{ \sqrt{runningVar + eps} } * weight
    $$

    $$
    gradWeight = \sum^m_{i=0}[{gradOut} * (x - runningMean)] * \frac{1}{ \sqrt{runningVar + eps} }
    $$

    $$
    gradBias = \sum^m_{i=0}{gradOut}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnFastBatchNormBackwardGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the process, and then call "aclnnFastBatchNormBackward" to perform the computation.

```Cpp
aclnnStatus aclnnFastBatchNormBackwardGetWorkspaceSize(
  const aclTensor    *gradOut,
  const aclTensor    *input,
  const aclTensor    *weight,
  const aclTensor    *runningMean,
  const aclTensor    *runningVar,
  const aclTensor    *saveMean,
  const aclTensor    *saveInvstd,
  bool                training,
  double              eps,
  const aclBoolArray *outputMask,
  int                 version,
  aclTensor          *gradInput,
  aclTensor          *gradWeight,
  aclTensor          *gradBias,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnFastBatchNormBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnFastBatchNormBackwardGetWorkspaceSize

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
      <td>gradOut (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the gradient tensor, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Supported shapes and formats: 2D (format: NC), 3D (format: NCL), 4D (format: NCHW), 5D (format: NCDHW), and 6-8D (format: ND, where the second dimension is fixed to the channel axis).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NCDHW, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor in forward propagation, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type, shape, and data format must be consistent with those of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NCDHW, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor, corresponding to `weight` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape length is equal to the channel axis length of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the mean calculated during training. Corresponds to `runningMean` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar (aclTensor*)</td>
      <td>Input</td>
      <td>Variance calculated during training. Corresponds to `runningVar` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Values are non-negative.</li><li>The shape length is equal to the channel axis length in the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveMean (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the saved mean, corresponding to `saveMean` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveInvstd (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the saved reciprocal of the standard deviation, corresponding to the reciprocal of the square root of (Var(x) + eps) in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Values are non-negative.</li><li>The shape length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>training (bool)</td>
      <td>Input</td>
      <td>Indicates whether the training scenario is flagged. It corresponds to `training` in the formula description.</td>
      <td>true indicates the training scenario, and false indicates the inference scenario.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Represents the value added to the variance to avoid division by zero. It corresponds to `eps` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask (aclBoolArray*)</td>
      <td>Input</td>
      <td>Represents the mask of the outputs.</td>
      <td>The size is 3, indicating whether to output `gradInput`, `gradWeight`, and `gradBias` respectively. If a value is true, the corresponding output is generated. Otherwise, an empty tensor is returned at the corresponding position.</td>
      <td>BoolArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>version (int)</td>
      <td>Input</td>
      <td>Represents the algorithm version used inside the operator.</td>
      <td>Currently supported optional values: 0 and 1. Default value: 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput(aclTensor*)</td>
      <td>Output</td>
      <td>Represents the gradient of the input Tensor, which is `gradInput` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Optional output. If outputMask[0] is True, output is required; otherwise, no output.</li><li>The data type, shape, and data format must be consistent with those of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NCDHW, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the scaling parameter, corresponding to `gradWeight` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Optional output. If outputMask[1] is True, output is required; otherwise, no output.</li><li>The length is equal to the axis length of channel in the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the bias parameter, corresponding to `gradBias` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Optional output. If outputMask[2] is True, this output is required; otherwise, it is not output.</li><li>The shape length is equal to the channel axis length of input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to allocate on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    - The data types of `weight`, `runningMean`, `runningVar`, `saveMean`, `saveInvstd`, `gradWeight`, and `gradBias` must be consistent with that of `gradOut`.
  <!-- end id7 -->

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
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
      <td>The gradOut and input pointers are null pointers.</td>
    </tr>
    <tr>
      <td>When outputMask[0] is true, the gradInput pointer is a null pointer.</td>
    </tr>
    <tr>
      <td>When outputMask[1] is true, the gradWeight pointer is a null pointer.</td>
    </tr>
    <tr>
      <td>When outputMask[2] is true, the gradBias pointer is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data type, data format, or shape of input and gradOut is out of the supported range.</td>
    </tr>
    <tr>
      <td>When weight, runningMean, runningVar, saveMean, or saveInvstd is non-empty and the data type, data format, or shape is not within the supported range.</td>
    </tr>
    <tr>
      <td>The length of outputMask is not 3.</td>
    </tr>
    <tr>
      <td>When outputMask[0] is true and the data type, data format, or shape of gradInput is not within the supported range.</td>
    </tr>
    <tr>
      <td>When outputMask[1] is true and the data type, data format, or shape of gradWeight is not within the supported range.</td>
    </tr>
    <tr>
      <td>When outputMask[2] is true and the data type, data format, or shape of gradBias is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape length of weight, runningMean, runningVar, saveMean, saveInvstd, gradWeight (when non-empty), or gradBias (when non-empty) is not equal to the channel axis length in the input shape.</td>
    </tr>
    <tr>
      <td>The data formats of input, gradOut, and gradInput (when non-empty) are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of input, gradOut, and gradInput (when non-empty) are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of input, gradOut, and gradInput (when non-empty) are inconsistent, or the number of dimensions in the shape is greater than 8 or less than 2.</td>
    </tr>
  </tbody></table>

## aclnnFastBatchNormBackward

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
      <td>The workspace memory address on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size on the Device, obtained by the first-phase API aclnnFastBatchNormBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFastBatchNormBackward defaults to a deterministic implementation.

## Example

The sample code is as follows for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fast_batch_norm_backward.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
      aclrtResetDevice(deviceId); aclFinalize(); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the ACL API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> gradOutShape = {1, 2, 4};
  std::vector<int64_t> selfShape = {1, 2, 4};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> rMeanShape = {2};
  std::vector<int64_t> rVarShape = {2};
  std::vector<int64_t> sMeanShape = {2};
  std::vector<int64_t> sVarShape = {2};
  std::vector<int64_t> gradInShape = {1, 2, 4};
  std::vector<int64_t> gradWeightShape = {2};
  std::vector<int64_t> gradBiasShape = {2};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* rMeanDeviceAddr = nullptr;
  void* rVarDeviceAddr = nullptr;
  void* sMeanDeviceAddr = nullptr;
  void* sVarDeviceAddr = nullptr;
  void* outMaskDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;
  void* gradBiasDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* rMean = nullptr;
  aclTensor* rVar = nullptr;
  aclTensor* sMean = nullptr;
  aclTensor* sVar = nullptr;
  aclBoolArray* outMask = nullptr;
  aclTensor* gradIn = nullptr;
  aclTensor* gradWeight = nullptr;
  aclTensor* gradBias = nullptr;
  std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weightHostData = {1, 1};
  std::vector<float> rMeanHostData = {0, 0};
  std::vector<float> rVarHostData = {1, 1};
  std::vector<float> sMeanHostData = {0, 0};
  std::vector<float> sVarHostData = {1, 1};
  std::vector<float> gradInHostData(8, 0);
  std::vector<float> gradWeightHostData(2, 0);
  std::vector<float> gradBiasHostData(2, 0);
  bool training = true;
  double eps = 1e-5;
  // Create the gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the rMean aclTensor.
  ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the rVar aclTensor.
  ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the sMean aclTensor.
  ret = CreateAclTensor(sMeanHostData, sMeanShape, &sMeanDeviceAddr, aclDataType::ACL_FLOAT, &sMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the sVar aclTensor.
  ret = CreateAclTensor(sVarHostData, sVarShape, &sVarDeviceAddr, aclDataType::ACL_FLOAT, &sVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the outMask aclBoolArray.
  bool maskData[3] = {true, true, true};
  outMask = aclCreateBoolArray(&(maskData[0]), 3);
  // Create the gradIn aclTensor.
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradWeight aclTensor.
  ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradBias aclTensor.
  ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Example of the aclnnFastBatchNormBackward API call.
  // 3. Call the CANN operator library API. Change it to the specific API name.
  // Call the first-phase API of aclnnFastBatchNormBackward.
  ret = aclnnFastBatchNormBackwardGetWorkspaceSize(gradOut, self, weight, rMean, rVar, sMean, sVar, training, eps, outMask, 0, gradIn, gradWeight, gradBias, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFastBatchNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFastBatchNormBackward.
  ret = aclnnFastBatchNormBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFastBatchNormBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from the device memory to the host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(weight);
  aclDestroyTensor(rMean);
  aclDestroyTensor(rVar);
  aclDestroyTensor(sMean);
  aclDestroyTensor(sVar);
  aclDestroyBoolArray(outMask);
  aclDestroyTensor(gradIn);
  aclDestroyTensor(gradWeight);
  aclDestroyTensor(gradBias);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(rMeanDeviceAddr);
  aclrtFree(rVarDeviceAddr);
  aclrtFree(sMeanDeviceAddr);
  aclrtFree(sVarDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  aclrtFree(gradWeightDeviceAddr);
  aclrtFree(gradBiasDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
