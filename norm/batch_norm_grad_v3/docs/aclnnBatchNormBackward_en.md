# aclnnBatchNormBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:57:59.133Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Backward propagation of [aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm.md). Calculates gradients of the input tensor for updating model parameters during backward propagation.

- Formula:

  - When training is true:

    $$
    gradInput = \frac{weight}{ n{\sqrt{saveVar + eps}} }(n * gradOut - \sum^m_{i=0}{gradOut} - \frac{x-saveMean}{\sqrt{saveVar + eps}} \sum^m_{i=0}({gradOut} *\frac{x-saveMean}{ \sqrt{saveVar + eps} } ))
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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnBatchNormBackwardGetWorkspaceSize" API to obtain input parameters and calculate the required workspace size based on the process, and then call the "aclnnBatchNormBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(
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
  aclTensor          *gradInput,
  aclTensor          *gradWeight,
  aclTensor          *gradBias,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnBatchNormBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormBackwardGetWorkspaceSize

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
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOut (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the gradient tensor, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Supported shapes and formats: 2D (format: NC), 3D (format: NCL), 4D (formats: NCHW, NHWC), 5D (formats: NCDHW, NDHWC), and 6D to 8D (format: ND, where the second dimension is fixed to the channel axis).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NHWC, NCDHW, NDHWC, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input tensor of forward propagation, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Data type, shape, and data format must be the same as those of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NHWC, NCDHW, NDHWC, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the weight tensor, corresponding to `weight` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Shape length is equal to the channel axis length in input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the mean calculated during training, corresponding to `runningMean` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape length is equal to the channel axis length in the `input` parameter.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the variance computed during training, corresponding to `runningVar` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Values are non-negative.</li><li>The shape length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveMean (aclTensor*)</td>
      <td>Input</td>
      <td>Saved mean, corresponding to `saveMean` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape length is equal to the channel axis length of input parameter `input`, or is 0 in the inference scenario.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveInvstd (aclTensor*)</td>
      <td>Input</td>
      <td>Output only in training scenarios. Indicates the saved input variance or the reciprocal of the input standard deviation, corresponding to `Var(x)` and the reciprocal of the square root of (Var(x) + eps) in the formula, respectively.</td>
      <td><ul><li>Supports empty tensors.</li><li>Values are non-negative.</li><li>The shape length is equal to the channel axis length in input `input`, or 0 in the inference scenario.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>training (bool)</td>
      <td>Input</td>
      <td>Indicates whether the scenario is training, corresponding to `training` in the formula description.</td>
      <td>true indicates a training scenario, and false indicates an inference scenario.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Value added to the variance to avoid division by zero. Corresponds to `eps` in the corresponding formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask (aclBoolArray*)</td>
      <td>Input</td>
      <td>Mask of the outputs.</td>
      <td>The size is 3. The elements indicate whether to output `gradInput`, `gradWeight`, and `gradBias` respectively. If true, the output is generated; otherwise, the output at the corresponding position is empty.</td>
      <td>BoolArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the input tensor, corresponding to `gradInput` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Optional output. If outputMask[0] is True, it is required; otherwise, it is not output.</li><li>The data type, shape, and data format must be the same as those of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NHWC, NCDHW, NDHWC, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the scaling parameter, corresponding to `gradWeight` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Optional output. Output is required when outputMask[1] is True; otherwise, it is not output.</li><li>The length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the bias parameter, corresponding to `gradBias` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Optional output. Output is required when outputMask[2] is True; otherwise, it is not output.</li><li>The shape length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
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
  - <term>Atlas training products</term> and <term>Atlas inference products</term>:
    - The data type of the parameters `gradOut`, `input`, `weight`, `runningMean`, `runningVar`, `saveMean`, `saveInvstd`, `gradInput`, `gradWeight`, and `gradBias` does not support BFLOAT16.
    - The data types of `weight`, `runningMean`, `runningVar`, `saveMean`, `saveInvstd`, `gradWeight`, and `gradBias` must be consistent with that of `gradOut`.
    - The data format of the parameters `gradOut`, `input`, and `gradInput` does not support NHWC or NDHWC.
    - The `saveInvstd` parameter represents the variance of input.
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The data types of `weight`, `runningMean`, `runningVar`, `saveMean`, `saveInvstd`, `gradWeight`, and `gradBias` must be consistent with that of `gradOut`.
    - The data format of the parameters `gradOut`, `input`, and `gradInput` does not support NHWC or NDHWC.
    - The `saveInvstd` parameter represents the variance of input.
  <!-- end id8 -->

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - When gradOut is FLOAT32, weight, runningMean, runningVar, saveMean, and saveInvstd must be FLOAT32.
    - When gradOut is FLOAT16, weight, terrunningMean, runningVar, saveMean, and saveInvstd must be FLOAT16 or FLOAT32.
    - When gradOut is BFLOAT16, weight, runningMean, runningVar, saveMean, and saveInvstd must be BFLOAT16 or FLOAT32.
    - When weight is FLOAT32, gradWeight must be FLOAT32; when weight is FLOAT16, gradWeight must be FLOAT16 or FLOAT32; when weight is BFLOAT16, gradWeight must be BFLOAT16 or FLOAT32.
    - The data types of `gradWeight` and `gradBias` must be consistent.
    - The data type of `runningVar` must be the same as that of `runningMean`.
    - The data type of `saveInvstd` must be the same as that of `saveMean`.
    - The `saveInvstd` parameter represents the reciprocal of the standard deviation of input.
  <!-- end id9 -->

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
      <td>The passed-in gradOut or input pointer is a null pointer.</td>
    </tr>
    <tr>
      <td>When outputMask[0] is true, the passed-in gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[1] is true and the passed gradWeight is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[2] is true and the passed gradBias is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data type, data format, or shape of input or gradOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>When weight, runningMean, runningVar, saveMean, and saveInvstd are non-empty, their data type, data format, or shape is not within the supported range.</td>
    </tr>
    <tr>
      <td>The length of outputMask is not 3.</td>
    </tr>
    <tr>
      <td>outputMask[0] is true and the data type, data format, or shape of gradInput is not within the supported range.</td>
    </tr>
    <tr>
      <td>outputMask[1] is true and the data type, data format, or shape of gradWeight is not within the supported range.</td>
    </tr>
    <tr>
      <td>outputMask[2] is true and the data type, data format, or shape of gradBias is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape length of weight, runningMean, runningVar, saveMean, saveInvstd, gradWeight (when non-empty), and gradBias (when non-empty) is not equal to the axis length of the channel dimension in the input shape.</td>
    </tr>
    <tr>
      <td>input, gradOut, and gradInput (when non-empty) have inconsistent data formats.</td>
    </tr>
    <tr>
      <td>input, gradOut, and gradInput (when non-empty) have inconsistent data types.</td>
    </tr>
    <tr>
      <td>input, gradOut, and gradInput (when non-empty) have inconsistent shapes, or the shape dimension is greater than 8 or less than 2.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormBackward

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnBatchNormBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b,910" id10 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas training products</term>: aclnnBatchNormBackward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnBatchNormBackward defaults to a deterministic implementation.
  <!-- end id11 -->

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_backward.h"

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
    // Boilerplate, resource initialization
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the Device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy Host-side data to Device-side memory.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
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
    // Create gradOut aclTensor.
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create rMean aclTensor.
    ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create rVar aclTensor.
    ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create sMean aclTensor.
    ret = CreateAclTensor(sMeanHostData, sMeanShape, &sMeanDeviceAddr, aclDataType::ACL_FLOAT, &sMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create sVar aclTensor.
    ret = CreateAclTensor(sVarHostData, sVarShape, &sVarDeviceAddr, aclDataType::ACL_FLOAT, &sVar);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create outMask aclBoolArray.
    bool maskData[3] = {true, true, true};
    outMask = aclCreateBoolArray(&(maskData[0]), 3);
    // Create gradIn aclTensor.
    ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradWeight aclTensor.
    ret = CreateAclTensor(
        gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradBias aclTensor.
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNormBackward API call example
    // 3. Call the CANN operator library API, and change it to the specific API name.
    // Call the first-phase API of aclnnBatchNormBackward.
    ret = aclnnBatchNormBackwardGetWorkspaceSize(
        gradOut, self, weight, rMean, rVar, sMean, sVar, training, eps, outMask, gradIn, gradWeight, gradBias,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnBatchNormBackward.
    ret = aclnnBatchNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, and copy the result from the Device memory to the Host side. Modify it based on the definition of the specific API.
    auto size = GetShapeSize(gradInShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify it based on the definition of the specific API.
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

    // 7. Release device resources. Modify it based on the definition of the specific API.
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
