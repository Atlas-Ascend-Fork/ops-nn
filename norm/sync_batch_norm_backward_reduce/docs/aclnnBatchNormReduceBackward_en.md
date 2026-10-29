# aclnnBatchNormReduceBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:38:40.500Z -->

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

- Description:

  It is primarily used to compute the gradient of the BatchNorm operation during backward propagation and perform reduction operations on some intermediate results to optimize computational efficiency. The computation results are as follows:
  - Computes the gradient of the loss function l with respect to the scale weight γ ($\frac{\partial l}{\partial γ}$).
  - Computes the gradient of the loss function l with respect to the offset β ($\frac{\partial l}{\partial β}$).
  - Derives the intermediate quantities sumDy and sumDyXmu required for computing $\frac{\partial l}{\partial x_i}$ from the deviation d<sub>yi</sub> of the loss function l with respect to the output (y<sub>i</sub>). Here ($\frac{\partial l}{\partial x_i}$) is the gradient of the loss function l with respect to each input (x<sub>i</sub>) of the corresponding layer.

- Formula:

  $$
  gradWeight = \frac{\partial l}{\partial γ} = \sum^m_{i=0} \frac{\partial l}{\partial y_i} \cdot \hat{(x_i)} = \frac{1}{{\sqrt{σ^2_B + eps}}} \cdot \sum^m_{i=0} \frac{\partial l}  {\partial y_i} \cdot (x_i-μ_B)
  $$

  $$
  gradBias = \frac{\partial l}{\partial β} = \sum^m_{i=0} \frac{\partial l}{\partial y_i}
  $$

  $$
  sumDy = sum(l, y_i) = \displaystyle \sum^m_{i=0} \frac{\partial l}{\partial y_i}
  $$

  $$
  sumDyXmu = sum(l, y_i, x_i, μ_B) = \displaystyle \sum^m_{i=0} \frac{\partial l}{\partial y_i} \cdot (x_i-μ_B)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnBatchNormReduceBackwardGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process. Then, call the "aclnnBatchNormReduceBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnBatchNormReduceBackwardGetWorkspaceSize(
  const aclTensor* gradOut,
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  const aclTensor* weight,
  const bool       inputG,
  const bool       weightG,
  const bool       biasG,
  aclTensor*       sumDy,
  aclTensor*       sumDyXmu,
  aclTensor*       gradWeight,
  aclTensor*       gradBias,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormReduceBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormReduceBackwardGetWorkspaceSize

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
      <td>Indicates the gradient tensor, corresponding to `<math><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>l</mi></mrow>/<mrow><mi mathvariant="normal">∂</mi><mi>y</mi></mrow></mfrac></math>` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Data type and shape need to be consistent with `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>By default, the second dimension is the channel axis, and the value of the channel axis cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the mean, corresponding to `μ<sub>B</sub>` in the formula.</td>
      <td><ul><li>The shape length equals the channel axis length in parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the reciprocal of the standard deviation, corresponding to <math><mfrac><mn>1</mn><msqrt><msubsup><mi>σ</mi><mi>B</mi><mn>2</mn></msubsup><mo>+</mo><mi>eps</mi></msqrt></mfrac></math> in the formula.</td>
      <td><ul><li>The length of shape is equal to the channel axis length in the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the weight tensor, corresponding to `γ` in the formula.</td>
      <td><ul><li>The shape length is equal to the channel axis length in the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputG (bool)</td>
      <td>Input</td>
      <td>Indicates the output mask, flag indicating whether output `sumDy` and `sumDyXmu` are required.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightG (bool)</td>
      <td>Input</td>
      <td>Indicates the output mask, flag indicating whether output `gradWeight` is required.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasG (bool)</td>
      <td>Input</td>
      <td>Indicates the output mask, flag indicating whether to output `gradBias`.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sumDy (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the accumulated sum of the forward output gradient `gradOut`, which is `sumDy` in the corresponding formula.</td>
      <td><ul><li>Optional output. If `inputG` is True, then output. The size of the shape must be equal to the channel axis length of `input`.</li><li>The data format must remain consistent with that of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDyXmu (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates output of the sum of products of the forward output gradient `gradOut` and the centered input data `(x-μ<sub>B</sub>)`, corresponding to `sumDyXmu` in the formula.</td>
      <td><ul><li>Optional output; output if `inputG` is True. The shape size needs to be equal to the channel axis length of `input`.</li><li>The data format remains consistent with that of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the gradient of the scale parameter, corresponding to `gradWeight` in the formula.</td>
      <td><ul><li>Optional output, output if `weightG` is True. The shape size needs to be equal to the channel axis length of `input`.</li><li>The data format remains consistent with that of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the gradient of the bias parameter, corresponding to `gradBias` in the formula.</td>
      <td><ul><li>Optional output. If `biasG` is True, then output. The size of the shape needs to be equal to the channel axis length of `input`.</li><li>The data format remains consistent with `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be applied on the side of the Device by the user.</td>
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
  - <term>Atlas training products</term>, <term>Atlas inference products</term>: The data type of gradOut, input, mean, invstd,weight,sumDy, sumDyXmu, gradWeight, and gradBias does not support BFLOAT16.
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
      <td>gradOut, input, mean, or invstd is a null pointer.</td>
    </tr>
    <tr>
      <td>When inputG is true, sumDy or sumDyXmu is a null pointer.</td>
    </tr>
    <tr>
      <td>When weightG is true, gradWeight is a null pointer.</td>
    </tr>
    <tr>
      <td>When biasG is true, gradBias is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The data types and data formats of gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, gradWeight, and gradBias are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data type of gradOut needs to be consistent with that of input.</td>
    </tr>
    <tr>
      <td>When inputG is true, the data types and data formats of sumDy or sumDyXmu are not within the supported range.</td>
    </tr>
    <tr>
      <td>When weightG is true, the data type and data format of gradWeight are not within the supported range.</td>
    </tr>
    <tr>
      <td>When biasG is true, the data type and data format of gradBias are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of gradOut and input are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of gradOut or input is greater than 8.</td>
    </tr>
    <tr>
      <td>The dimension of gradOut or input is less than 2.</td>
    </tr>
    <tr>
      <td>The size of the channel axis of input is 0.</td>
    </tr>
    <tr>
      <td>If inputG is true, the size of sumDy or sumDyXmu mismatches the channel axis of input.</td>
    </tr>
    <tr>
      <td>If weightG is true, the size of gradWeight mismatches the channel axis of input.</td>
    </tr>
    <tr>
      <td>If biasG is true, the size of gradBias mismatches the channel axis of input.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormReduceBackward

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
      <td>The workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size applied on the Device side, which is obtained by the first-phase API aclnnBatchNormReduceBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When any input is an empty tensor, the output is an empty tensor.
- Deterministic computation:
  - aclnnBatchNormReduceBackward defaults to a deterministic implementation.

## Example

The following is sample code for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_backward_reduce.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate for resource initialization
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
    // Call aclrtMalloc to apply for memory on the Device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data on the Host side to the memory on the Device side.
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
    // Handle the check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> gradOutShape = {4, 2};
    std::vector<int64_t> inputShape = {4, 2};
    std::vector<int64_t> meanShape = {2};
    std::vector<int64_t> invstdShape = {2};
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> sumDyShape = {2};
    std::vector<int64_t> sumDyXmuShape = {2};
    std::vector<int64_t> gradWeightShape = {2};
    std::vector<int64_t> gradBiasShape = {2};

    void* gradOutDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* invstdDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* sumDyDeviceAddr = nullptr;
    void* sumDyXmuDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;
    void* gradBiasDeviceAddr = nullptr;

    aclTensor* gradOut = nullptr;
    aclTensor* input = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* invstd = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* sumDy = nullptr;
    aclTensor* sumDyXmu = nullptr;
    aclTensor* gradWeight = nullptr;
    aclTensor* gradBias = nullptr;

    std::vector<float> gradOutHostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> meanHostData = {1, 1};
    std::vector<float> invstdHostData = {1, 1};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> sumDyHostData = {1, 1};
    std::vector<float> sumDyXmuHostData = {1, 1};
    std::vector<float> gradWeightHostData = {1, 1};
    std::vector<float> gradBiasHostData = {1, 1};

    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    ret = CreateAclTensor(sumDyHostData, sumDyShape, &sumDyDeviceAddr, aclDataType::ACL_FLOAT, &sumDy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sumDyXmuHostData, sumDyXmuShape, &sumDyXmuDeviceAddr, aclDataType::ACL_FLOAT, &sumDyXmu);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Example of calling the aclnnBatchNormReduceBackward API.
    // 3. Call the CANN operator library API. Replace it with the specific API name.
    // Call the first-phase API aclnnBatchNormReduceBackward
    ret = aclnnBatchNormReduceBackwardGetWorkspaceSize(
        gradOut, input, mean, invstd, weight, inputG, weightG, biasG, sumDy, sumDyXmu, gradWeight, gradBias,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormReduceBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnBatchNormReduceBackward
    ret = aclnnBatchNormReduceBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormReduceBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for the task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from Device memory to the Host. Modify it based on the specific API definition
    PrintOutResult(sumDyShape, &sumDyDeviceAddr);
    PrintOutResult(sumDyXmuShape, &sumDyXmuDeviceAddr);
    PrintOutResult(gradWeightShape, &gradWeightDeviceAddr);
    PrintOutResult(gradBiasShape, &gradBiasDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition
    aclDestroyTensor(gradOut);
    aclDestroyTensor(input);
    aclDestroyTensor(mean);
    aclDestroyTensor(invstd);
    aclDestroyTensor(weight);
    aclDestroyTensor(sumDy);
    aclDestroyTensor(sumDyXmu);
    aclDestroyTensor(gradWeight);
    aclDestroyTensor(gradBias);

    // 7. Release device resources. Modify it based on the specific API definition
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(invstdDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(sumDyDeviceAddr);
    aclrtFree(sumDyXmuDeviceAddr);
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
