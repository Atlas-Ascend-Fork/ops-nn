# aclnnBatchNormElemtBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:39:11.371Z -->

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

- Description: Backward computation of [aclnnBatchNormElemt](../../batch_norm_elemt/docs/aclnnBatchNormElemt.md). Computes element-wise gradients of the input tensor for updating model parameters during backward propagation.
- Formula:

  $$
  gradInput = (({gradOut} - \frac{sumDy}{ {counter}}) - (input - mean) * (invstd^{2} *   (\frac{sumDyXmu}{ {counter}}))) * invstd * weight
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnBatchNormElemtBackwardGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size according to the process, and then call the "aclnnBatchNormElemtBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnBatchNormElemtBackwardGetWorkspaceSize(
  const aclTensor* gradOut,
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  const aclTensor* weight,
  const aclTensor* sumDy,
  const aclTensor* sumDyXmu,
  aclTensor*       counter,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormElemtBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormElemtBackwardGetWorkspaceSize

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
      <td>Indicates the differential of the forward output, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>The shape must be the same as the shape of `input`.</li><li>The second dimension is fixed as the channel axis.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input for the BatchNorm computation, corresponding to `input` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The 2nd dimension is fixed to the channel axis, and the size of the channel axis cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input data mean, corresponding to `mean` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The shape length is equal to the length of the channel axis in input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the reciprocal of the standard deviation of input data, that is, `invstd` in the corresponding formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>The shape length is equal to the length of the channel axis in input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor, corresponding to `weight` in the corresponding formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>Optional parameter.</li><li>The shape length is equal to the length of the channel axis in input `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDy (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the average of the sum of output gradients across samples, corresponding to `sumDy` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The shape length is equal to the length of the channel axis in input `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDyXmu (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the average of the product of the sample mean sum and the input gradient, corresponding to `sumDyXmu` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The shape length is equal to the length of the channel axis in input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>counter (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the quantity of input data, corresponding to `counter` in the formula.</td>
      <td><ul><li>Supports null tensor.</li></ul></td>
      <td>INT32, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>indicates the gradient of input Tensor, corresponding to `gradInput` in the corresponding formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>shape is consistent with the shape of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace applied on the side of Device by the user.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: The data type of the parameters `gradOut`, `input`, `mean`, `invstd`, `weight`, `sumDy`, `sumDyXmu`, and `gradInput` does not support BFLOAT16.
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>gradOut, input, mean, invstd, sumDy, sumDyXmu, counter, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data types of gradOut, input, mean, invstd, sumDy, sumDyXmu, counter, and gradInput are not within the supported range.</td>
    </tr>
    <tr>
      <td>If weight is not a null pointer, the data type of weight is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of gradOut, input, and gradInput are not within the supported range.</td>
    </tr>
    <tr>
      <td>The number of dimensions of input is less than 2.</td>
    </tr>
    <tr>
      <td>The dimension of input, gradOut, gradInput, or counter is greater than 8.</td>
    </tr>
    <tr>
      <td>The size of the channel axis of input is 0.</td>
    </tr>
    <tr>
      <td>The shape of gradOut or gradInput is inconsistent with that of input.</td>
    </tr>
    <tr>
      <td>The shape of mean, invstd, sumDy, or sumDyXmu is inconsistent with the channel axis of input.</td>
    </tr>
    <tr>
      <td>If weight is a non-null pointer, the shape of weight is inconsistent with the channel axis of input.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormElemtBackward

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
      <td>Workspace memory address applied on the side of Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the side of Device, obtained by the first-phase API aclnnBatchNormElemtBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBatchNormElemtBackward defaults to a deterministic implementation.

## Example

The sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_elemt_backward.h"

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
    // Fixed boilerplate, resource initialization
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

    // Call aclrtMemcpy to copy data from the Host side to the Device side.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor.
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
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> gradOutShape = {1, 2, 4};
    std::vector<int64_t> inputShape = {1, 2, 4};
    std::vector<int64_t> meanShape = {2};
    std::vector<int64_t> invstdShape = {2};
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> sumDyShape = {2};
    std::vector<int64_t> sumDyXmuShape = {2};
    std::vector<int64_t> counterShape = {2};
    std::vector<int64_t> gradInputShape = {1, 2, 4};
    void* gradOutDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* invstdDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* sumDyDeviceAddr = nullptr;
    void* sumDyXmuDeviceAddr = nullptr;
    void* counterDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* input = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* invstd = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* sumDy = nullptr;
    aclTensor* sumDyXmu = nullptr;
    aclTensor* counter = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> meanHostData = {0, 0};
    std::vector<float> invstdHostData = {1, 1};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> sumDyHostData = {0, 0};
    std::vector<float> sumDyXmuHostData = {1, 1};
    std::vector<float> counterHostData = {5, 5};
    std::vector<float> gradInputHostData(8, 0);

    // Create the gradOut aclTensor.
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the invstd aclTensor.
    ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the sumDy aclTensor.
    ret = CreateAclTensor(sumDyHostData, sumDyShape, &sumDyDeviceAddr, aclDataType::ACL_FLOAT, &sumDy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the sumDyXmu aclTensor.
    ret = CreateAclTensor(sumDyXmuHostData, sumDyXmuShape, &sumDyXmuDeviceAddr, aclDataType::ACL_FLOAT, &sumDyXmu);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the counter aclTensor.
    ret = CreateAclTensor(counterHostData, counterShape, &counterDeviceAddr, aclDataType::ACL_FLOAT, &counter);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradInput aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNormElemtBackward API call example
    // 3. Call the CANN operator library API. Replace it with the actual API name.
    // Call the first-phase API aclnnBatchNormElemtBackward.
    ret = aclnnBatchNormElemtBackwardGetWorkspaceSize(
        gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, counter, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormElemtBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnBatchNormElemtBackward.
    ret = aclnnBatchNormElemtBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormElemtBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from Device-side memory to the Host side. Modify it based on the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
    aclDestroyTensor(gradOut);
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(mean);
    aclDestroyTensor(invstd);
    aclDestroyTensor(sumDy);
    aclDestroyTensor(sumDyXmu);
    aclDestroyTensor(counter);
    aclDestroyTensor(gradInput);

    // 7. Release device resources. Modify it based on the specific API definition.
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(invstdDeviceAddr);
    aclrtFree(sumDyDeviceAddr);
    aclrtFree(sumDyXmuDeviceAddr);
    aclrtFree(counterDeviceAddr);
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
