# aclnnDeepNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:01:34.919Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs deep normalization on the elements of the input tensor x by computing its mean and standard deviation, normalizing each element to an output tensor with zero mean and unit variance.
- Formula:

  $$
  DeepNorm(x_i^{\prime}) = ({x_i^{\prime} - \bar{x^{\prime}}})* rstd * gamma + beta,
  $$

  $$
  \text { where } rstd = \frac{1} {\sqrt{\frac{1}{n} \sum_{i=1}^n (x^{\prime}_i - \bar{x^{\prime}})^2 + epsilon} }, \quad x^{\prime}_i = alpha * x_i   + gx_i
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnDeepNormGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the computation flow of the operator, and then call "aclnnDeepNorm" to perform the computation.

```Cpp
aclnnStatus aclnnDeepNormGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gx,
  const aclTensor *beta,
  const aclTensor *gamma,
  double           alpha,
  double           epsilon,
  const aclTensor *meanOut,
  const aclTensor *rstdOut,
  const aclTensor *yOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDeepNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDeepNormGetWorkspaceSize

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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input data, typically the output of an intermediate layer of a neural network, corresponding to `x` in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gx (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient of the input data, used for backward propagation, corresponding to `gx` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as the data type of `x`.</li><li>The shape dimensions are the same as those of input `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Denotes the bias parameter, used to adjust the normalized output, corresponding to `beta` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as the data type of `x`.</li><li>The shape dimensions are the same as those of the last few dimensions of input `x`, which indicate the dimensions to normalize.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scale parameter used to adjust the normalized output, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type is the same as the data type of `x`.</li><li>The shape dimensions are the same as the last few dimensions of input `x`, which indicate the dimensions to normalize.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha (double)</td>
      <td>Input</td>
      <td>Specifies the weight parameter, used to adjust the weight of input data. It corresponds to `alpha` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Value added to the variance to avoid division by zero. Corresponds to `epsilon` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meanOut (aclTensor*)</td>
      <td>Output</td>
      <td>Mean of the computed output, used for normalization. Corresponds to `mean` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape satisfies the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with input `x` (the leading dimensions have the same sizes as those of input `x`, where the leading dimensions are the ones that do not need normalization, and the remaining dimensions are of size 1).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Reciprocal of the standard deviation in the computation output, used for normalization, corresponding to `rstd` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape satisfies the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with input `x` (the leading dimensions have the same sizes as those of input `x`, where the leading dimensions are the ones that do not need normalization, and the remaining dimensions are of size 1).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the normalized output data, corresponding to `y` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `x`.</li><li>The shape dimensions are the same as those of input `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
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
      <td>If the input parameter is a required input, output, or required attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of the input and output do not match or are not within the supported dimension range.</td>
    </tr>
    <tr>
      <td>The data types of the input and output do not meet the constraints in Parameters.</td>
    </tr>
  </tbody></table>

## aclnnDeepNorm

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnDeepNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Functional dimensions:
  - Supported data types:

    <!-- npu="310p" id7 -->
    - <term>Atlas inference products</term>: FLOAT32 and FLOAT16 are supported for x, gx, beta, gamma, and yOut.
    <!-- end id7 -->
    <!-- npu="950,A3,910b" id8 -->
    - <term>Ascend 950PR&950DT products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: x, gx, beta, gamma, and yOut support FLOAT32, FLOAT16, and BFLOAT16.
    <!-- end id8 -->
    - rstdOut and meanOut support: FLOAT32.
  - Data format support: ND.

- Unsupported type description:

  DOUBLE: The instruction does not support DOUBLE.

- Boundary value scenario description:
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.

- Deterministic computation:
  - aclnnDeepNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_deep_norm.h"

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
    // Boilerplate: resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Process the check as needed
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct input and output. Custom construction is required based on the API definition.
    float alpha = 0.3;
    float eps = 1e-6;
    std::vector<int64_t> xShape = {3, 1, 4};
    std::vector<int64_t> gxShape = {3, 1, 4};
    std::vector<int64_t> betaShape = {4};
    std::vector<int64_t> gammaShape = {4};
    std::vector<int64_t> outputMeanShape = {3, 1, 1};
    std::vector<int64_t> outputRstdShape = {3, 1, 1};
    std::vector<int64_t> outputYShape = {3, 1, 4};

    void* xDeviceAddr = nullptr;
    void* gxDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* outputMeanDeviceAddr = nullptr;
    void* outputRstdDeviceAddr = nullptr;
    void* outputYDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gx = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* outputMean = nullptr;
    aclTensor* outputRstd = nullptr;
    aclTensor* outputY = nullptr;

    std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> gxHostData = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8};
    std::vector<float> betaHostData = {0, 1, 2, 3};
    std::vector<float> gammaHostData = {0, 1, 2, 3};
    std::vector<float> outputMeanHostData = {0, 1, 2};
    std::vector<float> outputRstdHostData = {0, 1, 2};
    std::vector<float> outputYHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gxHostData, gxShape, &gxDeviceAddr, aclDataType::ACL_FLOAT, &gx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(
        outputMeanHostData, outputMeanShape, &outputMeanDeviceAddr, aclDataType::ACL_FLOAT, &outputMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostData, outputRstdShape, &outputRstdDeviceAddr, aclDataType::ACL_FLOAT, &outputRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnDeepNorm API call example.
    // 3. Call the CANN operator library API. Replace the API name with the actual API name.
    // Call the first-phase API of aclnnDeepNorm.
    LOG_PRINT("\nUse aclnnDeepNorm Port.");
    ret = aclnnDeepNormGetWorkspaceSize(
        x, gx, beta, gamma, alpha, eps, outputMean, outputRstd, outputY, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory according to the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnDeepNorm.
    ret = aclnnDeepNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait synchronously until the task execution ends.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the result from device memory to host memory. Modify based on the specific API definition.
    auto outputMeanSize = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMean(outputMeanSize, 0);
    ret = aclrtMemcpy(
        resultDataMean.data(), resultDataMean.size() * sizeof(resultDataMean[0]), outputMeanDeviceAddr,
        outputMeanSize * sizeof(resultDataMean[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdx output");
    for (int64_t i = 0; i < outputMeanSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
    }

    auto outputRstdSize = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstd(outputRstdSize, 0);
    ret = aclrtMemcpy(
        resultDataRstd.data(), resultDataRstd.size() * sizeof(resultDataRstd[0]), outputRstdDeviceAddr,
        outputRstdSize * sizeof(resultDataRstd[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdx output");
    for (int64_t i = 0; i < outputRstdSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstd[i]);
    }

    auto outputYSize = GetShapeSize(outputYShape);
    std::vector<float> resultDataY(outputYSize, 0);
    ret = aclrtMemcpy(
        resultDataY.data(), resultDataY.size() * sizeof(resultDataY[0]), outputYDeviceAddr,
        outputYSize * sizeof(resultDataY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdx output");
    for (int64_t i = 0; i < outputYSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataY[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(gx);
    aclDestroyTensor(beta);
    aclDestroyTensor(gamma);
    aclDestroyTensor(outputMean);
    aclDestroyTensor(outputRstd);
    aclDestroyTensor(outputY);

    // 7. Release the device resources. Modify based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(gxDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(outputMeanDeviceAddr);
    aclrtFree(outputRstdDeviceAddr);
    aclrtFree(outputYDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
