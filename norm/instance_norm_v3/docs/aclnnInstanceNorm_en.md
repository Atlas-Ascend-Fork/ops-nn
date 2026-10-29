# aclnnInstanceNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:13:25.254Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs Instance Normalization operations. Compared with [aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm.md), aclnnInstanceNorm normalizes instances of each sample instead of the entire batch, making this function more suitable for processing data such as images.
- Formula:

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+eps}} * gamma + beta
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnInstanceNormGetWorkspaceSize` API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the `aclnnInstanceNorm` API to perform the computation.

```Cpp
aclnnStatus aclnnInstanceNormGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *beta,
  const char      *dataFormat,
  double           eps,
  aclTensor       *y,
  aclTensor       *mean,
  aclTensor       *variance,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInstanceNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnInstanceNormGetWorkspaceSize

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
      <td>Input data for the InstanceNorm computation, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The actual data format is determined by the `dataFormat` parameter.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scale factor (weight) for the InstanceNorm computation, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type is the same as that of `x`.</li><li>The shape is the same as the C dimension of input `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Bias used in InstanceNorm computation, corresponding to `beta` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type is the same as that of `x`.</li><li>The shape is the same as the C dimension of input `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dataFormat (char)</td>
      <td>Input</td>
      <td>Indicates the actual data format of the operator input Tensor.</td>
      <td>Supports NHWC or NCHW.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Value added to the variance to avoid division by zero. Corresponds to `eps` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the output of InstanceNorm, corresponding to `y` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape and data type are the same as those of input `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the mean of InstanceNorm, corresponding to `E(x)` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of input `x`.</li><li>The shape complies with the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with input `x` (the shape of the first 2 dimensions is the same as the first 2 dimensions of input x, the first 2 dimensions are the dimensions that do not require normalization, and the size of the remaining dimensions is 1).</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>variance (aclTensor*)</td>
      <td>Output</td>
      <td>Variance of InstanceNorm, corresponding to `Var(x)` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type is the same as that of `x`.</li><li>The shape complies with the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with input `x` (the shape of the first 2 dimensions is the same as the first 2 dimensions of input x, the first 2 dimensions are the dimensions that do not require normalization, and the size of the remaining dimensions is 1).</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size the user needs to allocate on the Device.</td>
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
      <td>The input x, gamma, beta, y, mean, or variance is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of the input x, gamma, beta, y, mean, or variance is not in the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of x is not 4, or the dimension of gamma/beta is not 1.</td>
    </tr>
    <tr>
      <td>The shapes of gamma and beta are inconsistent with the C axis of x.</td>
    </tr>
    <tr>
      <td>Unsupported product model.</td>
    </tr>
    <tr>
      <td>dataFormat is not set to NCHW or NHWC.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="2">561103</td>
      <td>The intermediate calculation result of the aclnn API is nullptr.</td>
    </tr>
    <tr>
      <td>The C axis or H*W length of x and y is less than 32 Bytes.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>The API fails to create aclOpExecutor internally.</td>
    </tr>
  </tbody></table>

## aclnnInstanceNorm

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
      <td>Address of the workspace memory requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnInstanceNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. It contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: Return status code. (For details, see [aclnn return codes](../../../docs/en/context/aclnn_return_code.md))

## Constraints

- Functional dimensions:
  - Supported data types:
    - x, gamma, beta, y, mean, and variance support: FLOAT32 and FLOAT16.
  - Data format support: ND.
  - x and y require 4-dimensional shapes. gamma and beta require 1-dimensional shapes and must align with the C-axis of x and y.
  - The H\*W size of x and y must be greater than or equal to 32 Bytes, and the C axis must be greater than or equal to 32 Bytes.
  - The dataFormat parameter supports only "NHWC" and "NCHW".
- Boundary value scenario description:
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.
- Deterministic computation:
  - aclnnInstanceNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_instance_norm.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t N = 1;
    int64_t C = 8;
    int64_t H = 4;
    int64_t W = 4;

    // 2. Construct the input and output. Custom construction is required based on the API. In this example, the case without the optional bias input and the case with the bias input are each called once.
    std::vector<int64_t> xShape = {N, C, H, W};
    std::vector<int64_t> weightShape = {C};
    std::vector<int64_t> yShape = {N, C, H, W};
    std::vector<int64_t> reduceShape = {N, C, 1, 1};

    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* varianceDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* y = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* variance = nullptr;

    std::vector<float> xHostData(N * C * H * W, 0.77);
    std::vector<float> gammaHostData(C, 1.5);
    std::vector<float> betaHostData(C, 0.5);
    std::vector<float> yHostData(N * C * H * W, 0.0);
    std::vector<float> meanHostData(N * C, 0.0);
    std::vector<float> varianceHostData(N * C, 0.0);
    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, weightShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, weightShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, reduceShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(varianceHostData, reduceShape, &varianceDeviceAddr, aclDataType::ACL_FLOAT, &variance);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Example of calling the aclnnInstanceNorm API
    // Call the first-phase API of aclnnInstanceNorm.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnInstanceNorm Non-Bias Port.");
    ret = aclnnInstanceNormGetWorkspaceSize(
        x, gamma, beta, dataFormat, eps, y, mean, variance, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInstanceNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    // Call the second-phase API of aclnnInstanceNorm.
    ret = aclnnInstanceNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInstanceNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the results from the device memory to the host. Modify this part based on the specific API definition.

    // 5.1 Copy the output without bias.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: y output");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    auto outputMeanSize = GetShapeSize(reduceShape);
    std::vector<float> resultDataMean(outputMeanSize, 0);
    ret = aclrtMemcpy(
        resultDataMean.data(), resultDataMean.size() * sizeof(resultDataMean[0]), meanDeviceAddr,
        outputMeanSize * sizeof(resultDataMean[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: mean output");
    for (int64_t i = 0; i < outputMeanSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
    }

    auto outputVarSize = GetShapeSize(reduceShape);
    std::vector<float> resultDataVar(outputVarSize, 0);
    ret = aclrtMemcpy(
        resultDataVar.data(), resultDataVar.size() * sizeof(resultDataVar[0]), varianceDeviceAddr,
        outputVarSize * sizeof(resultDataVar[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: rstd output");
    for (int64_t i = 0; i < outputVarSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataVar[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify this part based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(y);
    aclDestroyTensor(mean);
    aclDestroyTensor(variance);

    // 7. Release device resources. Modify this part based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);

    aclrtFree(yDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(varianceDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
