# aclnnLayerNormQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:18:01.606Z -->

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

- Description: The LayerNorm operator is a normalization operation commonly used in large models. The LayerNormQuant operator fuses the LayerNorm normalized output with the downstream quantization operator, reducing data move-in and move-out operations.
- Formula:
  - LayerNorm operation:

    $$
    y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
    $$

    $$
    E(x) = {\frac{1}{n} \sum_{i=1}^{n} x_i }
    $$

    $$
    Var(x) = {\frac{1}{n} \sum_{i=1}^{n} (x_i-E(x))^2 }
    $$

  - When quantMode is 0, the quantization mode is static quantization, and the output scaleOut has no practical significance:

    $$
    res = y / scale + zeroPointsOptional
    $$

  - When quantMode is 1, the quantization mode is dynamic quantization (not supported in the current version):

    $$
    tmp = y * scale
    $$

    $$
    scaleOut = row\_max(abs(tmp))/dtypeMax
    $$

    $$
    res = round(y / scaleOut )
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnLayerNormQuantGetWorkspaceSize` first to obtain the workspace size required for computation and the executor that contains the computation flow. Then call `aclnnLayerNormQuant` to perform the computation.

```Cpp
aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *scale,
  const aclTensor *zeroPointsOptional,
  int              quantMode,
  double           epsilon,
  aclTensor       *res,
  aclTensor       *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnLayerNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLayerNormQuantGetWorkspaceSize

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
      <td>Represents the x parameter in layer normalization. The corresponding formula parameter is `x`.</td>
      <td><ul><li>Supports empty tensors.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the gamma parameter in layer normalization. The corresponding formula parameter is `gamma`.</td>
      <td><ul><li>Supports empty tensors.</li><li>When quantMode is 0, the shape must be 2-dimensional, and the first dimension must be 1.</li><li>When quantMode is 1, the shape must be consistent with the dimensions of x, with all dimensions except the last one set to 1.</li><li>The last dimension must be the same as the last dimension of `x`.</li><li>The data type must be the same as that of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Corresponds to `beta` in the LayerNorm formula, indicating the beta parameter in layer normalization.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must be the same as that of `gamma`.</li><li>The data type must be the same as that of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale(aclTensor*)</td>
      <td>Input</td>
      <td>Represents the scale input of the fused quantization computation. The corresponding formula variable is `scale`.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is [1] and the dimension is 1.</li><li>The data type must be the same as that of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPointsOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input. Specifies the zeroPointsOptional input in the fused quantization computation. Valid only when quantMode is 0. Corresponds to `zeroPointsOptional` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must be the same as that of `scale`.</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode (int)</td>
      <td>Input</td>
      <td>Quantization mode. Determines whether the fused operator fuses a static or dynamic quantization operator. Corresponds to `quantMode` in the formula. The value is 0 (static quantization) or 1 (dynamic quantization).</td>
      <td>Currently, only 0 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Epsilon in the corresponding LayerNorm. A value added to the denominator to ensure numerical stability. Corresponds to `epsilon` in the formula.</td>
      <td>Default value: 1e-5.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>res (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantized result of the LayerNorm output y. Corresponds to `res` in the corresponding formula.</td>
      <td>The shape must be the same as that of input x.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Specifies the scaleOut result output of the dynamic quantization computation, corresponding to `scaleOut` in the corresponding formula. Valid only when quantMode is 1.</td>
      <td>The shape is the shape of x with the last dimension removed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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

  <!-- npu="A3,910b" id7 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.

    The data types of input parameters `x`, `gamma`, `beta`, and `scale` do not support FLOAT32.
  <!-- end id7 -->
  <!-- npu="310p" id8 -->
  - <term>Atlas inference products</term>:
    - The data types of input parameters `x`, `gamma`, `beta`, and `scale` support only FLOAT16.
    - The tail axis length of input parameters `x`, `gamma`, and `beta` must be greater than or equal to 32 bytes.
  <!-- end id8 -->

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
      <td>The input parameter is a required input, output, or required attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The hardware platform is not in the list of supported products.</td>
    </tr>
    <tr>
      <td>The input data type does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The last dimension of gamma is inconsistent with the last dimension of x.</td>
    </tr>
    <tr>
      <td>The shapes of x and res are different.</td>
    </tr>
    <tr>
      <td>The shapes of gamma and beta are different.</td>
    </tr>
    <tr>
      <td>The shapes of zeroPointsOptional and scale are different.</td>
    </tr>
    <tr>
      <td>The value of quantMode is not 0.</td>
    </tr>
  </tbody></table>

## aclnnLayerNormQuant

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
      <td>Size of the workspace allocated on the Device. The value is obtained by calling the first-phase API aclnnLayerNormQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnLayerNormQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm_quant.h"

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
    // Boilerplate for resource initialization.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output as required by the API.
    float eps = 1e-6;
    int quantMode = 0;

    std::vector<int64_t> xShape = {1, 1, 32};
    std::vector<int64_t> gammaShape = {1, 32};
    std::vector<int64_t> betaShape = {1, 32};
    std::vector<int64_t> scaleOptionalShape = {1};
    std::vector<int64_t> zeroPointOptionalShape = {1};

    std::vector<int64_t> outputYShape = {1, 1, 32};
    std::vector<int64_t> outputScaleShape = {1, 1};

    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* scaleOptionalDeviceAddr = nullptr;
    void* zeroPointOptionalDeviceAddr = nullptr;

    void* outputYDeviceAddr = nullptr;
    void* outputScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* scaleOptional = nullptr;
    aclTensor* zeroPointOptional = nullptr;

    aclTensor* outputY = nullptr;
    aclTensor* outputScale = nullptr;

    std::vector<float> xHostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> gammaHostData = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> betaHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> scaleOptionalHostData = {1};
    std::vector<int8_t> zeroPointOptionalHostData = {1};

    std::vector<int8_t> outputYHostData(1 * 1 * 32);
    std::vector<float> outputScaleHostData(1);

    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleOptionalHostData, scaleOptionalShape, &scaleOptionalDeviceAddr, aclDataType::ACL_FLOAT16, &scaleOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        zeroPointOptionalHostData, zeroPointOptionalShape, &zeroPointOptionalDeviceAddr, aclDataType::ACL_INT8, &zeroPointOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_INT8, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, &outputScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Example of the aclnnLayerNormQuant API call.
    // 3. Call the CANN operator library API. Change it to the specific API name.

    // Call the first-phase API of aclnnLayerNormQuant.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnLayerNormQuant Port.");
    ret = aclnnLayerNormQuantGetWorkspaceSize(
        x, gamma, beta, scaleOptional, zeroPointOptional, quantMode, eps, outputY, outputScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory using the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnLayerNormQuant.
    ret = aclnnLayerNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormQuant failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait for the task to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
    auto outputYSize = GetShapeSize(outputYShape);
    std::vector<int8_t> resultDataY(outputYSize, 0);
    ret = aclrtMemcpy(
        resultDataY.data(), resultDataY.size() * sizeof(resultDataY[0]), outputYDeviceAddr,
        outputYSize * sizeof(resultDataY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < outputYSize; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultDataY[i]);
    }

    if (quantMode == 1){
        auto outputScaleSize = GetShapeSize(outputScaleShape);
        std::vector<float> resultDataScale(outputScaleSize, 0);
        ret = aclrtMemcpy(
            resultDataScale.data(), resultDataScale.size() * sizeof(resultDataScale[0]), outputScaleDeviceAddr,
            outputScaleSize * sizeof(resultDataScale[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < outputScaleSize; i++) {
            LOG_PRINT("result[%ld] is: %f\n", i, resultDataScale[i]);
        }
    }

    // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(scaleOptional);
    aclDestroyTensor(zeroPointOptional);

    aclDestroyTensor(outputY);
    aclDestroyTensor(outputScale);

    // 7. Release device resources. Modify it according to the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(scaleOptionalDeviceAddr);
    aclrtFree(zeroPointOptionalDeviceAddr);

    aclrtFree(outputYDeviceAddr);
    aclrtFree(outputScaleDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
