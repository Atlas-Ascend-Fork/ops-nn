# aclnnRmsNormDynamicQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:35:21.181Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean subtraction part. The DynamicQuant operator performs symmetric dynamic quantization on the input tensor. The RmsNormDynamicQuant operator fuses RmsNorm normalization and DynamicQuant dynamic quantization, reducing data move-in and move-out operations.

- Formula:

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - If smoothScalesOptional is not provided, quantization is performed directly on the rmsnorm output:

  $$
   scaleOut=row\_max(abs(y))/max\_val
  $$

  $$
   yOut=round(y/scaleOut)
  $$

  - If smoothScalesOptional is provided, smooth scaling is performed before quantization:

  $$
    input = y\cdot smoothScalesOptional
  $$

  $$
   scaleOut=row\_max(abs(input))/max\_val
  $$

  $$
   yOut=round(input/scaleOut)
  $$

  Here, row\_max represents the maximum value of each row. max\_val is 127 for INT8.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnRmsNormDynamicQuantGetWorkspaceSize` first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call `aclnnRmsNormDynamicQuant` to perform the computation.

```Cpp
aclnnStatus aclnnRmsNormDynamicQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *smoothScalesOptional,
  const aclTensor *betaOptional,
  double           epsilon,
  int64_t          dstType,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormDynamicQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormDynamicQuantGetWorkspaceSize

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
      <td>x (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the source data tensor in the normalization process, which is x in the corresponding formula.</td>
      <td>Does not support empty tensors.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the weight tensor in the normalization process, which is gamma in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensor.</li><li>The data type must remain consistent with x.</li><li>The shape must be consistent with the last dimension of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the smoothScale tensor used in the quantization process. Corresponds to smoothScalesOptional in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensor.</li><li>Optional parameter. A null pointer can be passed in.</li><li>The shape and data type must remain consistent with gamma.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the bias term in the normalization process. Corresponds to beta in the corresponding formula.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>Optional parameter. A null pointer can be passed in.</li><li>The shape and data type must remain consistent with gamma.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Indicates the epsilon in the corresponding formula, used to prevent division-by-zero errors.</td>
      <td>A small positive number is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the data type enumeration value of output yOut.</td>
      <td>DT_INT8(2) is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization output Tensor, which is yOut in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensor.</li><li>The shape must be consistent with that of input x.</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates quantization scale output, corresponding to scaleOut in the formula.</td>
      <td><ul><li>Does not support empty tensor.</li><li>The shape must be consistent with the shape of input x excluding the last dimension.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, gamma, yOut, or scaleOut is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>An exception occurs during the tiling call.</td>
    </tr>
  </tbody></table>

## aclnnRmsNormDynamicQuant

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
      <td>Memory address of the workspace allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnRmsNormDynamicQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
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

- Deterministic computation: aclnnRmsNormDynamicQuant defaults to a deterministic implementation.

- Input shape restriction: The last dimension of input `x` must be less than or equal to 8192. Otherwise, precision issues may occur.

- Input value range restriction: `x` does not support all-zero input.

## Example

The sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_dynamic_quant.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate code, resource initialization
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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate code) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output as required by the API.
    std::vector<int64_t> xShape = {2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> scaleShape = {2};

    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* smoothDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* smooth = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scale = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t gammaShapeSize = GetShapeSize(gammaShape);
    int64_t scaleShapeSize = GetShapeSize(scaleShape);

    // FP16 bit patterns: 0x3800=0.5, 0x3e00=1.5
    std::vector<short> xHostData(xShapeSize, 0x3800);
    std::vector<short> gammaHostData(gammaShapeSize, 0x3e00);
    std::vector<short> smoothHostData(gammaShapeSize, 0x3e00);
    std::vector<short> betaHostData(gammaShapeSize, 0);

    std::vector<short> yHostData(xShapeSize, 0);
    std::vector<float> scaleHostData(scaleShapeSize, 0);

    float epsilon = 1e-6;
    int64_t dstType = 2; // DT_INT8

    // Input
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(smoothHostData, gammaShape, &smoothDeviceAddr, aclDataType::ACL_FLOAT16, &smooth);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, gammaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Output
    ret = CreateAclTensor(yHostData, xShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, and change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRmsNormDynamicQuant.
    ret = aclnnRmsNormDynamicQuantGetWorkspaceSize(
        x, gamma, smooth, beta, epsilon, dstType, y, scale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Request device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnRmsNormDynamicQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormDynamicQuant failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result on device memory to the host side. Modify it based on the interface definition of the specific API.
    // Print y (INT8).
    std::vector<int8_t> yRet(xShapeSize, 0);
    ret = aclrtMemcpy(yRet.data(), xShapeSize, yDeviceAddr, xShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("=== y (INT8, {2,8}) ===\n");
    for (int64_t i = 0; i < xShapeSize; i++) {
        LOG_PRINT("y[%ld]: %d\n", i, yRet[i]);
    }

    // Print scale (FLOAT32).
    std::vector<float> sRet(scaleShapeSize, 0);
    ret = aclrtMemcpy(sRet.data(), scaleShapeSize * sizeof(float), scaleDeviceAddr, scaleShapeSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("=== scale (FLOAT32, {2}) ===\n");
    for (int64_t i = 0; i < scaleShapeSize; i++) {
        LOG_PRINT("scale[%ld]: %f\n", i, sRet[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the interface definition of the specific API.
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(smooth);
    aclDestroyTensor(beta);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);

    // 7. Release device resources, which must be modified according to the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(smoothDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
