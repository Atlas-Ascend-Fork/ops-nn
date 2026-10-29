# aclnnAddLayerNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:32:46.782Z -->

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

- Description: Implements the AddLayerNorm operation.
- Formula:

  $$
  x = x1 + x2 + biasOptional
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  y = (x-E(x)) * rstd * gamma + beta
  $$

  Where E(x) represents the mean and Var(x) represents the variance, both of which must be computed internally by the operator.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call `aclnnAddLayerNormGetWorkspaceSize` to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call `aclnnAddLayerNorm` to perform the computation.

```Cpp
aclnnStatus aclnnAddLayerNormGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *biasOptional,
  double           epsilon,
  bool             additionalOutput,
  const aclTensor *yOut,
  const aclTensor *meanOut,
  const aclTensor *rstdOut,
  const aclTensor *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormGetWorkspaceSize

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input of the addition computation in AddLayerNorm. The operator computes x1 + x2 + biasOptional and applies layer normalization to the result. Corresponds to `x1` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Does not support a dimension value of 0 in the input.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input of the addition computation in AddLayerNorm. The operator computes x1 + x2 + biasOptional and applies layer normalization to the result. Corresponds to `x2` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must be consistent with that of `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the gamma parameter in layer normalization, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The dimension values of the shape are the same as those of the dimensions to normalize in `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the beta parameter in layer normalization. It is the `beta` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>The dimension values of the shape are the same as the dimension values of `x1` to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter. Indicates the input for the addition operation in AddLayerNorm. The operator computes x1 + x2 + biasOptional and applies layer normalization to the result. Corresponds to `biasOptional` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>shape can be consistent with `gamma`/`beta` or `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Indicates the value added to the denominator to ensure numerical stability. Corresponds to `epsilon` in the corresponding formula.</td>
      <td><ul><li>Recommended value: 1e-5.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additionalOutput (bool)</td>
      <td>Input</td>
      <td>Indicates whether to enable the output of x=x1+x2+biasOptional.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output result of LayerNorm. It corresponds to `y` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>The shape must be consistent with that of input `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>meanOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the mean of the (x1 + x2 + biasOptional) result during the LayerNorm computation. It corresponds to `E(x)` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with `x1` (the dimension values of the leading dimensions are the same as those of the leading dimensions of `x1`, the trailing dimensions are 1, and the total dimensions are the same as those of `x1`. The leading dimensions refer to the dimensions of `x1` minus the dimensions of gamma, indicating the dimensions that do not require norm).</li><li>If the input `x1` is an empty tensor, `meanOut` must also be an empty tensor.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the `rstd` result output during the LayerNorm computation. It corresponds to `rstd` in the formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with `x1` (the leading dimensions are the same as the leading dimensions of `x1`, the trailing dimensions are 1, and the total number of dimensions is the same as that of `x1`; the leading dimensions refer to the dimensions of `x1` minus the dimensions of gamma, indicating the dimensions that do not require normalization).</li><li>If the input `x1` is an empty tensor, `rstdOut` must also be an empty tensor.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the result output `x` of Add, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must be consistent with input `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>:
    - The data types of the parameters `x1`, `x2`, `beta`, `gamma`, `biasOptional`, `yOut`, and `xOut` do not support BFLOAT16.
    - The parameters `meanOut` and `rstdOut` are invalid in the current product usage scenario.
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
      <td>Returns 161001 if a required input, output, or required attribute parameter is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of an input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="9">561002</td>
      <td>Failed to obtain the shapes of x1, x2, gamma, beta, yOut, meanOut, rstdOut, or xOut in the tiling phase.</td>
    </tr>
    <tr>
      <td>The number of dimensions of the shape of x1 or gamma is greater than 8 or less than 0.</td>
    </tr>
    <tr>
      <td>The numbers of dimensions of x1, x2, yOut, meanOut, rstdOut, and xOut are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of dimensions of x1 is less than that of gamma.</td>
    </tr>
    <tr>
      <td>Any dimension of (x1, gamma, meanOut) is 0.</td>
    </tr>
    <tr>
      <td>The shapes of (x1, x2, yOut, xOut) are not exactly the same.</td>
    </tr>
    <tr>
      <td>The shapes of (gamma, beta) are not exactly the same.</td>
    </tr>
    <tr>
      <td>The shapes of (meanOut, rstdOut) are not exactly the same.</td>
    </tr>
    <tr>
      <td>The dimensions of gamma are not the same as the dimensions of x that require normalization, or the dimensions of meanOut are not the same as the dimensions of x that do not require normalization, or the dimensions of meanOut that require normalization are not 1.</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNorm

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained from the first-phase API aclnnAddLayerNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor that contains the computation flow of the operator.</td>
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

- **Functional Dimensions**
  - Data type support

    <!-- npu="310p" id8 -->
    - <term>Atlas inference products</term>: x1, x2, beta, gamma, and biasOptional support FLOAT32 and FLOAT16.
    <!-- end id8 -->
    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: x1, x2, beta, gamma, and biasOptional support FLOAT32, FLOAT16, and BFLOAT16.
    <!-- end id9 -->
    - rstdOut and meanOut support: FLOAT32.
  - Data format support: ND.

  <!-- npu="310p" id10 -->
  - <term>Atlas inference products</term>: The length of the last dimension of the five inputs x1, x2, beta, gamma, and biasOptional must be greater than or equal to 32 bytes.
  <!-- end id10 -->

- **Unsupported Type Description**
  - DOUBLE: DOUBLE is not supported.
- **Boundary Value Scenario Description**
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.
- **Supported Data Types by Product**

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    | x1 Data Type | x2 Data Type | Gamma Data Type | Beta Data Type | biasOptional Data Type | yOut Data Type | meanOut Data Type | rstdOut Data Type | xOut Data Type |
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | FLOAT32  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT32  | BFLOAT16 | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | BFLOAT16 | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 |
    | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
  <!-- end id11 -->

  <!-- npu="310p" id12 -->
  - <term>Atlas inference products</term>:

    | x1 Data Type | x2 Data Type | Gamma Data Type | Beta Data Type | biasOptional Data Type | yOut Data Type | meanOut Data Type | rstdOut Data Type | xOut Data Type |
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT16 |
  <!-- end id12 -->

- Deterministic computation:
  - aclnnAddLayerNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm.h"

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
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Fill in deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check based on your needs.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.
    // Construct inputs and outputs as required by the API. This example calls the case without the optional biasOptional input and the case with biasOptional input once each.
    float eps = 1e-6;
    bool additionalOutput = true;

    std::vector<int64_t> x1Shape = {1, 2, 8};
    std::vector<int64_t> x2Shape = {1, 2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> betaShape = {8};
    std::vector<int64_t> biasOptionalShape = {8};

    std::vector<int64_t> outputYShape = {1, 2, 8};
    std::vector<int64_t> outputMeanShape = {1, 2, 1};
    std::vector<int64_t> outputRstdShape = {1, 2, 1};
    std::vector<int64_t> outputXShape = {1, 2, 8};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* biasOptionalDeviceAddr = nullptr;

    // Device address of the output without biasOptional
    void* outputYDeviceAddr = nullptr;
    void* outputMeanDeviceAddr = nullptr;
    void* outputRstdDeviceAddr = nullptr;
    void* outputXDeviceAddr = nullptr;

    // Device address of the output with biasOptional
    void* outputYDeviceAddrbiasOptional = nullptr;
    void* outputMeanDeviceAddrbiasOptional = nullptr;
    void* outputRstdDeviceAddrbiasOptional = nullptr;
    void* outputXDeviceAddrbiasOptional = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* biasOptional = nullptr;

    // aclTensor without biasOptional
    aclTensor* outputY = nullptr;
    aclTensor* outputMean = nullptr;
    aclTensor* outputRstd = nullptr;
    aclTensor* outputX = nullptr;

    // aclTensor with biasOptional
    aclTensor* outputYbiasOptional = nullptr;
    aclTensor* outputMeanbiasOptional = nullptr;
    aclTensor* outputRstdbiasOptional = nullptr;
    aclTensor* outputXbiasOptional = nullptr;

    std::vector<float> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> x2HostData = {4, 4, 4, 4, 4, 4, 4, 4, -3, -3, -3, -3, -3, -3, -3, -3};
    std::vector<float> gammaHostData = {2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> betaHostData = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    std::vector<float> biasOptionalHostData = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    // HostData without biasOptional
    std::vector<float> outputYHostData(1 * 2 * 8);
    std::vector<float> outputMeanHostData(2);
    std::vector<float> outputRstdHostData(2);
    std::vector<float> outputXHostData(1 * 2 * 8);

    // HostData with biasOptional
    std::vector<float> outputYHostDatabiasOptional(1 * 2 * 8);
    std::vector<float> outputMeanHostDatabiasOptional(2);
    std::vector<float> outputRstdHostDatabiasOptional(2);
    std::vector<float> outputXHostDatabiasOptional(1 * 2 * 8);

    // Create the self aclTensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        biasOptionalHostData, biasOptionalShape, &biasOptionalDeviceAddr, aclDataType::ACL_FLOAT, &biasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an aclTensor without biasOptional.
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputMeanHostData, outputMeanShape, &outputMeanDeviceAddr, aclDataType::ACL_FLOAT, &outputMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostData, outputRstdShape, &outputRstdDeviceAddr, aclDataType::ACL_FLOAT, &outputRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputXHostData, outputXShape, &outputXDeviceAddr, aclDataType::ACL_FLOAT, &outputX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an aclTensor with biasOptional.
    ret = CreateAclTensor(
        outputYHostDatabiasOptional, outputYShape, &outputYDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputYbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputMeanHostDatabiasOptional, outputMeanShape, &outputMeanDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputMeanbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostDatabiasOptional, outputRstdShape, &outputRstdDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputRstdbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputXHostDatabiasOptional, outputXShape, &outputXDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputXbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnAddLayerNorm API call example, including one call with biasOptional and one without biasOptional.
    // 3. Call the CANN operator library API. Change it to the specific API name.

    // 3.1 Example without the optional input biasOptional.
    // Call the first-phase API of aclnnAddLayerNorm.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnAddLayerNorm Non-biasOptional Port.");
    // Pass nullptr directly as the biasOptional parameter.
    ret = aclnnAddLayerNormGetWorkspaceSize(
        x1, x2, gamma, beta, nullptr, eps, additionalOutput, outputY, outputMean, outputRstd, outputX, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddLayerNorm.
    ret = aclnnAddLayerNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 3.2 Example with the optional input biasOptional.
    // Call the first-phase API of aclnnAddLayerNorm.
    uint64_t workspaceSizebiasOptional = 0;
    aclOpExecutor* executorbiasOptional;
    LOG_PRINT("\nUse aclnnAddLayerNorm biasOptional Port.");
    // Just pass biasOptional as normal.
    ret = aclnnAddLayerNormGetWorkspaceSize(
        x1, x2, gamma, beta, biasOptional, eps, additionalOutput, outputYbiasOptional, outputMeanbiasOptional,
        outputRstdbiasOptional, outputXbiasOptional, &workspaceSizebiasOptional, &executorbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddrbiasOptional = nullptr;
    if (workspaceSizebiasOptional > 0) {
        ret = aclrtMalloc(&workspaceAddrbiasOptional, workspaceSizebiasOptional, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddLayerNorm.
    ret = aclnnAddLayerNorm(workspaceAddrbiasOptional, workspaceSizebiasOptional, executorbiasOptional, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the results from device memory to the host. Modify the code according to the specific API definition.

    // 5.1 Copy the output without biasOptional.
    auto outputYSize = GetShapeSize(outputYShape);
    std::vector<float> resultDataY(outputYSize, 0);
    ret = aclrtMemcpy(
        resultDataY.data(), resultDataY.size() * sizeof(resultDataY[0]), outputYDeviceAddr,
        outputYSize * sizeof(resultDataY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: y output");
    for (int64_t i = 0; i < outputYSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataY[i]);
    }

    auto outputMeanSize = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMean(outputMeanSize, 0);
    ret = aclrtMemcpy(
        resultDataMean.data(), resultDataMean.size() * sizeof(resultDataMean[0]), outputMeanDeviceAddr,
        outputMeanSize * sizeof(resultDataMean[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: mean output");
    for (int64_t i = 0; i < outputMeanSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
    }

    auto outputRstdSize = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstd(outputRstdSize, 0);
    ret = aclrtMemcpy(
        resultDataRstd.data(), resultDataRstd.size() * sizeof(resultDataRstd[0]), outputRstdDeviceAddr,
        outputRstdSize * sizeof(resultDataRstd[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: rstd output");
    for (int64_t i = 0; i < outputRstdSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstd[i]);
    }

    auto outputXSize = GetShapeSize(outputXShape);
    std::vector<float> resultDataX(outputXSize, 0);
    ret = aclrtMemcpy(
        resultDataX.data(), resultDataX.size() * sizeof(resultDataX[0]), outputXDeviceAddr,
        outputXSize * sizeof(resultDataX[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: x output");
    for (int64_t i = 0; i < outputXSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataX[i]);
    }

    // 5.2 Copy the output with biasOptional.
    auto outputYSizebiasOptional = GetShapeSize(outputYShape);
    std::vector<float> resultDataYbiasOptional(outputYSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataYbiasOptional.data(), resultDataYbiasOptional.size() * sizeof(resultDataYbiasOptional[0]),
        outputYDeviceAddrbiasOptional, outputYSizebiasOptional * sizeof(resultDataYbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: y output");
    for (int64_t i = 0; i < outputYSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataYbiasOptional[i]);
    }

    auto outputMeanSizebiasOptional = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMeanbiasOptional(outputMeanSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataMeanbiasOptional.data(), resultDataMeanbiasOptional.size() * sizeof(resultDataMeanbiasOptional[0]),
        outputMeanDeviceAddrbiasOptional, outputMeanSizebiasOptional * sizeof(resultDataMeanbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: mean output");
    for (int64_t i = 0; i < outputMeanSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMeanbiasOptional[i]);
    }

    auto outputRstdSizebiasOptional = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstdbiasOptional(outputRstdSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataRstdbiasOptional.data(), resultDataRstdbiasOptional.size() * sizeof(resultDataRstdbiasOptional[0]),
        outputRstdDeviceAddrbiasOptional, outputRstdSizebiasOptional * sizeof(resultDataRstdbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: rstd output");
    for (int64_t i = 0; i < outputRstdSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstdbiasOptional[i]);
    }

    auto outputXSizebiasOptional = GetShapeSize(outputXShape);
    std::vector<float> resultDataXbiasOptional(outputXSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataXbiasOptional.data(), resultDataXbiasOptional.size() * sizeof(resultDataXbiasOptional[0]),
        outputXDeviceAddrbiasOptional, outputXSizebiasOptional * sizeof(resultDataXbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: x output");
    for (int64_t i = 0; i < outputXSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataXbiasOptional[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(beta);
    aclDestroyTensor(gamma);
    aclDestroyTensor(biasOptional);

    aclDestroyTensor(outputY);
    aclDestroyTensor(outputMean);
    aclDestroyTensor(outputRstd);
    aclDestroyTensor(outputX);

    aclDestroyTensor(outputYbiasOptional);
    aclDestroyTensor(outputMeanbiasOptional);
    aclDestroyTensor(outputRstdbiasOptional);
    aclDestroyTensor(outputXbiasOptional);

    // 7. Release device resources. Modify the code according to the specific API definition.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(biasOptionalDeviceAddr);

    aclrtFree(outputYDeviceAddr);
    aclrtFree(outputMeanDeviceAddr);
    aclrtFree(outputRstdDeviceAddr);
    aclrtFree(outputXDeviceAddr);

    aclrtFree(outputYDeviceAddrbiasOptional);
    aclrtFree(outputMeanDeviceAddrbiasOptional);
    aclrtFree(outputRstdDeviceAddrbiasOptional);
    aclrtFree(outputXDeviceAddrbiasOptional);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (workspaceSizebiasOptional > 0) {
        aclrtFree(workspaceAddrbiasOptional);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
