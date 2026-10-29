# aclnnRmsNormQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:35:23.277Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: The RMSNorm operator is a common normalization operation in large models. Compared with the LayerNorm operator, it removes the part of subtracting the mean. The RmsNormQuant operator fuses the RMSNorm operator and the Quantize operator after RMSNorm, reducing data movement operations.
- Formula:

$$
quant\_in_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i + beta_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
$$

$$
y=round((quant\_in/scale)+offset)
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnRmsNormQuantGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnRmsNormQuant" API to perform computation.

```Cpp
aclnnStatus aclnnRmsNormQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *scale,
  const aclTensor *offset,
  double           epsilon,
  aclTensor       *y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormQuantGetWorkspaceSize

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
      <td>Source data tensor in the normalization process. Corresponds to `x` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Does not support null tensor.</li><li>Data type must remain consistent with `x`.</li><li>If shape is 1-dimensional, it must remain consistent with the last dimension of `x`.</li><li>If shape is 2-dimensional, the first dimension must be 1, and the second dimension must remain consistent with the last dimension of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Offset tensor in the normalization process. Corresponds to `beta` in the formula.</td>
      <td><ul><li>Does not support null tensor.</li><li>Data type must remain consistent with `x`.</li><li>If shape is 1-dimensional, it must remain consistent with the last dimension of `x`.</li><li>If shape is 2-dimensional, the first dimension must be 1, and the second dimension must remain consistent with the last dimension of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>Scale tensor used for computing y during quantization, corresponding to `scale` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>The shape is 1, and the number of dimensions is 1.</li><li>The value of this parameter cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset(aclTensor*)</td>
      <td>Input</td>
      <td>Represents the offset tensor used to compute y during quantization, corresponding to `offset` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>shape must remain consistent with `scale`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon(double)</td>
      <td>Input</td>
      <td>Input `epsilon` in the formula, used to prevent division-by-zero errors, with data type DOUBLE.</td>
      <td>A small positive value is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the final quantization output tensor, corresponding to `y` in the formula.</td>
      <td><ul><li>Does not support null tensor.</li><li>The shape must remain consistent with input `x`.</li></ul></td>
      <td>INT32, INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
      <td>1-8</td>
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
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="310p,310b" id7 -->
  - <term>Atlas inference products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type of input parameters <code>x</code>, <code>gamma</code>, <code>beta</code>, and <code>scale</code> only supports FLOAT16.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: the `x` data type, `gamma` data type, `beta` data type, and `scale` data type of the input parameters support only FLOAT16 and BFLOAT16, and the `offset` data type supports only INT8.
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
      <td>If the input parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The input or output data type is out of the supported range, or the input and output data types do not satisfy the corresponding relationship described in Constraints.</td>
    </tr>
    <tr>
      <td>The input/output shape relationship does not meet expectations.</td>
    </tr>
  </tbody></table>

## aclnnRmsNormQuant

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
      <td>Pointer to the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device, obtained by calling the first-phase API aclnnRmsNormQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation process of the operator.</td>
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

- If the value of input parameter scale or offset is NaN, or the RmsNorm computation result (corresponding to $quant\_in_i$ in the formula) is NaN, the final output of this API is 0.

<!-- npu="310p" id9 -->
- <term>Atlas inference products</term>: The tail axis lengths of x and y, and the tail axis length of gamma, must be greater than or equal to 32 bytes.
<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>: When the data type of <code>y</code> is INT4, the last dimension of <code>x</code>, <code>gamma</code>, and <code>beta</code> must be an even number.
<!-- end id10 -->
<!-- npu="950" id11 -->
- <term>Ascend 950PR&950DT products</term>: When the data type of <code>y</code> is INT32, the last dimension of <code>y</code> must be 1/8 of the last dimension of <code>x</code>.
<!-- end id11 -->
- Supported data types for each product model:

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    | X Data Type | Gamma Data Type | Beta Data Type | Scale Data Type | Offset Data Type | Epsilon Data Type | y Data Type |
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | FLOAT16   | FLOAT16       | FLOAT16       | FLOAT16       | INT8           | DOUBLE      |INT8      |
    | BFLOAT16  | BFLOAT16      | BFLOAT16      | BFLOAT16      | INT8           | DOUBLE      |INT8      |
    | FLOAT16   | FLOAT16       | FLOAT16       | FLOAT16       | INT8           | DOUBLE      |INT4      |
    | BFLOAT16  | BFLOAT16      | BFLOAT16      | BFLOAT16      | INT8           | DOUBLE      |INT4      |
  <!-- end id12 -->

  <!-- npu="310p,310b" id13 -->
  - <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>:

    | X Data Type | Gamma Data Type | Beta Data Type | Scale Data Type | Offset Data Type | Epsilon Data Type | y Data Type |
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | FLOAT16   | FLOAT16       | FLOAT16       | FLOAT16       | INT8           | DOUBLE      |INT8      |
    | FLOAT16   | FLOAT16       | FLOAT16       | FLOAT16       | INT8           | DOUBLE      |INT4      |
  <!-- end id13 -->

  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT products</term>:

    | X Data Type | Gamma Data Type | Beta Data Type | Scale Data Type | Offset Data Type | Epsilon Data Type | y Data Type |
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | FLOAT16   | FLOAT16       | FLOAT16       | FLOAT16       | INT8           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      |
    | BFLOAT16   | BFLOAT16       | BFLOAT16       | BFLOAT16       | INT8           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      |
    | FLOAT16   | FLOAT16       | FLOAT16       | FLOAT16       | FLOAT16           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      |
    | BFLOAT16   | BFLOAT16       | BFLOAT16       | BFLOAT16       | BFLOAT16           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      |
    | FLOAT32   | FLOAT32       | FLOAT32       | FLOAT32       | FLOAT32           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      |
    | FLOAT16   | FLOAT16       | FLOAT16       | FLOAT32       | INT32           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      |
    | BFLOAT16   | BFLOAT16       | BFLOAT16       | FLOAT32      | INT32           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      |
    | FLOAT16   | FLOAT16       | FLOAT16       | FLOAT32       | FLOAT32           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      |
    | BFLOAT16   | BFLOAT16       | BFLOAT16       | FLOAT32       | FLOAT32           | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8     |
  <!-- end id14 -->

- Deterministic computation:
  - aclnnRmsNormQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_quant.h"

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
    // Boilerplate code, resource initialization.
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
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
    // 1. (Boilerplate) Initialize device/stream. Refer to the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Process the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> xShape = {3, 5};
    std::vector<int64_t> gammaShape = {1, 5};
    std::vector<int64_t> betaShape = {1, 5};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> offsetShape = {1};
    std::vector<int64_t> yShape = {3, 5};
    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* y = nullptr;

    std::vector<aclFloat16> xHostData;
    std::vector<aclFloat16> gammaHostData;
    std::vector<aclFloat16> betaHostData;
    std::vector<aclFloat16> scaleHostData;
    float values = 0;
    for (int i = 0; i < 15; ++i) {
        values = i;
        xHostData.push_back(aclFloatToFloat16(values));
    }

    for (int i = 0; i < 5; ++i) {
        values = i;
        gammaHostData.push_back(aclFloatToFloat16(values));
        betaHostData.push_back(aclFloatToFloat16(values));
    }
    values = 1;
    scaleHostData.push_back(aclFloatToFloat16(values));

    std::vector<int8_t> offsetHostData(1, 1);
    std::vector<int8_t> yHostData(15, 0);
    double epsilon = 1e-6;
    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_INT8, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRmsNormQuant.
    ret = aclnnRmsNormQuantGetWorkspaceSize(x, gamma, beta, scale, offset, epsilon, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnRmsNormQuant.
    ret = aclnnRmsNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (boilerplate) Synchronously wait for the task execution to finish.

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the results from device memory to host memory. Modify based on the specific API definition.
    auto size = GetShapeSize(yShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release the aclTensor. Modify based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(y);

    // 7. Release device resources. Modify based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(yDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
