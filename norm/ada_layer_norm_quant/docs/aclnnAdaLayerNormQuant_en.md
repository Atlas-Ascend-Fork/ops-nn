# aclnnAdaLayerNormQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:28:43.485Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: The AdaLayerNormQuant operator fuses AdaLayerNorm with downstream quantization (currently only DynamicQuant is supported). This operator is mainly used to perform quantization with adaptive layer normalization, that is, it normalizes the input data and quantizes it into low-precision integers to improve computing efficiency and reduce memory usage.

- Formula:

  1. First, perform LayerNorm normalization on the input x:

     $$
     LayerNorm(x) = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * weightOptional + biasOptional
     $$

  2. Then adjust the normalization result using the adaptive parameters scale and shift:

     $$
     y = LayerNorm(x) * (1 + scale) + shift
     $$

  3. If smoothScalesOptional is not null:

     $$
     y = y \cdot smoothScalesOptional
     $$

  4. Then compute the maximum absolute value of y and divide it by $(FP8\_MAX / HIF8\_MAX / INT8\_MAX)$ to calculate the quantization factor for quantizing to the FLOAT8/HIF8/INT8 format:

     $$
     quantScale = row\_max(abs(y)) / (FP8\_MAX / HIF8\_MAX / INT8\_MAX)
     $$

  5. Finally, divide y by the quantization factor and round the result to obtain the quantized output:

     $$
     out = round(y / quantScale)
     $$

  where E(x) indicates input mean, Var(x) indicates input variance, and row_max indicates the maximum value of each row.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAdaLayerNormQuantGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnAdaLayerNormQuant" API to perform the computation.

```Cpp
aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* scale,
  const aclTensor* shift,
  const aclTensor* weightOptional,
  const aclTensor* biasOptional,
  const aclTensor* smoothScalesOptional,
  double           epsilon,
  const char*      quantMode,
  aclTensor*       out,
  aclTensor*       quantScale,
  aclTensor*       quantOffsetOptional,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAdaLayerNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaLayerNormQuantGetWorkspaceSize

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
      <td>Indicates input data to be processed. Corresponds to `x` in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The shape is [B…, S, H], where B supports 0 to 6 dimensions.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the adaptive scaling parameter. Corresponds to `scale` in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The data type is the same as the data type of input parameter `x`.</li><li>shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions. The number of dimensions and sizes remain consistent with B in `x`, and H is consistent with the H dimension in `x`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shift (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the adaptive offset parameter, corresponding to `shift` in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The data type is the same as the data type of input parameter `x`.</li><li>shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions. The number of dimensions and sizes remain consistent with B in `x`, and H is consistent with the H dimension in `x`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, indicating the normalization scale parameter. Corresponds to `weightOptional` in the corresponding formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The data type must be the same as that of input `x`.</li><li>The shape is [H], where H is the same as the H dimension in `x`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input, indicating the normalization offset parameter. Corresponds to `biasOptional` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>The data type is the same as the data type of input `x`.</li><li>The shape is [H], where H is the same as the H dimension of `x`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, indicates the quantization smoothing weight. Corresponds to `smoothScalesOptional` in the formula.</td>
      <td><ul><li>Does not support null tensor.</li><li>The data type is consistent with the data type of input `x`.</li><li>The shape is [H], where H remains consistent with the H dimension of `x`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability and prevent division-by-zero errors. Corresponds to `epsilon` in the formula.</td>
      <td>It is recommended to pass a small positive value, such as 1e-5.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (char*)</td>
      <td>Input</td>
      <td>Indicates the quantization mode.</td>
      <td>The current version supports only "dynamic".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization output tensor. Corresponds to `out` in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The shape must remain consistent with the shape of input parameter `x`.</li></ul></td>
      <td>INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScale (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates quantization factors. The corresponding formula variable is `quantScale`.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The shape is [B…, S], where B supports 0 to 6 dimensions. The number of dimensions and size of B remain consistent with those of `x`, and S is consistent with the S dimension in `x`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantOffsetOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, indicating the offset used for asymmetric quantization.</td>
      <td><ul><li>Does not support null tensors.</li><li>The shape must remain consistent with the shape of `quantScale`.</li><li>Not supported in the current version. Pass nullptr.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device side.</td>
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

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The data type of output `out` supports only INT8.
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
      <td>The passed-in x, scale, shift, out, or quantScale is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type or data format of x, scale, shift, out, or quantScale is not within the supported range.</td>
    </tr>
    <tr>
      <td>When weightOptional is a non-null pointer, its data type or data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>When biasOptional is a non-null pointer, its data type or data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>When smoothScalesOptional is a non-null pointer, its data type or data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>quantMode is not "dynamic".</td>
    </tr>
    <tr>
      <td>quantOffsetOptional is a non-null pointer.</td>
    </tr>
    <tr>
      <td>The data type of scale, shift, weightOptional, biasOptional, or smoothScalesOptional is inconsistent with that of x.</td>
    </tr>
    <tr>
      <td>The shapes of x, scale, shift, weightOptional, biasOptional, smoothScalesOptional, out, and quantScale are inconsistent with those described in Parameters.</td>
    </tr>
  </tbody></table>

## aclnnAdaLayerNormQuant

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnAdaLayerNormQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnAdaLayerNormQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ada_layer_norm_quant.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host side to the memory on the device side.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs, which need to be customized based on the API.
    std::vector<int64_t> xShape = {2, 4, 8};
    std::vector<int64_t> scaleShape = {2, 8};
    std::vector<int64_t> shiftShape = {2, 8};
    std::vector<int64_t> weightShape = {8};
    std::vector<int64_t> biasShape = {8};
    std::vector<int64_t> smoothScalesShape = {8};
    std::vector<int64_t> outShape = {2, 4, 8};
    std::vector<int64_t> quantScaleShape = {2, 4};
    double epsilon = 1e-5;
    const char* quantMode = "dynamic";
    void* xDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* smoothScalesDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* quantScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* smoothScales = nullptr;
    aclTensor* out = nullptr;
    aclTensor* quantScale = nullptr;
    std::vector<short> xHostData(2 * 4 * 8, 1);
    std::vector<short> scaleHostData(2 * 8, 1);
    std::vector<short> shiftHostData(2 * 8, 1);
    std::vector<short> weightHostData(8, 1);
    std::vector<short> biasHostData(8, 1);
    std::vector<short> smoothScalesHostData(8, 1);
    std::vector<int8_t> outHostData(2 * 4 * 8, 0);
    std::vector<float> quantScaleHostData(2 * 4, 0);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create shift aclTensor.
    ret = CreateAclTensor(shiftHostData, shiftShape, &shiftDeviceAddr, aclDataType::ACL_FLOAT16, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create bias aclTensor.
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create smoothScales aclTensor.
    ret = CreateAclTensor(smoothScalesHostData, smoothScalesShape, &smoothScalesDeviceAddr, aclDataType::ACL_FLOAT16, &smoothScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create quantScale aclTensor.
    ret = CreateAclTensor(quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be modified to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAdaLayerNormQuant.
    ret = aclnnAdaLayerNormQuantGetWorkspaceSize(x, scale, shift, weight, bias, smoothScales, epsilon, quantMode, out, quantScale, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaLayerNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAdaLayerNormQuant.
    ret = aclnnAdaLayerNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaLayerNormQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values, copy the result in device-side memory to the host side, and modify it based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar, and modify them based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(smoothScales);
    aclDestroyTensor(out);
    aclDestroyTensor(quantScale);

    // 7. Release device resources, and modify them based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(smoothScalesDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(quantScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
