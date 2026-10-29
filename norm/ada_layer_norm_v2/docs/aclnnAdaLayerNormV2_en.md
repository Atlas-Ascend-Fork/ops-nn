# aclnnAdaLayerNormV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:31:38.141Z -->

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

- Description: The AdaLayerNormV2 operator fuses LayerNorm with the downstream Add and Mul operations, and adjusts the normalization process through adaptive parameters scale and shift. Compared with the AdaLayerNorm operator, the output adds two new parameters (the mean of the input and the reciprocal of the standard deviation of the input); the data types supported by weight and bias have additional corresponding constraints.

- Formula:

  $$
  out = LayerNorm(x) * (1 + scale) + shift
  $$

  LayerNorm formula:

  $$
  mean = E(x)
  $$

  $$
  rstd = {1.0\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  LayerNorm(x) = (x-mean) * rstd * weightOptional + biasOptional
  $$

  where E(x) indicates the mean of the input, and Var(x) indicates the variance of the input.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAdaLayerNormV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnAdaLayerNormV2" API to perform the computation.

```Cpp
aclnnStatus aclnnAdaLayerNormV2GetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* scale,
  const aclTensor* shift,
  const aclTensor* weightOptional,
  const aclTensor* biasOptional,
  double           epsilon,
  aclTensor*       out,
  aclTensor*       meanOutOptional,
  aclTensor*       rstdOutOptional,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAdaLayerNormV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaLayerNormV2GetWorkspaceSize

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
      <td>Input tensor for the computation. Corresponds to `x` in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>shape is [B…, S, H], among which B supports 0 to 6 dimensions.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>Adaptive scaling parameter. Corresponds to `scale` in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type is consistent with the data type of input parameter `x`.</li><li>The shape is [B…, H] or [B…, 1, H], among which B supports 0 to 6 dimensions, the number and size of dimensions remain consistent with B in `x`, and H is consistent with the H dimension in `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shift (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the adaptive offset parameter. Corresponds to `shift` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type is consistent with the data type of input parameter `x`.</li><li>The shape is [B…, H] or [B…, 1, H], among which B supports 0 to 6 dimensions, the number and size of dimensions remain consistent with B in `x`, and H is consistent with the H dimension in `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter. It indicates the normalization scale parameter. It is the `weightOptional` in the corresponding formula.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>The data type is of type consistency with input parameter `x`, or is FLOAT32. When `biasOptional` exists, the data type of `weightOptional` is the same as that of `biasOptional`.</li><li>The shape is [H]. H is of dimension consistency with the H dimension in `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, indicates the normalization offset parameter. Corresponds to `biasOptional` in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type is the same as the data type of input parameter `x` or is FLOAT32. When `weightOptional` exists, the data type of `biasOptional` is the same as that of `weightOptional`.</li><li>The shape is [H], where H is consistent with the H dimension in `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability and prevent division-by-zero errors. Corresponds to `epsilon` in the corresponding formula.</td>
      <td>A small positive value, such as 1e-5, is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor of the computation. Corresponds to `out` in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>Data type consistency with input parameter `x`.</li><li>Shape remains consistent with `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, indicates the mean after normalization. Corresponds to the `mean` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>Data type consistency with input parameter `x`.</li><li>Shape: [B…, S, 1]. The last dimension is fixed to 1. The sizes of other dimensions remain consistent with `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, indicates the reciprocal of the standard deviation after normalization. Corresponds to `rstd` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type is consistent with the data type of input parameter `x`.</li><li>The shape is [B…, S, 1]. The last dimension is fixed to 1, and the sizes of other dimensions are consistent with `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size applied on the side of Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
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
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input x, scale, shift, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or data format of x, scale, shift, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>If weightOptional is not a null pointer, its data type or data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>If biasOptional is not a null pointer, its data type or data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of scale, shift, out, meanOutOptional, and rstdOutOptional are inconsistent with x.</td>
    </tr>
    <tr>
      <td>The data types of weightOptional and biasOptional are not FLOAT32 and are inconsistent with the data type of x.</td>
    </tr>
    <tr>
      <td>The shapes of x, scale, shift, weightOptional, biasOptional, out, meanOutOptional, and rstdOutOptional are inconsistent with those described in the Parameters section.</td>
    </tr>
  </tbody></table>

## aclnnAdaLayerNormV2

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
      <td>Workspace memory address applied on the side of the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the side of the Device, obtained by the first-phase API aclnnAdaLayerNormV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
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

- deterministic computation:
  - aclnnAdaLayerNormV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ada_layer_norm_v2.h"

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
    // Boilerplate, resource initialization
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
    // Call aclrtMalloc to apply for memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data on the host side to memory on the device side.
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
    // 1. (Boilerplate) device/stream initialization. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs. Custom construction is required based on the API.
    std::vector<int64_t> xShape = {2, 4, 8};
    std::vector<int64_t> scaleShape = {2, 8};
    std::vector<int64_t> shiftShape = {2, 8};
    std::vector<int64_t> weightShape = {8};
    std::vector<int64_t> biasShape = {8};
    std::vector<int64_t> outShape = {2, 4, 8};
    std::vector<int64_t> meanShape = {2, 4, 1};
    std::vector<int64_t> rstdShape = {2, 4, 1};
    double epsilon = 1e-5;
    void* xDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    std::vector<float> xHostData(2 * 4 * 8, 1);
    std::vector<float> scaleHostData(2 * 8, 1);
    std::vector<float> shiftHostData(2 * 8, 1);
    std::vector<float> weightHostData(8, 1);
    std::vector<float> biasHostData(8, 1);
    std::vector<float> outHostData(2 * 4 * 8, 0);
    std::vector<float> meanHostData(2 * 4 * 1, 0);
    std::vector<float> rstdHostData(2 * 4 * 1, 0);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the shift aclTensor.
    ret = CreateAclTensor(shiftHostData, shiftShape, &shiftDeviceAddr, aclDataType::ACL_FLOAT, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the bias aclTensor.
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAdaLayerNormV2.
    ret = aclnnAdaLayerNormV2GetWorkspaceSize(x, scale, shift, weight, bias, epsilon, out, mean, rstd, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaLayerNormV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAdaLayerNormV2.
    ret = aclnnAdaLayerNormV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaLayerNormV2 failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, and copy the result on the device side to the host side. Modify the code based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);

    // 7. Release the device resources. Modify the code based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
