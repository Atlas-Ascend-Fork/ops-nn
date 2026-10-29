# aclnnAdaLayerNormBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:27:56.170Z -->

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

- Description: Backward propagation of [aclnnAdaLayerNormV2](../../ada_layer_norm_v2/docs/aclnnAdaLayerNormV2.md). Computes the gradients of the input tensors to update model parameters during backward propagation.

- Formula:

  Forward:

  $$
  out = LayerNorm(input)\times(1+scale)+shift
  $$

  LayerNorm formula:

  $$
  LayerNorm(input) = {{input-E(input)}\over\sqrt {Var(input)+epsilon}} * weightOptional + biasOptional
  $$

  In the preceding formula, E(input) indicates the mean of the input, and Var(input) indicates the variance of the input.

  Backward:

  $$
  z = (input - mean) \times rstd
  $$

  $$
  dy\_g = gradOut \times weightOptional \times (1 + scale)
  $$

  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_2} (gradOut \times weightOptional \times (1 + scale))
  $$

  $$
  temp_2 = 1/N \times (input - mean) \times rstd \times \sum_{reduce\_axis\_2}(gradOut \times weightOptional \times (1 + scale) \times (input - mean) \times rstd)
  $$

  $$
  gradInputOut = (gradOut \times weightOptional \times (1 + scale) - (temp_1 + temp_2)) \times rstd
  $$

  $$
  gradScaleOut =  \sum_{reduce\_axis\_1}(gradOut \times ((input - mean) \times rstd \times weightOptional + biasOptional))
  $$

  $$
  gradShiftOut =  \sum_{reduce\_axis\_1}gradOut
  $$

  $$
  gradWeightOut =  \sum_{reduce\_axis\_0,1}gradOut \times (1 + scale) \times (input - mean) \times rstd
  $$

  $$
  gradBiasOut = \sum_{reduce\_axis\_0,1}gradOut \times (1 + scale)
  $$

  Where N is the dimension of the axis for normalization computation, that is, the size of the normalized axis dimension.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnAdaLayerNormBackwardGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call "aclnnAdaLayerNormBackward" to perform computation.

```Cpp
aclnnStatus aclnnAdaLayerNormBackwardGetWorkspaceSize(
    const aclTensor*    gradOut,
    const aclTensor*    input,
    const aclIntArray*  normalizedShape,
    const aclTensor*    rstd,
    const aclTensor*    mean,
    const aclTensor*    scale,
    const aclTensor*    shift,
    const aclTensor*    weightOptional,
    const aclTensor*    biasOptional,
    aclTensor*          gradInputOut,
    aclTensor*          gradScaleOut,
    aclTensor*          gradShiftOut,
    aclTensor*          gradWeightOut,
    aclTensor*          gradBiasOut,
    uint64_t*           workspaceSize,
    aclOpExecutor**     executor)
```

```Cpp
aclnnStatus aclnnAdaLayerNormBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAdaLayerNormBackwardGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 430px">
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
      <td>Gradient tensor for backward computation, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Same data type as input `input`.</li><li>shape is [B…, S, H], where B supports 0 to 6 dimensions.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>First input of the forward computation, corresponding to `input` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Same data type as the input `gradOut`.</li><li>Shape is the same as the shape of `gradOut`: [B…, S, H], where B supports 0 to 6 dimensions.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape (aclIntArray*)</td>
      <td>Input</td>
      <td>Dimensions on which norm computation is performed, corresponding to `reduce_axis_2` in the formula.</td>
      <td><ul><li>In the formula, `reduce_axis_0` corresponds to dimension B, and `reduce_axis_1` corresponds to dimension S. Norm computation is not performed on dimensions B and S.</li><li>The value is fixed to an array of length 1, whose fixed element is [H]. Empty value is not supported.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Input</td>
      <td>The second output of the forward computation, which represents the mean of `input` and corresponds to `mean` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Same data type as input `gradOut`, with a bit width no less than that of input `input`.</li><li>The shape is the same as that of `rstd`, in the format of [B..., S, 1]. The last dimension is fixed to 1, and the other dimensions are the same as those of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd (aclTensor*)</td>
      <td>Input</td>
      <td>The third output of the forward computation, representing the reciprocal of the standard deviation of `input`, corresponding to `rstd` in the computation formula.</td>
      <td><ul><li>Empty tensors are not supported.</li>The data type is the same as that of input `gradOut`, and the bit width is not lower than that of input `input`.<li>The shape is the same as the shape of `mean`, which is [B..., S, 1]. The last dimension is fixed to 1, and the other dimensions are the same as those of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>Adaptive scaling parameter. Corresponds to `scale` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type must be the same as the data type of `gradOut`.</li><li>The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions. The number and size of the dimensions must be the same as those of B in `gradOut`, and H must be the same as the H dimension in `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shift (aclTensor*)</td>
      <td>Input</td>
      <td>Adaptive shift parameter.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type must be the same as that of input parameter `gradOut`.</li><li>The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions. The number and sizes of the B dimensions must be the same as those of B in `gradOut`, and H must be the same as the H dimension in `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, indicating the normalization scaling parameter. Corresponds to `weightOptional` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>When the data type is not FLOAT32, the data type must be the same as that of `input`.</li><li>The shape is [H], where H is the same as the H dimension in `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input indicating the normalization bias parameter. It corresponds to `biasOptional` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type is the same as that of the optional parameter `weightOptional`.</li><li>The shape is [H], where H is the same as the H dimension of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInputOut (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, representing the output gradient of backward propagation, corresponding to `gradInputOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>When the pointer `gradInputOut` is not null, the gradient computation result of `input` is output and has the same data type as input `input`.</li><li>The shape is equal to the shape of `input`, which is [B…, S, H].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradScaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output. Indicates the gradient of the adaptive scaling coefficient in backward propagation, corresponding to `gradScaleOut` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>When the pointer `gradScaleOut` is not null, outputs the gradient computation result of `scale`. The data type is the same as that of input `input`.</li><li>The shape is the same as the shape of `scale`, which is [B…, H] or [B…, 1, H].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradShiftOut (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output. Indicates the gradient of the adaptive shift coefficient in backward propagation, corresponding to `gradShiftOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>When the pointer `gradShiftOut` is not empty, outputs the gradient computation result of `shift`. Its data type is the same as that of input `input`.</li><li>The shape is the same as that of `shift`, which is [B..., H] or [B..., 1, H].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeightOut (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, representing the gradient of the normalization scale coefficient in backward propagation, corresponding to `gradWeightOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>When the pointer `gradWeightOut` is not empty, outputs the gradient computation result of `weightOptional`. Its data type is the same as that of input `weightOptional`.</li><li>The shape is the same as that of `weightOptional`, which is [H].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBiasOut (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output. Gradient of the normalization bias coefficient in backward propagation, corresponding to `gradBiasOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>When the pointer `gradBiasOut` is not empty, outputs the gradient computation result of `biasOptional`. Its data type is the same as that of the input `weightOptional`.</li><li>The shape is the same as the shape of `gradWeightOut`, which is [H].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for on the device.</td>
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
      <td>gradOut, input, normalizedShape, mean, rstd, scale, or shift is a null pointer.</td>
    </tr>
   <tr>
      <td rowspan="18">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="18">161002</td>
      <td>The data type of gradOut, weightOptional (if not null), or biasOptional (if not null) is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data types of input, mean, rstd, scale, and shift are different from that of gradOut.</td>
    </tr>
    <tr>
      <td>The data types of weightOptional (if not null) and biasOptional (if not null) are different.</td>
    </tr>
    <tr>
      <td>weightOptional (if not null) and biasOptional (if not null) do not meet the constraint (the data type is the same as that of input when the data type is not FLOAT32).</td>
    </tr>
    <tr>
      <td>The data type of gradInputOut (if not empty), gradScaleOut (if not empty), gradShiftOut (if not empty), gradWeightOut (if not empty), or gradBiasOut (if not empty) is inconsistent with that described in Parameters.</td>
    </tr>
    <tr>
      <td>The dimension of input is less than 2.</td>
    </tr>
    <tr>
      <td>The length of the normalizedShape array is not 1.</td>
    </tr>
    <tr>
      <td>The corresponding dimensions are not equal when the shape array of input and the normalizedShape array are right-aligned.</td>
    </tr>
    <tr>
      <td>The shape of gradOut is not equal to that of input.</td>
    </tr>
    <tr>
      <td>The product of the shape of mean is not equal to the product of the dimensions of input from axis 0 to axis len(input) - len(normalizedShape).</td>
    </tr>
    <tr>
      <td>The product of the shape of rstd is not equal to the product of the dimensions of input from axis 0 to axis len(input) - len(normalizedShape).</td>
    </tr>
    <tr>
      <td>The shape of scale is not [B..., 1, H] or [B..., H].</td>
    </tr>
    <tr>
      <td>The shape of shift is not [B..., 1, H] or [B..., H].</td>
    </tr>
    <tr>
      <td>The shape of gradScaleOut (if not empty) is not equal to that of scale.</td>
    </tr>
    <tr>
      <td>The shape of gradShiftOut (when not empty) is inconsistent with that of shift.</td>
    </tr>
    <tr>
      <td>weightOptional is not empty, and its shape array is inconsistent with the normalizedShape array.</td>
    </tr>
    <tr>
      <td>biasOptional is not empty, and its shape array is inconsistent with the normalizedShape array.</td>
    </tr>
    <tr>
      <td> The shape dimension of gradOut, input, mean, rstd, scale, shift, weightOptional (when not empty), biasOptional (when not empty), gradInputOut (when not empty), gradScaleOut (when not empty), gradShiftOut (when not empty), gradWeightOut (when not empty), or gradBiasOut (when not empty) is greater than 8.</td>
    </tr>
  </tbody></table>

## aclnnAdaLayerNormBackward

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnAdaLayerNormBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the computation flow of the operator.</td>
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

- Mixed precision combinations:
  - The following parameters of the AdaLayerNormBackward operator must have the same data type:

    | Grouping | Specific Parameters Included |
    | --------| --------|
    | **Group A** | `gradOut`, `input`, `rstd`, `mean`, `scale`, `shift`, `gradInputOut`, `gradScaleOut`, `gradShiftOut` |
    | **Group B** | `weightOptional`, `biasOptional`, `gradWeightOut`, `gradBiasOut` |

  - The fixed data type combinations of the two parameter groups above are as follows (the left and right data types strictly correspond):

    | First Group Of Parameters (Group A) | Second Group Of Parameters (Group B) |
    | --------| --------|
    | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 |
    | FLOAT16 | FLOAT32 |
    | BFLOAT16 | BFLOAT16 |
    | BFLOAT16 | FLOAT32 |

- Deterministic computation:
  - aclnnAdaLayerNormBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ada_layer_norm_backward.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
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
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> xShape = {2, 2, 2};
    std::vector<int64_t> meanShape = {2, 2, 1};
    std::vector<int64_t> scaleShape = {2, 2};
    std::vector<int64_t> shiftShape = {2, 2};
    std::vector<int64_t> normShape = {2};
    void* dyDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* dsDeviceAddr = nullptr;
    void* dtDeviceAddr = nullptr;
    void* dwDeviceAddr = nullptr;
    void* dbDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclIntArray* norm = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* ds = nullptr;
    aclTensor* dt = nullptr;
    aclTensor* dw = nullptr;
    aclTensor* db = nullptr;
    std::vector<float> dyHostData = {2, 3, 4, 5, 2, 3, 4, 5};
    std::vector<float> xHostData = {2, 3, 4, 5, 2, 3, 4, 5};
    std::vector<int64_t> normData = {2};
    std::vector<float> meanHostData = {2, 3, 4, 5};
    std::vector<float> rstdHostData = {4, 5, 6, 7};
    std::vector<float> scaleHostData = {1, 1, 2, 2};
    std::vector<float> shiftHostData = {2, 2, 1, 1};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> outHostData(8, 0);
    std::vector<float> dsHostData(4, 0);
    std::vector<float> dtHostData(4, 0);
    std::vector<float> dwHostData(2, 0);
    std::vector<float> dbHostData(2, 0);

    // Create dy aclTensor.
    ret = CreateAclTensor(dyHostData, xShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create normalizedShape aclIntArray.
    norm = aclCreateIntArray(normData.data(), 1);
    CHECK_RET(norm != nullptr, return ret);
    // Create rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create shift aclTensor.
    ret = CreateAclTensor(shiftHostData, shiftShape, &shiftDeviceAddr, aclDataType::ACL_FLOAT, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weight aclTensor.
    ret = CreateAclTensor(weightHostData, normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create bias aclTensor.
    ret = CreateAclTensor(biasHostData, normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create ds aclTensor.
    ret = CreateAclTensor(dsHostData, scaleShape, &dsDeviceAddr, aclDataType::ACL_FLOAT, &ds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dt aclTensor.
    ret = CreateAclTensor(dtHostData, shiftShape, &dtDeviceAddr, aclDataType::ACL_FLOAT, &dt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dw aclTensor.
    ret = CreateAclTensor(dwHostData, normShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create db aclTensor.
    ret = CreateAclTensor(dbHostData, normShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, and change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAdaLayerNormBackward.
    ret = aclnnAdaLayerNormBackwardGetWorkspaceSize(
        dy, x, norm, rstd, mean, scale, shift, weight, bias, out, ds, dt, dw, db, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaLayerNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnAdaLayerNormBackward.
    ret = aclnnAdaLayerNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaLayerNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify it according to the specific API definition.
    auto size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(scaleShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), dsDeviceAddr, size1 * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("ds result[%ld] is: %f\n", i, resultData1[i]);
    }

    auto size2 = GetShapeSize(shiftShape);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), dtDeviceAddr, size2 * sizeof(resultData2[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("dt result[%ld] is: %f\n", i, resultData2[i]);
    }

    auto size3 = GetShapeSize(normShape);
    std::vector<float> resultData3(size3, 0);
    ret = aclrtMemcpy(
        resultData3.data(), resultData3.size() * sizeof(resultData3[0]), dwDeviceAddr, size3 * sizeof(resultData3[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size3; i++) {
        LOG_PRINT("dw result[%ld] is: %f\n", i, resultData3[i]);
    }

    auto size4 = GetShapeSize(normShape);
    std::vector<float> resultData4(size4, 0);
    ret = aclrtMemcpy(
        resultData4.data(), resultData4.size() * sizeof(resultData4[0]), dbDeviceAddr, size4 * sizeof(resultData4[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size4; i++) {
        LOG_PRINT("db result[%ld] is: %f\n", i, resultData4[i]);
    }

    // 6. Release aclTensor, aclIntArray, and aclBoolArray. Modify it according to the specific API definition.
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyIntArray(norm);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(ds);
    aclDestroyTensor(dt);
    aclDestroyTensor(dw);
    aclDestroyTensor(db);

    // 7. Release device resources.
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(dsDeviceAddr);
    aclrtFree(dtDeviceAddr);
    aclrtFree(dwDeviceAddr);
    aclrtFree(dbDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
