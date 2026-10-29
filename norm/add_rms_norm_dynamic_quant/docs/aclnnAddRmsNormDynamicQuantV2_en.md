# aclnnAddRmsNormDynamicQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:53:21.762Z -->

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

- Description: The RmsNorm operator is a commonly used normalization operation in large models. Compared with the LayerNorm operator, it removes the part of subtracting the mean. The DynamicQuant operator performs symmetric dynamic quantization on input tensors. The AddRmsNormDynamicQuant operator fuses the Add operator before RmsNorm and one or two DynamicQuant operators that receive the RmsNorm normalized output, reducing data movement operations. Compared with aclnnAddRmsNormDynamicQuant, aclnnAddRmsNormDynamicQuantV2 adds the bias term betaOptional parameter during the RmsNorm computation, i.e., the beta in the formula, and adds the output configuration item outputMaskOptional parameter, which configures whether to output the quantization result at the corresponding position.

- Formula:

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  input1 =\begin{cases}
    y\cdot smoothScale1Optional & \ \ smoothScale1Optional \\
    y & !\ smoothScale1Optional
    \end{cases}
  $$

  $$
  input2 =\begin{cases}
    y\cdot smoothScale2Optional & \ \ smoothScale2Optional \\
    y & !\ smoothScale2Optional
    \end{cases}
  $$

  $
  scale1Out=\begin{cases}
    row\_max(abs(input1))/127 & (outputMask[0]=True\ ||\ !outputMask) \& y1Out\ \text{is INT8} \\
    row\_max(abs(input1))/7 & (outputMask[0]=True\ ||\ !outputMask) \& y1Out\ \text{is INT4} \\
    \text{invalid output} & outputMask[0]=False
    \end{cases}
  $

  $
  y1Out=\begin{cases}
    round(input1/scale1Out) & outputMask[0]=True\ ||\ !outputMask \\
    \text{invalid output} & outputMask[0]=False
    \end{cases}
  $

  $$
  scale2Out=\begin{cases}
    row\_max(abs(input2))/127 & (outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional)) \& y2Out is INT8 \\
    row\_max(abs(input2))/7 & (outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional)) \& y2Out is INT4 \\
    invalid output & outputMask[1]=False\ ||\ (!outputMask\ \&\ (!smoothScale1Optional\ ||\ !smoothScale2Optional))
    \end{cases}
  $$

  $$
  y2Out=\begin{cases}
    round(input2/scale2Out) & outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional)\\
    invalid output & outputMask[1]=False\ ||\ (!outputMask\ \&\ (!smoothScale1Optional\ ||\ !smoothScale2Optional))
    \end{cases}
  $$

  In the formula, row\_max represents the maximum value of each row.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize` first to obtain the workspace size required for computation and the executor containing the operator computation flow. Then call `aclnnAddRmsNormDynamicQuantV2` to execute the computation.

```Cpp
aclnnStatus aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(
  const aclTensor    *x1,
  const aclTensor    *x2,
  const aclTensor    *gamma,
  const aclTensor    *smoothScale1Optional,
  const aclTensor    *smoothScale2Optional,
  const aclTensor    *betaOptional,
  double              epsilon,
  const aclBoolArray *outputMaskOptional,
  aclTensor          *y1Out,
  aclTensor          *y2Out,
  aclTensor          *xOut,
  aclTensor          *scale1Out,
  aclTensor          *scale2Out,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormDynamicQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormDynamicQuantV2

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
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process. Corresponds to `x1` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When the data type of output `y1Out` or `y2Out` is INT4, the last dimension of `x1` must be divisible by 2.</li><li>When the data type of output `y1Out` or `y2Out` is INT32, the last dimension of `x1` must be 8 times the last dimension of output `y1Out` or `y2Out`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process. Corresponds to `x2` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape and data type need to be consistent with `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type needs to be consistent with `x1`.</li><li>The shape needs to be consistent with the last dimension of `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScale1Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the smoothScale tensor used to obtain `y1Out` in the quantization process. For the corresponding formula, see `smoothScale1Optional`.</td>
      <td><ul><li>Empty tensors are supported.</li><li>This is an optional parameter, and passing a null pointer is supported.</li><li>The shape and data type need to be consistent with those of `gamma`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScale2Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the smoothScale tensor used to obtain `y2Out` in the quantization process. Corresponds to `smoothScale2Optional` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Optional parameter; supports passing a null pointer.</li><li>Shape and data type need to be consistent with `gamma`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the bias term in the normalization process. It corresponds to `beta` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Optional parameter. Passing a null pointer is supported.</li><li>The shape and data type need to be consistent with `gamma`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Represents the value used to prevent division-by-zero errors. It corresponds to `epsilon` in the corresponding formula.</td>
      <td>It is recommended to pass in a small positive number, such as 1e-6.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMaskOptional (aclBoolArray*)</td>
      <td>Input</td>
      <td><ul><li>Represents the output mask, corresponding to `outputMask` in the formula.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>Supports passing a null pointer or an array of length 2.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output tensor, corresponding to `y1Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>If `y1Out` is a valid output, the shape and data type need to be consistent with those of input `x1`; if `y1Out` is an invalid output, the shape is [1].</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>INT4, INT32, INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to `y2Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>If `y2Out` is a valid output, its shape and data type need to be consistent with `y1Out`; if `y2Out` is an invalid output, its shape is [1].</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>INT4, INT32, INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the sum of x1 and x2, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape and data type need to be consistent with input `x1`/`x2`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the output of the first quantization path, corresponding to `scale1Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>If this output is a valid output, the shape needs to be consistent with the shape of `x1` except the last dimension, or the product of the dimensions of `x1` except the last dimension; otherwise, the shape is [1].</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the output of the second quantization path, corresponding to `scale2Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>If this output is a valid output, the shape needs to be consistent with the shape of `x1` except the last dimension, or the product of the dimensions of `x1` except the last dimension; otherwise, the shape is [1].</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns an op executor that contains the operator computation flow.</td>
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
      <td>If a passed-in parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>When outputMaskOptional is a null pointer, smoothScale2Optional is passed but smoothScale1Optional is not.</td>
    </tr>
    <tr>
      <td>When outputMaskOptional is not a null pointer, the length of outputMaskOptional is not 2, or smoothScale1Optional or smoothScale2Optional is not empty but the corresponding position in outputMaskOptional is false.</td>
    </tr>
    <tr>
      <td>When both y1Out and y2Out are valid outputs, their data types are different.</td>
    </tr>
    <tr>
      <td>When the data type of y1Out or y2Out is DT_INT32, the last dimension of y1Out or y2Out is not 1/8 of the last dimension of x1.</td>
    </tr>
    <tr>
      <td>When the data type of y1Out or y2Out is not DT_INT32, the shape of y1Out or y2Out is inconsistent with the shape of x1.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of the input or output does not meet the constraints.</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormDynamicQuantV2

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
      <td>Workspace memory address of the Device side application.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size of the Device side application, obtained by calling the first-phase API aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator calculation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. (For details, see [aclnn return code](../../../docs/en/context/aclnn_return_code.md))

## Constraints

- **Data Format Description**

  It is recommended to use the ND format for all input and output tensors. Other data formats are converted to the ND format by the framework by default for processing.

- **Input Parameter Constraints**

  - When outputMaskOptional is not empty and smoothScale1Optional has a value, outputMaskOptional[0] must be True. When smoothScale2Optional has a value, outputMaskOptional[1] must be True.
  - When outputMaskOptional is not empty, outputMaskOptional[0] and outputMaskOptional[1] cannot both be False.
  - When outputMaskOptional is empty and smoothScale2Optional has a value, smoothScale1Optional must also have a value.

- **Output Constraints**

  - When outputMaskOptional[0] is True, y1Out and scale1Out are valid outputs; when outputMaskOptional[0] is False, y1Out and scale1Out are invalid outputs.
  - When outputMaskOptional[1] is True, y2Out and scale2Out are valid outputs; when outputMaskOptional[1] is False, y2Out and scale2Out are invalid outputs.
  - When outputMaskOptional is empty, y1Out and scale1Out are always valid outputs.
  - When outputMaskOptional is empty, y2Out and scale2Out are valid outputs when both smoothScale1Optional and smoothScale2Optional are valid; otherwise, they are invalid outputs.

- **Supported Data Types by Product Model**

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    | `x1`/`x2`/`gamma`/`smoothScale1`/`smoothScale2`/`beta` Data Type | `y1Out`/`y2Out` Data Type | `xOut` Data Type | `scale1Out`/`scale2Out` Data Type |
    | :----------------------------------------------: | :-----------------------: | :-------------: | :------------------------------: |
    |                     FLOAT16                      |           INT8            |     FLOAT16     |              FLOAT32             |
    |                     BFLOAT16                     |           INT8            |    BFLOAT16     |              FLOAT32             |
    |                     FLOAT16                      |       FLOAT8_E4M3FN       |     FLOAT16     |              FLOAT32             |
    |                     BFLOAT16                     |       FLOAT8_E4M3FN       |    BFLOAT16     |              FLOAT32             |
    |                     FLOAT16                      |        FLOAT8_E5M2        |     FLOAT16     |              FLOAT32             |
    |                     BFLOAT16                     |        FLOAT8_E5M2        |    BFLOAT16     |              FLOAT32             |
    |                     FLOAT16                      |          HIFLOAT8         |     FLOAT16     |              FLOAT32             |
    |                     BFLOAT16                     |          HIFLOAT8         |    BFLOAT16     |              FLOAT32             |
    |                     FLOAT16                      |           INT4            |     FLOAT16     |              FLOAT32             |
    |                     BFLOAT16                     |           INT4            |    BFLOAT16     |              FLOAT32             |
    |                     FLOAT16                      |           INT32           |     FLOAT16     |              FLOAT32             |
    |                     BFLOAT16                     |           INT32           |    BFLOAT16     |              FLOAT32             |
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.

    | `x1`/`x2`/`gamma`/`smoothScale1`/`smoothScale2`/`beta` Data Type | `y1Out`/`y2Out` Data Type | `xOut` Data Type | `scale1Out`/`scale2Out` Data Type |
    | :----------------------------------------------: | :-----------------------: | :-------------: | :------------------------------: |
    |                     FLOAT16                      |           INT8            |     FLOAT16     |              FLOAT32             |
    |                     BFLOAT16                     |           INT8            |    BFLOAT16     |              FLOAT32             |
    |                     FLOAT16                      |           INT4            |     FLOAT16     |              FLOAT32             |
    |                     BFLOAT16                     |           INT4            |    BFLOAT16     |              FLOAT32             |
  <!-- end id8 -->
- Deterministic computation:
  - aclnnAddRmsNormDynamicQuantV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_dynamic_quant_v2.h"

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
    // Boilerplate, resource initialization
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
    // Call aclrtMalloc to apply for memory on the device side
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs. Customize the construction based on the API.
    std::vector<int64_t> xShape = {2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> betaShape = {8};
    std::vector<int64_t> reduceShape = {2};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* smooth1DeviceAddr = nullptr;
    void* smooth2DeviceAddr = nullptr;

    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* smooth1 = nullptr;
    aclTensor* smooth2 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scale1 = nullptr;
    aclTensor* scale2 = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t gammaShapeSize = GetShapeSize(gammaShape);
    int64_t betaShapeSize = GetShapeSize(betaShape);
    int64_t reduceShapeSize = GetShapeSize(reduceShape);

    std::vector<short> x1HostData(xShapeSize, 0x3800);
    std::vector<short> x2HostData(xShapeSize, 0x3800);
    std::vector<short> gammaHostData(gammaShapeSize, 0x3e00);
    std::vector<short> betaHostData(betaShapeSize, 0x3e00);
    std::vector<short> smooth1HostData(gammaShapeSize, 0x3e00);
    std::vector<short> smooth2HostData(gammaShapeSize, 0x3e00);

    std::vector<short> y1HostData(xShapeSize, 0);
    std::vector<short> y2HostData(xShapeSize, 0);
    std::vector<short> xHostData(xShapeSize, 0);
    std::vector<short> scale1HostData(reduceShapeSize, 0);
    std::vector<short> scale2HostData(reduceShapeSize, 0);

    float epsilon = 1e-6;

    // Create the x1 aclTensor.
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x2 aclTensor.
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create beta aclTensor.
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create smooth1 aclTensor.
    ret = CreateAclTensor(smooth1HostData, gammaShape, &smooth1DeviceAddr, aclDataType::ACL_FLOAT16, &smooth1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create smooth2 aclTensor.
    ret = CreateAclTensor(smooth2HostData, gammaShape, &smooth2DeviceAddr, aclDataType::ACL_FLOAT16, &smooth2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create y1 aclTensor.
    ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create y2 aclTensor.
    ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create outScale1 aclTensor.
    ret = CreateAclTensor(scale1HostData, reduceShape, &scale1DeviceAddr, aclDataType::ACL_FLOAT, &scale1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create outScale2 aclTensor.
    ret = CreateAclTensor(scale2HostData, reduceShape, &scale2DeviceAddr, aclDataType::ACL_FLOAT, &scale2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAddRmsNormDynamicQuantV2.
    ret = aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(
        x1, x2, gamma, smooth1, smooth2, beta, epsilon, nullptr, y1, y2, x, scale1, scale2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddRmsNormDynamicQuantV2.
    ret = aclnnAddRmsNormDynamicQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicQuantV2 failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronously wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device side memory to the host side. Modify the code based on the actual API definition.
    auto size = GetShapeSize(xShape);
    std::vector<int8_t> y1Ret(size, 0);
    ret = aclrtMemcpy(
        y1Ret.data(), y1Ret.size() * sizeof(y1Ret[0]), y1DeviceAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, y1Ret[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the actual API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(smooth1);
    aclDestroyTensor(smooth2);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(x);
    aclDestroyTensor(scale1);
    aclDestroyTensor(scale2);

    // 7. Release device resources. Modify the code based on the actual API definition.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(smooth1DeviceAddr);
    aclrtFree(smooth2DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(scale1DeviceAddr);
    aclrtFree(scale2DeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
