# aclnnLayerNormBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:15:22.621Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Backward propagation of [aclnnLayerNorm](../../layer_norm_v4/docs/aclnnLayerNorm&aclnnLayerNormWithImplMode.md). Computes the gradients of the input tensors for updating model parameters during backward propagation.
- Formula:

  $$
  res\_for\_gamma = (input - mean) \times rstd
  $$

  $$
  dy\_g = gradOut \times weightOptional
  $$

  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_1} gradOut \times weightOptional
  $$

  $$
  temp_2 = 1/N \times (input - mean) \times rstd \times \sum_{reduce\_axis\_1}(gradOut \times weightOptional \times (input - mean) \times rstd)
  $$

  $$
  gradInputOut = (gradOut \times weightOptional - (temp_1 + temp_2)) \times rstd
  $$

  $$
  gradWeightOut =  \sum_{reduce\_axis\_0}gradOut \times (input - mean) \times rstd
  $$

  $$
  gradBiasOut = \sum_{reduce\_axis\_0}gradOut
  $$

  Where N is the dimension of the axis for normalization computation, that is, the size of the normalized axis dimension.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnLayerNormBackwardGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call the "aclnnLayerNormBackward" API to perform computation.

```Cpp
aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(
  const aclTensor    *gradOut,
  const aclTensor    *input,
  const aclIntArray  *normalizedShape,
  const aclTensor    *mean,
  const aclTensor    *rstd,
  const aclTensor    *weightOptional,
  const aclTensor    *biasOptional,
  const aclBoolArray *outputMask,
  aclTensor          *gradInputOut,
  aclTensor          *gradWeightOut,
  aclTensor          *gradBiasOut,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnLayerNormBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLayerNormBackwardGetWorkspaceSize

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
      <td>gradOut (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the gradient tensor for backward computation, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Same data type as input.</li><li>The shape is equal to the shape of input, that is, [A1,...,Ai,R1,...,Rj], and the shape length is greater than or equal to the length of normalizedShape.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the first input of forward computation, corresponding to `input` in the formula.</td>
      <td><ul><li>Same as the data type of input gradOut.</li><li>The shape is equal to that of gradOut and is [A1,...,Ai,R1,...,Rj]. The shape length is greater than or equal to the length of normalizedShape.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape (aclIntArray*)</td>
      <td>Input</td>
      <td>Specifies the dimensions for norm computation, corresponding to reduce_axis_1 in the formula.</td>
      <td><ul><li>reduce_axis_0 in the formula refers to the dimensions on which norm computation is not performed.</li><li>The value is [R1,...,Rj]. The length is less than or equal to the shape length of input. Empty values are not supported.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean(aclTensor*)</td>
      <td>Input</td>
      <td>Second output of the forward computation, indicating the mean of input, corresponding to mean in the computation formula.</td>
      <td><ul><li>Same data type as input rstd, with a bit width not lower than that of input input.</li><li>Shape is equal to that of rstd, which is [A1,...,Ai,1,...,1], with a total of j 1s after Ai, kept the same as the length of the axes to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd (aclTensor*)</td>
      <td>Input</td>
      <td>Third output of forward computation, denoting the reciprocal of the standard deviation of input, corresponding to rstd in the computation formula.</td>
      <td><ul><li>Same data type as the input mean, with a bit width not lower than the data type bit width of input.</li><li>Its shape is equal to that of mean, i.e., [A1,...,Ai,1,...,1], with a total of j 1s after Ai, matching the length of the axes to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional parameter, indicating the weight, corresponding to weightOptional in the formula.</td>
      <td><ul><li>When non-null, weightOptional has the same data type as input or is of the FLOAT type, and has the same data type as biasOptional when biasOptional exists.</li><li>When weightOptional is null, construct a Tensor with shape [R1,...,Rj], the same data type as input, and all values of 1.</li><li>The shape is equal to normalizedShape, that is, [R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional parameter, indicating the bias.</td>
      <td><ul><li>When biasOptional is non-null, the data type must be the same as that of input or be FLOAT, and the same as the data type of weightOptional when weightOptional exists.</li><li>When biasOptional is null, no operation is performed.</li><li>shape is equal to normalizedShape, that is, [R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputMask (aclBoolArray*)</td>
      <td>Input</td>
      <td>Indicates the output mask.</td>
      <td><ul><li>The length is fixed to 3.</li><li>When the value is True, the output at the corresponding position is non-null.</li></ul></td>
      <td>BoolArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInputOut (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, indicating the output gradient of backward propagation, corresponding to `gradInputOut` in the formula.</td>
      <td><ul><li>Whether to output is controlled by the 0th element of outputMask. Output is performed when the 0th element of outputMask is True. Same data type as input.</li><li>shape is equal to the shape of input, that is, [A1,...,Ai,R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeightOut (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output representing the gradient of the weight in backward propagation, corresponding to `gradWeightOut` in the formula.</td>
      <td><ul><li>Whether to output is controlled by the first element of outputMask. The output is generated when the first element of outputMask is True. The data type is the same as that of input weightOptional.</li><li>The shape is equal to the shape of gradBiasOut, which is [R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBiasOut (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, representing the gradient of the bias in backward propagation, corresponding to `gradBiasOut` in the calculation formula.</td>
      <td><ul><li>Whether to output is controlled by the 2nd element of outputMask; output is performed when the 2nd element of outputMask is True. The data type is the same as that of the input weightOptional.</li><li>The shape is equal to that of gradWeightOut, which is [R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
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
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>:

    The data type of the parameters `gradOut`, `input`, `mean`, `rstd`, `weightOptional`, `biasOptional`, `gradInputOut`, `gradWeightOut`, and `gradBiasOut` does not support BFLOAT16.
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
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
      <td>gradOut, input, normalizedShape, mean, rstd, and outputMask are null pointers.</td>
    </tr>
    <tr>
      <td>outputMask[0] is True and gradInputOut is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[1] is True and gradWeightOut is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[2] is True and gradBiasOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data type of gradOut, input, mean, rstd, weightOptional (when non-null), or biasOptional (when non-null) is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of gradOut is not equal to the shape of input.</td>
    </tr>
    <tr>
      <td>The dimension of normalizedShape is less than 1.</td>
    </tr>
    <tr>
      <td>The product of the shape of mean is not equal to the product of the dimensions of input from axis 0 to axis len(input) - len(normalizedShape).</td>
    </tr>
    <tr>
      <td>The product of the rstd shape is not equal to the product of the input shape from axis 0 to axis len(input) - len(normalizedShape).</td>
    </tr>
    <tr>
      <td>weightOptional is non-null and its shape is not equal to normalizedShape.</td>
    </tr>
    <tr>
      <td>biasOptional is non-null and its shape is not equal to normalizedShape.</td>
    </tr>
    <tr>
      <td>The number of dimensions of input is less than that of normalizedShape.</td>
    </tr>
    <tr>
      <td>The shapes of the corresponding dimensions are not equal when the input shape is right-aligned with normalizedShape.</td>
    </tr>
    <tr>
      <td>The length of outputMask is not 3.</td>
    </tr>
    <tr>
      <td> The shape dimension of gradOut, input, mean, rstd, weightOptional (when non-null), biasOptional (when non-null), gradInputOut (when non-null), gradWeightOut (when non-null), or gradBiasOut (when non-null) exceeds 8 dimensions or is less than 1 dimension.</td>
    </tr>
  </tbody></table>

## aclnnLayerNormBackward

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
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnLayerNormBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Shape constraints:

  gradOut, input, mean, rstd, weightOptional (when non-null), biasOptional (when non-null), gradInputOut (when non-null), gradWeightOut (when non-null), gradBiasOut (when non-null): the shape supports 1 to 8 dimensions.

- Empty tensor support:

  input supports empty tensors.

- Deterministic computation:

  aclnnLayerNormBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm_backward.h"

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
    // Boilerplate code, resource initialization.
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
    // Call aclrtMalloc to request memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to device memory.
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
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. The construction must be customized based on the API.
    std::vector<int64_t> xShape = {2, 2};
    std::vector<int64_t> meanShape = {2, 1};
    std::vector<int64_t> normShape = {2};
    void* dyDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* dwDeviceAddr = nullptr;
    void* dbDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclIntArray* norm = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclBoolArray* mask = nullptr;
    aclTensor* out = nullptr;
    aclTensor* dw = nullptr;
    aclTensor* db = nullptr;
    std::vector<float> dyHostData = {2, 3, 4, 5};
    std::vector<float> xHostData = {2, 3, 4, 5};
    std::vector<int64_t> normData = {2};
    std::vector<float> meanHostData = {2, 3};
    std::vector<float> rstdHostData = {4, 5};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> outHostData(4, 0);
    std::vector<float> dwHostData(2, 0);
    std::vector<float> dbHostData(2, 0);

    // Create the dy aclTensor.
    ret = CreateAclTensor(dyHostData, xShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create normalizedShape aclIntArray.
    norm = aclCreateIntArray(normData.data(), 1);
    CHECK_RET(norm != nullptr, return ret);
    // Create mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weight aclTensor.
    ret = CreateAclTensor(weightHostData, normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create bias aclTensor.
    ret = CreateAclTensor(biasHostData, normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create outputMask aclBoolArray.
    bool maskData[3] = {true, true, true};
    mask = aclCreateBoolArray(&(maskData[0]), 3);
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
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
    // Call the first-phase API of aclnnLayerNormBackward.
    ret = aclnnLayerNormBackwardGetWorkspaceSize(
        dy, x, norm, mean, rstd, weight, bias, mask, out, dw, db, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnLayerNormBackward.
    ret = aclnnLayerNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the results from the device memory to the host memory. Modify the code based on the specific API definition.
    auto size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(normShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), dwDeviceAddr, size1 * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("dw result[%ld] is: %f\n", i, resultData1[i]);
    }

    auto size2 = GetShapeSize(normShape);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), dbDeviceAddr, size2 * sizeof(resultData2[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("db result[%ld] is: %f\n", i, resultData2[i]);
    }

    // 6. Release aclTensor, aclIntArray, and aclBoolArray. Modify the code based on the specific API definition.
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyIntArray(norm);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyBoolArray(mask);
    aclDestroyTensor(out);
    aclDestroyTensor(dw);
    aclDestroyTensor(db);

    // 7. Release device resources.
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
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
