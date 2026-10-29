# aclnnBatchNormElemt

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:55:29.662Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Takes the global mean and the reciprocal of the standard deviation as operator inputs and performs BatchNorm computation on x. This operator is an element-wise BatchNorm function used to normalize input data in specific use cases. Compared with [aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm.md), aclnnBatchNormElemt may be adjusted for specific hardware or optimization requirements.

- Formula:

  $$
  y = \frac{(x-E[x])}{\sqrt{Var(x)+ eps}} * weight + bias
  $$

  The relationship between standard deviation and variance is as follows:

  $$
  \frac{1}{S} = \frac{1}{\sqrt{Var(x) + eps}}
  $$

  Where:

  - E[x] denotes the mean of x.
  - Var(x) denotes the variance of x.
  - S denotes the standard deviation of x.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnBatchNormElemtGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size according to the process, and then call "aclnnBatchNormElemt" to perform the computation.

```Cpp
aclnnStatus aclnnBatchNormElemtGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* weight,
  const aclTensor* bias,
  aclTensor*       mean,
  aclTensor*       invstd,
  double           eps,
  aclTensor*       output,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormElemt(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormElemtGetWorkspaceSize

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
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the BatchNorm computation, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Supported shapes and formats: 2D (format NC), 3D (format NCL), 4D (format NCHW), 5D (format NCDHW), and 6-8D (format ND, where the second dimension is fixed to the channel axis).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NCDHW, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input. Weight tensor for the BatchNorm computation, corresponding to `weight` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type must be the same as the data type of `input`.</li><li>The shape length must be the same as the length of the channel axis of the `input` parameter.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input, the bias tensor for the BatchNorm computation, corresponding to `bias` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type must be the same as the data type of `input`.</li><li>The shape length must be the same as the length of the channel axis of the `input` parameter.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean(aclTensor*)</td>
      <td>Input</td>
      <td>Specifies the mean of the input data, corresponding to `E(x)` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type is the same as the data type of `input`.</li><li>The shape length is equal to the length of the channel axis in `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the reciprocal of the standard deviation of the input data, that is, the reciprocal of the square root of `Var(x) + eps` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Supports the use case where all element values are greater than 0.</li><li>The data type must be the same as that of `input`.</li><li>The shape length must be equal to the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
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
      <td>output (aclTensor*)</td>
      <td>Output</td>
      <td>Final output result. Corresponds to `y` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape are the same as those of the input `input`.</li><li>The shape is the same as that of the input parameter `input`. Supported shapes and formats: 2D (format: NC), 3D (format: NCL), 4D (format: NCHW), 5D (format: NCDHW), and 6-8D (format: ND, where the second dimension is fixed to the channel axis).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NCDHW, ND</td>
      <td>2-8</td>
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
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: The data types of `input`, `weight`, `bias`, `mean`, `invstd`, and `output` do not support BFLOAT16.
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
      <td>A required pointer-type input parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data types and data formats of input and output are out of the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of input and output data are out of the supported range.</td>
    </tr>
    <tr>
      <td>An empty tensor with a C dimension of 0 is not supported for input.</td>
    </tr>
    <tr>
      <td>mean/invstd is not 1-dimensional, or its shape is not equal to the length of the C dimension of input.</td>
    </tr>
    <tr>
      <td>When weight is not empty, weight is not 1-dimensional, or its shape is not equal to the length of the C dimension of input.</td>
    </tr>
    <tr>
      <td>When bias is not empty, bias is not 1-dimensional, or its shape is not equal to the length of the C dimension of input.</td>
    </tr>
    <tr>
      <td>The data formats of input and output are inconsistent.</td>
    </tr>
    <tr>
      <td>When weight and bias are not empty, the data types of input, bias, weight, mean, invstd, and output are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of input and output are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormElemt

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
      <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnBatchNormElemtGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation process.</td>
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

- Deterministic computation:

  <!-- npu="A3,910b,910" id8 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas training products</term>: aclnnBatchNormElemt defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnBatchNormElemt defaults to a deterministic implementation.
  <!-- end id9 -->

## Example

The following sample code is for reference only. For details about the compilation and running procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_elemt.h"

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
    // Boilerplate code for resource initialization
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
    // Call aclrtMalloc to allocate memory on the device
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host to the device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device/stream, see the acl API manual for details
    // Fill in deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output, and customize the construction based on the API
    std::vector<int64_t> inputShape = {2, 4, 2};
    std::vector<int64_t> meanShape = {4};
    std::vector<int64_t> invstdShape = {4};
    std::vector<int64_t> outShape = {2, 4, 2};
    double eps = 1e-2;
    void* inputDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* invstdDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* invstd = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<float> meanHostData = {1, 2, 3, 4};
    std::vector<float> invstdHostData = {5, 6, 7, 8};
    std::vector<float> outHostData(16, 0);
    // Create the input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the invstd aclTensor.
    ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNormElemt API call example
    // 3. Call the CANN operator library API. Change it to the specific API name.
    // Call the first-phase API of aclnnBatchNormElemt.
    ret =
        aclnnBatchNormElemtGetWorkspaceSize(input, nullptr, nullptr, mean, invstd, eps, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormElemtGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnBatchNormElemt.
    ret = aclnnBatchNormElemt(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormElemt failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait synchronously for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result on Device memory to Host memory. Modify based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify based on the specific API definition.
    aclDestroyTensor(input);
    aclDestroyTensor(mean);
    aclDestroyTensor(invstd);
    aclDestroyTensor(out);

    // 7. Release Device resources. Modify based on the specific API definition.
    aclrtFree(inputDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(invstdDeviceAddr);
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
