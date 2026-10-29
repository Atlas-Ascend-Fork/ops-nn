# aclnnBatchNormGatherStatsWithCounts

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:40:27.019Z -->

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

- Description:

  Collects the mean and variance from all devices, and updates the global mean and the inverse of the standard deviation. The performance of BatchNorm is related to BatchSize. A larger BatchSize yields more accurate BatchNorm statistics. However, for tasks such as detection, which consume a large amount of video memory, a single GPU often uses only a small number of images, for example two, for training. This degrades the performance of BatchNorm. One solution is SyncBatchNorm, where all devices share the same BatchNorm to obtain global statistics.

  The computation of aclnnBatchNormGatherStatsWithCounts depends on aclnnBatchNormStats to calculate the mean and the inverse of the standard deviation for single-device data.

- Formula:

  $$
  y = \frac{(x-E[x])}{\sqrt{Var(x)+ eps}} * γ + β
  $$

  where the update formulas for runningMean and runningVar are as follows:

  $$
      runningMean=runningMean*(1-momentum) + E[x]*momentum
  $$

  $$
      runningVar=runningVar*(1-momentum) + Var(x)*momentum
  $$

## Function Prototype

Each operator corresponds to a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size according to the process, and then call the "aclnnBatchNormGatherStatsWithCounts" API to perform the computation.

```Cpp
aclnnStatus aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  aclTensor*       runningMean,
  aclTensor*       runningVar,
  double           momentum,
  double           eps,
  const aclTensor* counts,
  aclTensor*       meanAll,
  aclTensor*       invstdAll,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormGatherStatsWithCounts(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize

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
      <td>Indicates the sample values for statistics computation. Corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The second dimension is fixed to the channel axis.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the mean of input data, corresponding to `E(x)` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The size of the first dimension must be consistent with `invstd` and `counts`, and the size of the second dimension must be consistent with the Channel axis size of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the reciprocal of the standard deviation of the input data, corresponding to the reciprocal of the square root of `Var(x)+ eps` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Element values must all be greater than 0. If an element value is less than or equal to 0, precision is not guaranteed.</li><li>The size of the first dimension must be consistent with `mean` and `counts`, and the size of the second dimension must be consistent with the Channel axis size of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean (aclTensor*)</td>
      <td>Input</td>
      <td>Optional parameter, indicates the mean used for tracking the entire training process, corresponding to `runningMean` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Element values must all be greater than 0. Precision is not performed when the value is less than or equal to 0.</li><li>When `runningMean` is a non-null pointer, the shape supports 1 dimension, and the size must be consistent with the Channel axis size in `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar (aclTensor*)</td>
      <td>Input</td>
      <td>Optional parameter, indicates the variance used to track the entire training process, corresponding to `runningVar` in the formula.</td>
      <td><ul><li>Empty tensor not supported.</li><li>Element values must all be greater than 0. Precision is not performed when the value is less than or equal to 0.</li><li>When `runningVar` is a non-null pointer, 1-D shape is supported, and the size must be consistent with the size of the Channel axis in `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>momentum (double)</td>
      <td>Input</td>
      <td>Indicates the exponential smoothing parameter of runningMean and runningVar, corresponding to `momentum` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Indicates the value added to the variance to avoid division by zero. Corresponds to `eps` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>counts(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of input data elements.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Element values must all be positive integers; other cases are not guaranteed.</li><li>The size of the first dimension must be consistent with the size of the first dimension of mean and invstd.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanAll(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the mean of the data on all devices, corresponding to `E(x)` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape must be consistent with the channel axis in input parameter `input`.</li><li>The data format must be consistent with the data format in input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstdAll (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the reciprocal of the standard deviation of data on all devices, that is, the reciprocal of the square root of `Var(x)+ eps` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape must be consistent with the channel axis in input parameter `input`.</li><li>The data format must be consistent with the data format in input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied by the user on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type of the parameters <codeph>input</codeph>, <codeph>mean</codeph>, <codeph>invstd</codeph>, <codeph>runningMean</codeph>, <codeph>runningVar</codeph>, <codeph>counts</codeph>, <codeph>meanAll</codeph>, and <codeph>invstdAll</codeph> does not support BFLOAT16.
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
      <td>The input, mean, invstd, counts, meanAll, or invstdAll passed in is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data type of input, meanAll, invstdAll, mean, invstd, runningMean, runningVar, or counts is out of the supported range.</td>
    </tr>
    <tr>
      <td>input has more than 8 dimensions.</td>
    </tr>
    <tr>
      <td>input has fewer than 2 dimensions.</td>
    </tr>
    <tr>
      <td>mean or invstd does not have 2 dimensions.</td>
    </tr>
    <tr>
      <td>The sizes of counts, mean, and invstd on the first dimension are inconsistent.</td>
    </tr>
    <tr>
      <td>The size of mean or invstd on the second dimension is inconsistent with the channel axis of input.</td>
    </tr>
    <tr>
      <td>If runningMean is not empty, the size of runningMean mismatches the channel axis of input.</td>
    </tr>
    <tr>
      <td>If runningVar is not empty, the size of runningVar mismatches the channel axis of input.</td>
    </tr>
    <tr>
      <td>The dimension of counts is not 1.</td>
    </tr>
    <tr>
      <td>The shape of meanAll or invstdAll mismatches the channel axis of input.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormGatherStatsWithCounts

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
      <td>Workspace memory address applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied by the Device side, obtained by calling the first-phase API aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor that contains the operator computation flow.</td>
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
  - aclnnBatchNormGatherStatsWithCounts defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_gather_stats_with_counts.h"

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
    // Call aclrtMalloc to apply for memory on the Device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the Host side to the memory on the Device side.
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
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output, which need to be custom constructed according to the API interface.
    std::vector<int64_t> inputShape = {2, 4, 2};
    std::vector<int64_t> meanShape = {4, 4};
    std::vector<int64_t> rMeanShape = {4};
    std::vector<int64_t> rVarShape = {4};
    std::vector<int64_t> countsShape = {4};
    std::vector<int64_t> invstdShape = {4, 4};
    std::vector<int64_t> meanAllShape = {4};
    std::vector<int64_t> invstdAllShape = {4};
    void* inputDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rMeanDeviceAddr = nullptr;
    void* rVarDeviceAddr = nullptr;
    void* countsDeviceAddr = nullptr;
    void* invstdDeviceAddr = nullptr;
    void* meanAllDeviceAddr = nullptr;
    void* invstdAllDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rMean = nullptr;
    aclTensor* rVar = nullptr;
    aclTensor* counts = nullptr;
    aclTensor* invstd = nullptr;
    aclTensor* meanAll = nullptr;
    aclTensor* invstdAll = nullptr;
    std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<float> meanHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> rMeanHostData = {1, 2, 3, 4};
    std::vector<float> rVarHostData = {5, 6, 7, 8};
    std::vector<float> countsHostData = {1, 2, 3, 4};
    std::vector<float> invstdHostData = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    std::vector<float> meanAllHostData = {4, 0};
    std::vector<float> invstdAllHostData = {4, 0};
    double momentum = 1e-2;
    double eps = 1e-4;

    // Create the input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rMean aclTensor.
    ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rVar aclTensor.
    ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the counts aclTensor.
    ret = CreateAclTensor(countsHostData, countsShape, &countsDeviceAddr, aclDataType::ACL_FLOAT, &counts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the invstd aclTensor.
    ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the meanAll aclTensor.
    ret = CreateAclTensor(meanAllHostData, meanAllShape, &meanAllDeviceAddr, aclDataType::ACL_FLOAT, &meanAll);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the invstdAll aclTensor.
    ret = CreateAclTensor(invstdAllHostData, invstdAllShape, &invstdAllDeviceAddr, aclDataType::ACL_FLOAT, &invstdAll);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNormGatherStatsWithCounts API call example.
    // 3. Call the CANN operator library API, and replace it with the actual API name.
    // Call the first-phase API of aclnnBatchNormGatherStatsWithCounts.
    ret = aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(
        input, mean, invstd, rMean, rVar, momentum, eps, counts, meanAll, invstdAll, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnBatchNormGatherStatsWithCounts.
    ret = aclnnBatchNormGatherStatsWithCounts(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormGatherStatsWithCounts failed. ERROR: %d\n", ret);
              return ret);

    // 4. (Boilerplate) Wait synchronously for the task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the Device memory to the Host. The code needs to be modified based on the actual API definition.
    auto size = GetShapeSize(meanAllShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), meanAllDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. The code needs to be modified based on the actual API definition.
    aclDestroyTensor(input);
    aclDestroyTensor(mean);
    aclDestroyTensor(rMean);
    aclDestroyTensor(rVar);
    aclDestroyTensor(counts);
    aclDestroyTensor(invstd);
    aclDestroyTensor(meanAll);
    aclDestroyTensor(invstdAll);

    // 7. Release Device resources. The code needs to be modified based on the actual API definition.
    aclrtFree(inputDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rMeanDeviceAddr);
    aclrtFree(rVarDeviceAddr);
    aclrtFree(countsDeviceAddr);
    aclrtFree(invstdDeviceAddr);
    aclrtFree(meanAllDeviceAddr);
    aclrtFree(invstdAllDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
