# aclnnCrossEntropySumExpAndIndexLogit

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:15:40.571Z -->

## Applicable Products

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|    ×     |
|<term>Atlas A2 products</term>|    ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- **Description**: This operator is a fusion operator for the local CrossEntropy computation segment in the Tensor Parallel use case. For the local logits of each TP rank after splitting along the vocab dimension, the operator completes in one pass, after `all_reduce(MAX)` and before the `all_reduce(SUM)` of `predictedLogits` / `sumExpLogits`, the following steps: logits shifting (subtracting the global maximum), out-of-range target mask determination, local target offset computation, gather of the logit corresponding to the target, and `exp` and local summation along the vocab dimension. This reduces the overhead of small operator launches, GM reads and writes, and intermediate tensor materialization. It targets the training of large models with ultra-large vocabularies.

- **Formula**:

  Let `vocabParallelLogits` be the local vocab shard of the current rank, `globalLogitsMax` be the global maximum logit, and `[vocabStartIndex, vocabEndIndex)` be the vocab shard range. For each token `i` and each local vocab position `j`:

  **1. target mask:**

  $$
  targetMask[i] = \begin{cases} 1, & target[i] < vocabStartIndex \text{ or } target[i] \geq vocabEndIndex \\ 0, & \text{otherwise} \end{cases}
  $$

  **2. Local target offset:**

  $$
  targetOffset[i] = \begin{cases} 0, & targetMask[i] = 1 \\ target[i] - vocabStartIndex, & targetMask[i] = 0 \end{cases}
  $$

  **3. predicted logit gather (logits shifting):**

  $$
  predictedLogits[i] = \begin{cases} 0, & targetMask[i] = 1 \\ vocabParallelLogits[i, targetOffset[i]] - globalLogitsMax[i], & targetMask[i] = 0 \end{cases}
  $$

  **4. Exponential computation:**

  $$
  expLogits[i, j] = \exp\big(vocabParallelLogits[i, j] - globalLogitsMax[i]\big)
  $$

  **5. Local sum:**

  $$
  sumExpLogits[i] = \sum_{j=0}^{V\_local-1} expLogits[i, j]
  $$

- **Special handling**: Intermediate computation uses FLOAT (the `BFLOAT16` input is automatically converted to higher precision). The exp operand has the global maximum subtracted to suppress overflow, and underflow naturally returns 0 as FLOAT. This operator does not involve division, so divide-by-zero protection is not required. The accumulation of `sumExpLogits` along the V_local dimension is completed within a single core (no cross-core reduction and no AtomicAdd). The accumulation order is fixed. Multiple calls with the same input return the same result. The operator defaults to a deterministic implementation.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the `aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize` API to obtain the workspace size required for computation and the executor that contains the computation process of the operator. Then call the `aclnnCrossEntropySumExpAndIndexLogit` API to perform the computation.

```cpp
aclnnStatus aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize(
  const aclTensor *vocabParallelLogits,
  const aclTensor *target,
  const aclTensor *globalLogitsMax,
  int64_t          vocabStartIndex,
  int64_t          vocabEndIndex,
  const aclTensor *predictedLogitsOut,
  const aclTensor *sumExpLogitsOut,
  const aclTensor *expLogitsOut,
  const aclTensor *targetOffsetOut,
  const aclTensor *targetMaskOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnCrossEntropySumExpAndIndexLogit(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 160px">
  <col style="width: 100px">
  <col style="width: 260px">
  <col style="width: 260px">
  <col style="width: 140px">
  <col style="width: 80px">
  <col style="width: 280px">
  <col style="width: 100px">
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
      <td>vocabParallelLogits (aclTensor*)</td>
      <td>Input</td>
      <td>Local vocab shard logits of the current TP rank, corresponding to vocabParallelLogits in the formula.</td>
      <td>Does not support empty tensor. Values do not support ±inf and nan.</td>
      <td>FLOAT, BFLOAT16</td>
      <td>ND</td>
      <td><ul><li>Supports only 2D [N, V_local] or 3D [S, B, V_local].</li><li>shape[:-1] must be exactly the same as target.shape.</li><li>Range of the last dimension V_local: [16, 200K]; must be a multiple of 16 for BFLOAT16 and a multiple of 8 for FLOAT.</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>target (aclTensor*)</td>
      <td>Input</td>
      <td>Global vocab index, corresponding to target in the formula.</td>
      <td>Does not support empty Tensor. The value is a non-negative integer.</td>
      <td>INT32</td>
      <td>ND</td>
      <td><ul><li>The shape is exactly the same as vocabParallelLogits.shape[:-1].</li><li>After flattening, N = prod(shape), range: [1, 32K].</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>globalLogitsMax (aclTensor*)</td>
      <td>Input</td>
      <td>The global maximum logit obtained after all_reduce(MAX), which is globalLogitsMax in the formula.</td>
      <td>Does not support empty Tensor. The data type must be the same as that of vocabParallelLogits. The value must satisfy globalLogitsMax[i] >= max_j(vocabParallelLogits[i, j]) (that is, greater than or equal to all logits of the same token on the current rank).</td>
      <td>Same as vocabParallelLogits</td>
      <td>ND</td>
      <td>The shape is exactly the same as target.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vocabStartIndex (int64_t)</td>
      <td>Input</td>
      <td>Start index of the vocab shard on the current rank (global). It is vocabStartIndex in the formula.</td>
      <td>vocabEndIndex must be greater than vocabStartIndex, and vocabStartIndex must be greater than or equal to 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocabEndIndex (int64_t)</td>
      <td>Input</td>
      <td>End index of the vocab shard on the current rank (global), which is vocabEndIndex in the formula.</td>
      <td>vocabEndIndex - vocabStartIndex == vocabParallelLogits.size(-1) must be satisfied.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>predictedLogits (aclTensor*)</td>
      <td>Output</td>
      <td>The result of subtracting global_max from the logit corresponding to target, predictedLogits in the formula.</td>
      <td>Same shape as target; positions corresponding to targets not in the current rank shard are set to 0.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Same as target.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sumExpLogits (aclTensor*)</td>
      <td>Output</td>
      <td>The sum of local exp(logits - global_max) along the last dimension, sumExpLogits in the formula.</td>
      <td>Same shape as target.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Same as target.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expLogits (aclTensor*)</td>
      <td>Output</td>
      <td>exp(vocabParallelLogits - globalLogitsMax), the expLogits in the formula.</td>
      <td>Same shape as vocabParallelLogits.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Same as vocabParallelLogits.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>targetOffset (aclTensor*)</td>
      <td>Output</td>
      <td>target - vocabStartIndex, which is targetOffset in the formula.</td>
      <td>Shape is the same as target. Set to 0 if target is not in the current rank shard.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>Same as target.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>targetMask (aclTensor*)</td>
      <td>Output</td>
      <td>Vocab out-of-bounds mask. The value 1 indicates that the target is not in the current rank shard, and the value 0 indicates that it is within. This is the targetMask in the formula.</td>
      <td>Same shape as target.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>Same as target.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 263px">
  <col style="width: 88px">
  <col style="width: 619px">
  </colgroup>
  <thead>
  <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>vocabParallelLogits, target, globalLogitsMax, predictedLogits, sumExpLogits, expLogits, targetOffset, or targetMask is a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The dimension of vocabParallelLogits is not 2 or 3.</td>
  </tr>
  <tr>
      <td>vocabParallelLogits.shape[:-1] is inconsistent with target.shape.</td>
  </tr>
  <tr>
      <td>globalLogitsMax.shape is inconsistent with target.shape.</td>
  </tr>
  <tr>
      <td>vocabEndIndex &lt;= vocabStartIndex.</td>
  </tr>
  <tr>
      <td>vocabEndIndex - vocabStartIndex != vocabParallelLogits.size(-1).</td>
  </tr>
  <tr>
      <td>The data type of vocabParallelLogits or globalLogitsMax is not in the supported range (FLOAT / BFLOAT16), or the two data types are inconsistent.</td>
  </tr>
  <tr>
      <td>The data type of target is not INT32.</td>
  </tr>
  <tr>
      <td>V_local does not meet the alignment constraint (not a multiple of 16 for BFLOAT16 or a multiple of 8 for FLOAT).</td>
  </tr>
  <tr>
      <td>N is not in the range [1, 32K] or V_local is not in the range [16, 200K].</td>
  </tr>
  </tbody>
  </table>

## aclnnCrossEntropySumExpAndIndexLogit

- **Parameters:**

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
      <td>Memory address of the workspace applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic Computation**:
  aclnnCrossEntropySumExpAndIndexLogit defaults to a deterministic implementation.

- **Batch Consistency Instruction**:
  aclnnCrossEntropySumExpAndIndexLogit defaults to a batch consistency implementation.

- **Input shape restrictions**:
    - vocabParallelLogits supports only 2D [N, V_local] or 3D [S, B, V_local].
    - vocabParallelLogits.shape[:-1] must be exactly the same as target.shape.
    - globalLogitsMax.shape must be exactly the same as target.shape.
    - V_local alignment constraint: V_local must be a multiple of 16 for BFLOAT16 input and a multiple of 8 for FLOAT input (to ensure 32-byte UB alignment).
    - N (prod(target.shape)) range: [1, 32K].
    - V_local range: [16, 200K].

## Example

The following sample code is for reference only. For details about compiling and running, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_sum_exp_and_index_logit.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
        *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

void PrintOutIntResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
        *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device side.
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
        strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. Boilerplate, device/stream initialization, refer to the acl API.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output according to the API definition.
    // The sample uses a 2D shape: N=4, V_local=16 (V_local must be a multiple of 8 for FLOAT).
    int64_t N = 4;
    int64_t vLocal = 16;
    int64_t vocabStart = 0;
    int64_t vocabEnd = vLocal;   // All targets fall within [0, 16), and targetMask is all 0.

    std::vector<int64_t> logitsShape = {N, vLocal};
    std::vector<int64_t> targetShape = {N};

    void* logitsDeviceAddr = nullptr;
    void* targetDeviceAddr = nullptr;
    void* maxDeviceAddr = nullptr;
    void* predictedDeviceAddr = nullptr;
    void* sumExpDeviceAddr = nullptr;
    void* expLogitsDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* maskDeviceAddr = nullptr;

    aclTensor* logitsTensor = nullptr;
    aclTensor* targetTensor = nullptr;
    aclTensor* maxTensor = nullptr;
    aclTensor* predictedTensor = nullptr;
    aclTensor* sumExpTensor = nullptr;
    aclTensor* expLogitsTensor = nullptr;
    aclTensor* offsetTensor = nullptr;
    aclTensor* maskTensor = nullptr;

    // Construct the sample input data.
    std::vector<float> logitsHostData(N * vLocal, 0);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < vLocal; j++) {
            logitsHostData[i * vLocal + j] = static_cast<float>(i) * 1.0f + static_cast<float>(j) * 0.1f;
        }
    }
    std::vector<int32_t> targetHostData = {2, 5, 10, 15};
    std::vector<float> maxHostData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> predictedHostData(N, 0);
    std::vector<float> sumExpHostData(N, 0);
    std::vector<float> expLogitsHostData(N * vLocal, 0);
    std::vector<int32_t> offsetHostData(N, 0);
    std::vector<int32_t> maskHostData(N, 0);

    ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr,
        aclDataType::ACL_FLOAT, &logitsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr,
        aclDataType::ACL_INT32, &targetTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxHostData, targetShape, &maxDeviceAddr,
        aclDataType::ACL_FLOAT, &maxTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(predictedHostData, targetShape, &predictedDeviceAddr,
        aclDataType::ACL_FLOAT, &predictedTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sumExpHostData, targetShape, &sumExpDeviceAddr,
        aclDataType::ACL_FLOAT, &sumExpTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expLogitsHostData, logitsShape, &expLogitsDeviceAddr,
        aclDataType::ACL_FLOAT, &expLogitsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, targetShape, &offsetDeviceAddr,
        aclDataType::ACL_INT32, &offsetTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskHostData, targetShape, &maskDeviceAddr,
        aclDataType::ACL_INT32, &maskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API.
    ret = aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize(
        logitsTensor,
        targetTensor,
        maxTensor,
        vocabStart,
        vocabEnd,
        predictedTensor,
        sumExpTensor,
        expLogitsTensor,
        offsetTensor,
        maskTensor,
        &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API.
    ret = aclnnCrossEntropySumExpAndIndexLogit(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnCrossEntropySumExpAndIndexLogit failed. ERROR: %d\n", ret); return ret);

    // 4. Boilerplate. Wait synchronously for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value.
    LOG_PRINT("predictedLogits:\n");
    PrintOutResult(targetShape, &predictedDeviceAddr);
    LOG_PRINT("sumExpLogits:\n");
    PrintOutResult(targetShape, &sumExpDeviceAddr);
    LOG_PRINT("expLogits:\n");
    PrintOutResult(logitsShape, &expLogitsDeviceAddr);
    LOG_PRINT("targetOffset:\n");
    PrintOutIntResult(targetShape, &offsetDeviceAddr);
    LOG_PRINT("targetMask:\n");
    PrintOutIntResult(targetShape, &maskDeviceAddr);

    // 6. Release the aclTensor.
    aclDestroyTensor(logitsTensor);
    aclDestroyTensor(targetTensor);
    aclDestroyTensor(maxTensor);
    aclDestroyTensor(predictedTensor);
    aclDestroyTensor(sumExpTensor);
    aclDestroyTensor(expLogitsTensor);
    aclDestroyTensor(offsetTensor);
    aclDestroyTensor(maskTensor);

    // 7. Release the device resources.
    aclrtFree(logitsDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(maxDeviceAddr);
    aclrtFree(predictedDeviceAddr);
    aclrtFree(sumExpDeviceAddr);
    aclrtFree(expLogitsDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(maskDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
