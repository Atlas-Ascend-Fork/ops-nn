# aclnnEmbeddingDenseBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:16:39.726Z -->

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

Implements the backward computation of [aclnnEmbedding](../../../index/embedding/docs/aclnnEmbedding.md). Accumulates the rows of grad corresponding to the same `indices` into out.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnEmbeddingDenseBackwardGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the computation process, and then call "aclnnEmbeddingDenseBackward" to execute the computation.

- `aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *indices, uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnEmbeddingDenseBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnEmbeddingDenseBackwardGetWorkspaceSize

- **Parameters:**

  - grad(aclTensor*, computation input): Original gradient of the data, an aclTensor on the device. Supports 2-8 dimensions. The shape after merging all axes except the last one is the same as the merged shape of indices. Supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md), and [data format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="910" id7 -->
    - <term>Atlas training products</term>: Data types support FLOAT16 and FLOAT.
    <!-- end id7 -->
    <!-- npu="950,A3,910b" id8 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The data type supports BFLOAT16, FLOAT16, and FLOAT.
    <!-- end id8 -->

  - indices(aclTensor*, computation input): Index values corresponding to the grad input, an aclTensor on the device. The value range is [0, numWeights). Supports 1-8 dimensions. Supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md), and [data format](../../../docs/en/context/data_format.md) supports ND. Data types support FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, and BOOL.
  - numWeights(uint64_t, computation input): Size of the first axis of the output tensor.
  - paddingIdx(uint64_t, computation input): Fills the paddingIdx-th row of the output tensor with 0. If paddingIdx is negative, no processing is performed.
  - scaleGradByFreq(bool, computation input): Specifies whether to scale the gradient based on word occurrence frequency. If true, the result is scaled by word frequency; if false, no processing is performed.
  - out(aclTensor*, computation output): Output of the gradient summation, an aclTensor on the device. It has 2 dimensions, with the first-axis size of numWeights and the last-axis size the same as that of grad. The data type is the same as that of grad. [Data format](../../../docs/en/context/data_format.md) supports ND only.

    <!-- npu="910" id9 -->
    - <term>Atlas training products</term>: Data types support FLOAT16 and FLOAT.
    <!-- end id9 -->
    <!-- npu="950,A3,910b" id10 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The data type supports BFLOAT16, FLOAT16, and FLOAT.
    <!-- end id10 -->

  - workspaceSize(uint64_t *, output parameter): Returns the size of the workspace to be requested on the device.
  - executor(aclOpExecutor **, output parameter): Returns the op executor that contains the computation flow of the operator.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API validates input parameters. An error is reported in the following cases:
  Returns 161001(ACLNN_ERR_PARAM_NULLPTR): 1. grad, indices, or out is a null pointer.
  Returns 161002(ACLNN_ERR_PARAM_INVALID): 1. The data type and data format of grad, indices, or out are not supported.
                                      2. The dimensions of grad and indices exceed 8
                                      3. The shape of grad and indices does not meet the constraints
                                      4. The shape of out does not match the deduction result
  ```

## aclnnEmbeddingDenseBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnEmbeddingDenseBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="910" id11 -->
- <term>Atlas training products</term>:
  - For scenarios where scale is true, if the last dimension of grad is embeddingDim and its size exceeds the specified range, an error is reported. The valid range is as follows:
    - When indices is int32, the following must be satisfied:

      $$
      embeddingDim < \frac{180192 - countsSize * 4}{36}
      $$

      - When indices is int64, the following must be satisfied:

      $$
      embeddingDim < \frac{180192 - countsSize * 8}{20}
      $$

      - The formula for countsSize is as follows, where coreNum represents the number of AI processor cores:

      $$
      countsSize = numWeights / coreNum + numWeights \% coreNum
      $$

<!-- end id11 -->
<!-- npu="A3,910b" id12 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - When the parameter shape exceeds the following limits, the output cannot guarantee high precision. If deterministic computation is enabled, high performance also cannot be guaranteed.
    - After grad is coalesced into a two-dimensional shape, the first dimension exceeds INT32_MAX (2147483647).
    - numWeights exceeds INT32_MAX (2147483647).
  - When the dimension of indices after concatenation exceeds INT32_INF (2139095040), high performance cannot be guaranteed.

<!-- end id12 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_embedding_dense_backward.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate. Resource initialization.
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. The construction must be customized based on the API.
  uint64_t numWeights = 4;
  uint64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  std::vector<int64_t> gradOutputShape = {2, 3};
  std::vector<int64_t> indicesShape = {2};
  std::vector<int64_t> outShape = {4, 3};
  void* gradOutputDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradOutputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> indicesHostData = {1, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnEmbeddingDenseBackward.
  ret = aclnnEmbeddingDenseBackwardGetWorkspaceSize(gradOutput, indices, numWeights, paddingIdx, scaleGradByFreq, out,
                                                    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingDenseBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnEmbeddingDenseBackward.
  ret = aclnnEmbeddingDenseBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingDenseBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to the host side. Change it based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Change it based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(indices);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code according to the specific API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(indicesDeviceAddr);
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
