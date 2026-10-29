# aclnnEmbeddingBag

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:12:40.622Z -->

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

- Description: Obtains a group of aggregated values from weight based on indices, and then performs max, sum, or mean aggregation on the obtained values according to the offset of offsets and the aggregation mode specified by mode. The remaining parameters further refine the control of the computation process.
- Shape deduction formula:
  Assume that the shape of the input weight is (numWeight, embeddingDim), the shape of indices is (bagIndices), and the shape of offsets is (bagOffsets).

  <!-- npu="A3,910b,910,310p" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>:

    - When mode is set to sum:

    ```bash
    The shape of output is includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    The shape of offset2bag is (bagIndices,)
    The shape of bagSize is includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    The shape of maxIndices is includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    ```

    - When mode is set to mean:

    ```bash
    The shape of output is includeLastOffset? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    The shape of offset2bag is (bagIndices,)
    The shape of bagSize is includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    The shape of maxIndices is includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    ```

    - When mode is set to max:

    ```bash
    The shape of output is includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    The shape of offset2bag is (bagIndices,)
    The shape of bagSize is includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    The shape of maxIndices is includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    ```

  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:

    - When mode is set to sum:

    ```bash
    The shape of output is includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    The shape of offset2bag is (bagIndices,)
    The shape of bagSize is (bagOffsets,)
    The shape of maxIndices is (0,)
    ```

    - When mode is set to mean:

    ```bash
    The shape of output is includeLastOffset? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    The shape of offset2bag is (bagIndices,)
    The shape of bagSize is bagOffsets
    The shape of maxIndices is (0,)
    ```

    - When mode is set to max:

    ```bash
    The shape of output is includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    The shape of offset2bag is (bagIndices,)
    The shape of bagSize is (bagOffsets,)
    The shape of maxIndices is includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    ```

  <!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnEmbeddingBagGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnEmbeddingBag" API to execute the computation.

```Cpp
aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(
 const aclTensor* weight,
 const aclTensor* indices,
 const aclTensor* offsets,
 bool             scaleGradByFreq,
 int64_t          mode,
 bool             sparse,
 const aclTensor* perSampleWeights,
 bool             includeLastOffset,
 int64_t          paddingIdx,
 aclTensor*       output,
 aclTensor*       offset2bag,
 aclTensor*       bagSize,
 aclTensor*       maxIndices,
 uint64_t*        workspaceSize,
 aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnEmbeddingBag(
 void*            workspace,
 uint64_t         workspaceSize,
 aclOpExecutor*   executor,
 aclrtStream      stream)
```

## aclnnEmbeddingBagGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1391px"><colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 274px">
    <col style="width: 172px">
    <col style="width: 116px">
    <col style="width: 114px">
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
        <td>weight</td>
        <td>Input</td>
        <td>Word embedding matrix, containing the embedding vectors of all words.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>Tensor containing the indices, which specify the words whose embedding vectors are extracted from weight.</td>
        <td>-</td>
        <td>INT32, INT64, INT16, INT8, UINT8</td>
        <td>ND</td>
        <td>0-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>offsets</td>
        <td>Input</td>
        <td>Tensor of offsets used to split indices into multiple bags.</td>
        <td>Elements must be sorted in ascending order.</td>
        <td>INT32, INT64, INT16, INT8, UINT8</td>
        <td>-</td>
        <td>0-1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleGradByFreq</td>
        <td>Input</td>
        <td>Controls whether to scale gradients based on word frequency.</td>
        <td>When scaleGradByFreq is true, gradients are scaled based on word frequency. When scaleGradByFreq is false, they are not.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>mode</td>
        <td>Input</td>
        <td>Controls the aggregation mode.</td>
        <td>0 indicates the sum aggregation mode, 1 indicates the mean aggregation mode, and other values indicate the max aggregation mode.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparse</td>
        <td>Input</td>
        <td>Controls the sparse mode.</td>
        <td><ul><li>If false, indicates that weight is not a sparse matrix;</li><li>If true, indicates that weight is a sparse matrix.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>perSampleWeights</td>
        <td>Input</td>
        <td>Specifies sample weights.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>-</td>
        <td>0-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>includeLastOffset</td>
        <td>Input</td>
        <td>Controls whether to include the last offset.</td>
        <td><ul><li>If false, indicates that the last offset is not included;</li><li>If true, indicates that the last offset is included.</li><li>When indices is 2-dimensional, includeLastOffset supports only false.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>paddingIdx</td>
        <td>Input</td>
        <td>Controls the indices excluded from computation.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>Output</td>
        <td>Output</td>
        <td>Aggregation result of the word embedding matrix.</td>
        <td>-</td>
        <td>Same as weight</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>offset2bag</td>
        <td>Output</td>
        <td>Start offset of each bag.</td>
        <td>The data type of offset2bag is consistent with the higher-precision one between indices and offsets.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bagSize</td>
        <td>Output</td>
        <td>Size of each bag.</td>
        <td>The data type of offset2bag is consistent with the higher precision one between indices and offsets.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxIndices</td>
        <td>Output</td>
        <td>The row containing the maximum value of the word embedding vector when mode is max.</td>
        <td>The data type of offset2bag is consistent with the higher precision one between indices and offsets.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="A3,910b,910,310p" id9 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: indices supports only 0 or 1 dimensions, and perSampleWeights supports only 1 dimension and must be nullptr in modes other than sum mode.
  - At least one of the data types of indices and offsets is INT32 or INT64; if the data type is INT16, INT8, or UINT8, the API internally converts it to INT32 for computation.

  <!-- end id9 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>The input weight, indices, offsets, output, offset2bag, bagSize, or maxIndices is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of weight is not in the supported range, or weight is not 2-dimensional.</td>
      </tr>
      <tr>
      <td>The data type of indices is not in the supported range, or indices is not 1-dimensional.</td>
      </tr>
      <tr>
      <td>The data type of offsets is not in the supported range, or offsets is not 1-dimensional.</td>
      </tr>
      <tr>
      <td>Neither the data type of indices nor that of offsets is INT32 or INT64.</td>
      </tr>
      <tr>
      <td>When perSampleWeights is not nullptr, its data type is inconsistent with that of weight, perSampleWeights is not 1-dimensional, the number of elements in perSampleWeights is not equal to that of indices, or perSampleWeights is not nullptr in non-sum mode.</td>
      </tr>
      <tr>
      <td>The data type of output is inconsistent with that of weight, or the shape does not match the definition.</td>
      </tr>
      <tr>
      <td>The data types and shapes of offset2bag, bagSize, and maxIndices do not match the deduced data types and shapes.</td>
      </tr>
    </tbody>
    </table>

## aclnnEmbeddingBag

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnEmbeddingBagGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which includes the operator computation process.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Indicates the Stream for task execution.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnEmbeddingBag defaults to a deterministic implementation.

`sparse` and `scaleGradByFreq` only support input `False`.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_embedding_bag.h"

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
  // boilerplate, resource initialization
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> weightShape = {3, 3};
  std::vector<int64_t> indicesShape = {6};
  std::vector<int64_t> offsetsShape = {4};
  std::vector<int64_t> perSampleWeightsShape = {6};
  std::vector<int64_t> outputShape = {4, 3};
  std::vector<int64_t> offset2bagShape = {6};
  std::vector<int64_t> bagSizeShape = {4};
  std::vector<int64_t> maxIndicesShape = {4};

  void* weightDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* offsetsDeviceAddr = nullptr;
  void* perSampleWeightsDeviceAddr = nullptr;
  void* outputDeviceAddr = nullptr;
  void* offset2bagDeviceAddr = nullptr;
  void* bagSizeDeviceAddr = nullptr;
  void* maxIndicesDeviceAddr = nullptr;

  aclTensor* weight = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* offsets = nullptr;
  aclTensor* perSampleWeights = nullptr;
  aclTensor* output = nullptr;
  aclTensor* offset2bag = nullptr;
  aclTensor* bagSize = nullptr;
  aclTensor* maxIndices = nullptr;

  std::vector<float> weightHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> indicesHostData = {1, 2, 0, 2, 2, 1};
  std::vector<int64_t> offsetsHostData = {0, 2, 4, 5};
  std::vector<float> perSampleWeightsHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> outputHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> offset2bagHostData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> bagSizeHostData = {0, 0, 0, 0};
  std::vector<int64_t> maxIndicesHostData = {0, 0, 0, 0};

  //Create the weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //Create the offsets aclTensor.
  ret = CreateAclTensor(offsetsHostData, offsetsShape, &offsetsDeviceAddr, aclDataType::ACL_INT64, &offsets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the perSampleWeights aclTensor
  ret = CreateAclTensor(perSampleWeightsHostData, perSampleWeightsShape, &perSampleWeightsDeviceAddr, aclDataType::ACL_FLOAT, &perSampleWeights);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the output aclTensor
  ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the offset2bag aclTensor
  ret = CreateAclTensor(offset2bagHostData, offset2bagShape, &offset2bagDeviceAddr, aclDataType::ACL_INT64, &offset2bag);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the bagSize aclTensor
  ret = CreateAclTensor(bagSizeHostData, bagSizeShape, &bagSizeDeviceAddr, aclDataType::ACL_INT64, &bagSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the maxIndices aclTensor
  ret = CreateAclTensor(maxIndicesHostData, maxIndicesShape, &maxIndicesDeviceAddr, aclDataType::ACL_INT64, &maxIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Non-tensor parameters
  bool scaleGradByFreq = false;
  int64_t mode = 0;
  bool sparse = false;
  bool includeLastOffset = false;
  int64_t paddingIdx = 1;

  // 3. Call the CANN operator library API. Change the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnEmbeddingBag.
  ret = aclnnEmbeddingBagGetWorkspaceSize(weight, indices, offsets, scaleGradByFreq, mode, sparse, perSampleWeights,
            includeLastOffset, paddingIdx, output, offset2bag, bagSize, maxIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingBagGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnEmbeddingBag.
  ret = aclnnEmbeddingBag(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingBag failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code according to the specific API definition.
  auto outputSize = GetShapeSize(outputShape);
  std::vector<float> outputResultData(outputSize, 0);
  ret = aclrtMemcpy(outputResultData.data(), outputResultData.size() * sizeof(outputResultData[0]), outputDeviceAddr,
                    outputSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < outputSize; i++) {
    LOG_PRINT("outputResult[%ld] is: %f\n", i, outputResultData[i]);
  }

  auto offset2bagSize = GetShapeSize(offset2bagShape);
  std::vector<int64_t> offset2bagResultData(offset2bagSize, 0);
  ret = aclrtMemcpy(offset2bagResultData.data(), offset2bagResultData.size() * sizeof(offset2bagResultData[0]), offset2bagDeviceAddr,
                    offset2bagSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < offset2bagSize; i++) {
    LOG_PRINT("offset2bagResult[%ld] is: %ld\n", i, offset2bagResultData[i]);
  }

  auto bagSizeSize = GetShapeSize(bagSizeShape);
  std::vector<int64_t> bagSizeResultData(bagSizeSize, 0);
  ret = aclrtMemcpy(bagSizeResultData.data(), bagSizeResultData.size() * sizeof(bagSizeResultData[0]), bagSizeDeviceAddr,
                    bagSizeSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < bagSizeSize; i++) {
    LOG_PRINT("bagSizeResult[%ld] is: %ld\n", i, bagSizeResultData[i]);
  }

  auto maxIndicesSize = GetShapeSize(maxIndicesShape);
  std::vector<int64_t> maxIndicesResultData(maxIndicesSize, 0);
  ret = aclrtMemcpy(maxIndicesResultData.data(), maxIndicesResultData.size() * sizeof(maxIndicesResultData[0]), maxIndicesDeviceAddr,
                    maxIndicesSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < maxIndicesSize; i++) {
    LOG_PRINT("maxIndicesResult[%ld] is: %ld\n", i, maxIndicesResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
  aclDestroyTensor(weight);
  aclDestroyTensor(indices);
  aclDestroyTensor(offsets);
  aclDestroyTensor(perSampleWeights);
  aclDestroyTensor(output);
  aclDestroyTensor(offset2bag);
  aclDestroyTensor(bagSize);
  aclDestroyTensor(maxIndices);

  // 7. Release device resources. Modify the code according to the specific API definition.
  aclrtFree(weightDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(offsetsDeviceAddr);
  aclrtFree(perSampleWeightsDeviceAddr);
  aclrtFree(outputDeviceAddr);
  aclrtFree(offset2bagDeviceAddr);
  aclrtFree(bagSizeDeviceAddr);
  aclrtFree(maxIndicesDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
