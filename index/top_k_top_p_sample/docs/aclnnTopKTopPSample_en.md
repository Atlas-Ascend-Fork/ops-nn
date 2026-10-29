# aclnnTopKTopPSample

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:00:40.635Z -->

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

- Description:
  Performs topK-topP-sample sampling computation based on the input word frequency logits, topK/topP sampling parameters, and random sampling weight distribution q. Outputs the maximum word frequency logitsSelectIdx for each batch, and the word frequency distribution logitsTopKPSelect after topK-topP sampling.

  The operator contains three sampling algorithms that can be enabled individually while keeping the upstream and downstream processing relationships unchanged (from the original input to the final output): TopK sampling, TopP sampling, and exponential sampling (referred to as Sample in this document). They can form eight computation scenarios. As shown in the following table:

  | Computation Scenario | TopK Sampling | TopP Sampling | Exponential Sampling |Note|
  | :-------:| :------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax Sampling|×|×|×|Applies Softmax to the input logits for each batch and takes the maximum result.|
  |TopK Sampling|√|×|×|Takes the top topK[batch] maximum results from the input logits for each batch.|
  |TopP Sampling|×|√|×|Sorts the input logits in descending order for each batch, and samples the first n results whose cumulative sum is greater than or equal to the topP[batch] value.|
  |Sample Sampling|×|×|√|Applies Softmax to the input logits for each batch, divides by q, and takes the maximum result.|
  |TopK-TopP Sampling|√|√|×|Performs TopK sampling first, then TopP sampling on the input logits for each batch, and takes the maximum result.|
  |TopK-Sample Sampling|√|×|√|Performs TopK sampling first, then Sample sampling on the input logits for each batch, and takes the maximum result.|
  |TopP-Sample Sampling|×|√|√|Performs TopP sampling first, then Sample sampling on the input logits for each batch, and takes the maximum result.|
  |TopK-TopP-Sample Sampling|√|√|√|Performs TopK sampling first, then TopP sampling, and finally Sample sampling on the input logits for each batch, and takes the maximum result.|

- Formula:
The input logits is a term frequency table of size [batch, voc_size], where each batch corresponds to an input sequence, and voc_size is the uniform length agreed upon for each batch.<br>
Each row logits[batch][:] in logits executes different compute scenarios based on the corresponding topK[batch], topP[batch], and q[batch, :].<br>
In the following formulas, b and v are used to denote indices along the batch and voc_size dimensions, respectively.

  TopK Sampling

  1. Uses segmented TopK merge sort with segment length v, where the TopK of block {s-1} is used to pre-filter the input of the current block {s}, progressively updating the TopK of a single batch to reduce redundant data and computation.
  2. topK[batch] corresponds to the k value sampled for the current batch. The valid range is 1≤topK[batch]≤min(voc_size[batch], 1024). If top[K] is out of the valid range, the topK sampling phase of the current batch is skipped, and the sorting of the current batch is also skipped; the input logits[batch] is directly passed to the next module.

  * Split the current batch into several sub-segments and compute topKValue[b] in a rolling manner:

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  Where:

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v denotes the preset fixed segment length for rolling topK:

  $$
  v=8 * 1024
  $$

  * Generate the mask to be filtered.

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask[b] = sortedValue[b]<Min(topKValue[b])
  $$

  * Set values below the threshold to -Inf via mask

  $$
  sortedValue[b][v]=
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true} \\
  sortedValue[b][v] & \text{topKMask[b][v]=false} &
  \end{cases}
  $$

  * Use safe\_softmax to convert the topK-filtered logits into a probability distribution along the last axis

  $$
  probsValue[b]=sortedValue[b].safe\_softmax (dim=-1)
  $$

  * Computes the cumulative probability along the last axis (accumulating from the smallest probability).

  $$
  probsSum[b]=probsValue[b].cumsum (dim=-1)
  $$

  TopP Sampling

  * If the preceding TopK sampling already produces sorted output results, computes the cumulative term frequency based on the TopK sampling output and performs truncation sampling according to TopP:

    $$
    topPMask[b] = probsSum[b][*] < topP[b]
    $$

  * If TopK sampling is skipped, first perform softmax on the input logits[b]:

  $$
  logitsValue[b] = logits[b].safe\_softmax(dim=-1)
  $$

  * Attempt to use topKGuess to perform rolling sort on logits and obtain the mask for computing topP:

  $$
  topPValue[b] = {Max(topKGuess)}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topPValue[b]\left \{s-1 \right \}  \cup \left \{ logitsValue[b][v] \ge topKMin[b][s-1] \right \} \right \}
  $$

  * If the following condition is met before accessing the 1e4-th element of logitsValue[b], topKGuess is considered successful:

  $$
  \sum^{topKGuess}(topPValue[b]) \ge topP[b]\\
  topPMask[b][Index(topPValue[b])] = false
  $$

  * If topKGuess fails, perform full sorting and cumsum on the current logitsValue[b], and truncate sampling by topP[b]:

  $$
  sortedLogits[b] = sort(logitsValue[b], descendant) \\
  probsSum[b]=sortedLogits[b].cumsum (dim=-1) \\
  topPMask[b] = (probsSum[b] - sortedLogits[b])>topP[b]
  $$

  * Set the positions to be filtered to -Inf, obtaining sortedValue[b][v]:

    $$
    sortedValue[b][v] = \begin{cases} -Inf& \text{topPMask[b][v]=true}\\sortedValue[b][v]& \text{topPMask[b][v]=false}\end{cases}
    $$

    Take the first topK elements in each row of the filtered sortedValue[b][v], find the original indices of these elements in the input, and consolidate them into `logitsIdx`:

    $$
    logitsIdx[b][v] = Index(sortedValue[b][v] \in logits)
    $$

  Exponential sampling (Sample)

  * If `isNeedLogits=true`, select the sampled result based on `logitsIdx` and output it to `logitsTopKPSelect`:

  $$
  logitsTopKPSelect[b][logitsIdx[b][v]]=sortedValue[b][v]
  $$

  * Perform exponential distribution sampling on `logitsSort`:

    $$
    probs = safe\_softmax(logitsSort)
    $$

    $$
    probsOpt = \frac{probs}{q + eps}
    $$

  * Take the maximum element of each batch from `probsOpt`, and gather the input index of the corresponding element from `logitsIdx` as the output `logitsSelectIdx`:

    $$
    logitsSelectIdx[b] = logitsIdx[b][argmax(probsOpt[b][:])]
    $$

  Where 0 ≤ b < sortedValue.size(0) and 0 ≤ v < sortedValue.size(1).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnTopKTopPSampleGetWorkspaceSize` API first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the `aclnnTopKTopPSample` API to perform the computation.

```Cpp
aclnnStatus aclnnTopKTopPSampleGetWorkspaceSize(
  const aclTensor *logits,
  const aclTensor *topK,
  const aclTensor *topP,
  const aclTensor *q,
  double           eps,
  bool             isNeedLogits,
  int64_t          topKGuess,
  const aclTensor *logitsSelectIdx,
  const aclTensor *logitsTopKPSelect,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)

```

```Cpp
aclnnStatus aclnnTopKTopPSample(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)

```

## aclnnTopKTopPSampleGetWorkspaceSize

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
      <col style="width: 146px">
      <col style="width: 120px">
      <col style="width: 271px">
      <col style="width: 392px">
      <col style="width: 228px">
      <col style="width: 101px">
      <col style="width: 100px">
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
        <td>logits</td>
        <td>Input</td>
        <td>Indicates the input word frequency to be sampled. The word frequency index is fixed to the last dimension, corresponding to `logits` in the formula.</td>
        <td><ul><li>Does not support empty tensors.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>Input</td>
        <td>Indicates the k value sampled per each batch. Corresponding to `topK[b]` in the formula.</td>
        <td><ul><li>Does not support empty tensor.</li><li>The shape needs to be consistent with the first n-1 dimensions of `logits`.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topP</td>
        <td>Input</td>
        <td>Indicates the p value sampled for each batch. Corresponds to `topP[b]` in the formula.</td>
        <td><ul><li>Does not support empty tensor.</li><li>The shape needs to be consistent with the first n-1 dimensions of `logits`, and the data type needs to be consistent with `logits`.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>q</td>
        <td>Input</td>
        <td>Indicates the exponential sampling matrix output by topK-topP sampling. Corresponds to `q` in the corresponding formula.</td>
        <td><ul><li>Does not support empty tensors.</li><li>The shape needs to be consistent with `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Input</td>
        <td>Prevents division by zero in softmax and weighted sampling. It is recommended to set it to 1e-8.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedLogits</td>
        <td>Input</td>
        <td>Indicates the output condition for controlling logitsTopKPselect. It is recommended to set the value to 0.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topKGuess</td>
        <td>Input</td>
        <td>Indicates each batch's candidate logits size when attempting to traverse and sample logits for the topP part. The value must be a positive integer.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsSelectIdx</td>
        <td>Output</td>
        <td>Indicates the position index in the input logits of max(probsOpt[batch, :]) (the element with the maximum word frequency in each batch) after the topK-topP-sample computation flow.</td>
        <td><ul><li>Does not support empty Tensor.</li><li>The shape needs to be consistent with the first n-1 dimensions of `logits`.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logitsTopKPSelect</td>
        <td>Output</td>
        <td>Indicates the remaining unfiltered logits in the input logits after the topK-topP computation process.</td>
        <td><ul><li>Does not support empty tensors.</li><li>The shape needs to be consistent with the first n-1 dimensions of `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied on the Device side.</td>
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
      </tbody>
      </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
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
      <td>Any of the input parameters logits, topK, or topP is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of logits, topK, topP, or q is not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimensions or sizes of logits and q are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimensions of topK and topP are inconsistent with the first n-1 dimensions of logits.</td>
    </tr>
    <tr>
      <td>The data types of logits and topP are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnTopKTopPSample

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnTopKTopPSampleGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnTopKTopPSample defaults to a deterministic implementation.
- For logits, ensure that its elements are no less than the minimum valid value of FLOAT32 after the $exp(*)$ computation; otherwise, the corresponding Safe_softmax result will be 0, which may lead to unexpected output of the operator.
- For all sampling parameters, their dimensions must satisfy batch > 0 and 0 < vocSize <= 2^20.
- topK and topP accept only non-negative values as valid input. Passing 0 or negative values does not skip the sampling of the corresponding batch; instead, it causes unexpected errors.
- The sizes and dimensions of logits, q, and logitsTopKPselect must be exactly the same.
- All dimensions of logits, topK, topP, and logitsSelectIdx except the last dimension must be in the same order and of the same size. Currently, logits can only be 2-dimensional, and topK, topP, and logitsSelectIdx must be 1-dimensional non-null tensors. Null tensors are not allowed as input for logits, topK, and topP. To skip the corresponding module, set the input according to the corresponding rules.
- To skip the topK module individually, pass a tensor of size [batch, 1] with each element set to an invalid value.
- If 1024 < topK[batch] < vocSize[batch], all valid elements of the current batch are selected and topK sampling is skipped.
- To skip the topP module individually, pass a tensor of size [batch, 1] with each element ≥ 1.
- To skip the sample module, pass `q=nullptr`. To use the sample module, pass a tensor of size [batch, vocSize].

## Example

The following is sample code for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_top_k_top_p_sample.h"

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
    // Boilerplate: resource initialization.
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
    // Call aclrtMalloc to apply for device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device.
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
      // 1. (Boilerplate) Initialize the device/stream by referring to the acl API.
      // Fill in deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Perform custom construction based on the API.
      std::vector<int64_t> logitsShape = {48, 131072};
      std::vector<int64_t> topKPShape = {48};
      long long vocShapeSize = GetShapeSize(logitsShape);
      long long batchShapeSize = GetShapeSize(topKPShape);

      void* logitsDeviceAddr = nullptr;
      void* topKDeviceAddr = nullptr;
      void* topPDeviceAddr = nullptr;
      void* qDeviceAddr = nullptr;
      void* logitsSelectedIdxDeviceAddr = nullptr;
      void* logitsTopKPSelectDeviceAddr = nullptr;

      aclTensor* logits = nullptr;
      aclTensor* topK = nullptr;
      aclTensor* topP = nullptr;
      aclTensor* q = nullptr;
      aclTensor* logitsSelectIdx = nullptr;
      aclTensor* logitsTopKPSelect = nullptr;
      std::vector<int16_t> logitsHostData(48 * 131072, 1);
      std::vector<int32_t> topKHostData(48, 128);
      std::vector<int16_t> topPHostData(48, 1);
      std::vector<float> qHostData(48 * 131072, 1.0f);

      std::vector<int64_t> logitsSelectedIdxHostData(48, 0);
      std::vector<float> logitsTopKPSelectHostData(48 * 131072, 0);

      float eps=1e-8;
      int64_t isNeedLogits=0;
      int32_t topKGuess=32;
      // Create the logits aclTensor.
      ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_BF16, &logits);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the topK aclTensor.
      ret = CreateAclTensor(topKHostData, topKPShape, &topKDeviceAddr, aclDataType::ACL_INT32, &topK);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the topPaclTensor.
      ret = CreateAclTensor(topPHostData, topKPShape, &topPDeviceAddr, aclDataType::ACL_BF16, &topP);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the q aclTensor.
      ret = CreateAclTensor(qHostData, logitsShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the logitsSelected aclTensor.
      ret = CreateAclTensor(logitsSelectedIdxHostData, topKPShape, &logitsSelectedIdxDeviceAddr, aclDataType::ACL_INT64, &logitsSelectIdx);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the logitsTopKPSelect aclTensor.
      ret = CreateAclTensor(logitsTopKPSelectHostData, logitsShape, &logitsTopKPSelectDeviceAddr, aclDataType::ACL_FLOAT, &logitsTopKPSelect);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, and replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call the first-phase API of aclnnTopKTopPSample.
      ret = aclnnTopKTopPSampleGetWorkspaceSize(logits, topK, topP, q, eps, isNeedLogits, topKGuess, logitsSelectIdx, logitsTopKPSelect, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnTopKTopPSample.
      ret = aclnnTopKTopPSample(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSample failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait synchronously for the task to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output values, copy the results from device memory to the host, and modify the code based on the interface definition of the specific API.
      auto size = GetShapeSize(topKPShape);
      std::vector<int64_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), logitsSelectedIdxDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %lld\n", i, static_cast<long long>(resultData[i]));
      }

      // 6. Release aclTensor. Modify it according to the specific API interface definition.
      aclDestroyTensor(logits);
      aclDestroyTensor(topK);
      aclDestroyTensor(topP);
      aclDestroyTensor(q);
      aclDestroyTensor(logitsSelectIdx);
      aclDestroyTensor(logitsTopKPSelect);
      // 7. Release Device resources. Modify it according to the specific API interface definition.
      aclrtFree(logitsDeviceAddr);
      aclrtFree(topKDeviceAddr);
      aclrtFree(topPDeviceAddr);
      aclrtFree(qDeviceAddr);
      aclrtFree(logitsSelectedIdxDeviceAddr);
      aclrtFree(logitsTopKPSelectDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
      }
  ```
