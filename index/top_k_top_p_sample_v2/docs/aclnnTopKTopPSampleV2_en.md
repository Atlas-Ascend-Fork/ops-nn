# aclnnTopKTopPSampleV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:02:20.357Z -->

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

- Description:
  Performs topK-topP-minP-sample sampling computation based on the input word frequency logits, topK/topP/minP sampling parameters, and random sampling weight distribution q. When the input isNeedSampleResult is false, outputs the maximum word frequency logitsSelectIdx for each batch, and the word frequency distribution logitsTopKPSelect after topK-topP-minP sampling. When the input isNeedSampleResult is true, outputs the intermediate computation results logitsIdx and logitsSortMasked after topK-topP-minP sampling, where logitsSortMasked is the intermediate result of the word frequency logits after topK-topP-minP sampling computation, and logitsIdx is the corresponding index of logitsSortMasked in logits.

  The operator contains four sampling algorithms that can be enabled individually while the upstream/downstream processing relationships remain unchanged (from the raw input to the final output): TopK sampling, TopP sampling, MinP sampling, and exponential distribution sampling (referred to as Sample in this document). The following computation scenarios are currently supported, as shown in the table below:

  | Computation Scenario | TopK Sampling | TopP Sampling | MinP Sampling | Exponential Sampling | Output Intermediate Results |Note|
  | :-------:| :------:|:-------:|:-------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax Sampling|×|×|×|×|×|Applies Softmax to the input logits for each batch and takes the maximum result.|
  |TopK Sampling|√|×|×|×|×|Takes the top topK[batch] maximum results from the input logits for each batch.|
  |TopP Sampling|×|√|×|×|×|Sorts the input logits in descending order for each batch, and samples the first n results whose cumulative sum is greater than or equal to the topP[batch] value.|
  |Sample Sampling|×|×|×|√|×|Applies Softmax to the input logits for each batch, divides by q, and takes the maximum result.|
  |TopK-TopP Sampling|√|√|×|×|×|Performs TopK sampling first, then TopP sampling on the input logits for each batch, and takes the maximum result.|
  |TopK-Sample Sampling|√|×|×|√|×|Performs TopK sampling first, then Sample sampling on the input logits for each batch, and takes the maximum result.|
  |TopP-Sample Sampling|×|√|×|√|×|Performs TopP sampling first, then Sample sampling on the input logits for each batch, and takes the maximum result.|
  |TopK-TopP-Sample Sampling|√|√|×|√|×|Performs TopK sampling first, then TopP sampling, and finally Sample sampling on the input logits for each batch, and takes the maximum result.|
  |TopK-TopP-MinP Sampling - Intermediate Results|√|√|√|×|√|Performs TopK sampling first, then TopP sampling, and finally MinP sampling on the input logits for each batch, and outputs the intermediate computation results.|
  |TopK-MinP Sampling - Intermediate Results|√|×|√|×|√|Performs TopK sampling first, then MinP sampling on the input logits for each batch, and outputs the intermediate computation results.|
  |TopK-TopP Sampling - Intermediate Results|√|√|×|×|√|Performs TopK sampling first, then MinP sampling on the input logits for each batch, and outputs the intermediate computation results.|
  |TopK Sampling - Intermediate Results|√|×|×|×|√|Performs TopK sampling on the input logits for each batch, and outputs the intermediate computation results.|

- Formula:
The input logits is a word frequency table of size [batch, voc_size], where each batch corresponds to an input sequence, and voc_size is the uniform length agreed upon for each batch.<br>
Each row logits[batch][:] in logits executes different computation scenarios based on the corresponding topK[batch], topP[batch], minP[batch, :], and q[batch, :].<br>
In the formulas below, b and v are used to denote the indices along the batch and voc_size dimensions, respectively.

  TopK Sampling

  1. Uses segmented TopK merge sort with segment length v, where the TopK of block {s-1} is used to pre-filter the input of the current block {s}, progressively updating the TopK of a single batch to reduce redundant data and computation.
  2. topK[batch] corresponds to the k value for sampling the current batch. The valid range is 1≤topK[batch]≤min(voc_size[batch], ks_max). If top[K] is out of the valid range, the topK sampling stage of the current batch is skipped, and the sorting of the current batch is also skipped. The input logits[batch] is directly passed to the next module.
  * Based on the value of the input `topK[b]`, the processing strategy of this module is as follows:

    | Type | ≤0 | Valid Range | Invalid Range |
    | :-------:| :------:|:-------:|:-------:|
    |`topK[b]`|skip topK sampling|1≤topK≤min(ksMax, 1024, vocSize), perform topK sampling|topK>min(ksMax, 1024, vocSize), skip topK sampling, select all valid elements|

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

  $
  v = 8 * \text{ks\_max}
  $
  The valid range of ks_max is [1, 1024], defaults to 1024, and must be aligned upward to a multiple of 8.

  * Generate the mask to be filtered.

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask = sortedValue \geq topKValue
  $$

  * Set the parts below the threshold to defLogit via the mask:

  $$
  sortedValue[b][v]=
  \begin{cases}
  defLogit & \text{topKMask[b][v] = false} \\
  sortedValue[b][v] & \text{topKMask[b][v] = true} &
  \end{cases}
  $$

  * defLogit depends on the input constraint attribute input_is_logits, which controls the normalization of the input logits and the output logits_top_kp_select:
  $$
    \text{defLogit} =
    \begin{cases}
    -inf, & \text{inputIsLogits} = \text{true} \\
    0, & \text{inputIsLogits} = \text{false}
    \end{cases}
  $$

  TopP Sampling

  * Based on the input constraint attribute inputIsLogits, if this attribute is True, the sorted results are normalized:
    $$
    \text{logit\_sortProb} =
    \begin{cases}
    \text{safe\_softmax}(\text{logits\_sort}), & \text{inputIsLogits} = \text{True} \\
    \text{logits\_sort}, & \text{inputIsLogits} = \text{False}
    \end{cases}
    $$

  * Based on the value of the input `top_p[b]`, this module adopts the following processing strategies:

    | Type | ≤0 | Valid Range | Invalid Range |
    | :-------:| :------:|:-------:|:-------:|
    |`top_p[b]`|Retains one token with the highest logit.|0 < top_p < 1, performs topP sampling.|top_p ≥ 1, skips topP sampling.|

  * If regular topP sampling is performed and the preceding topK stage has already produced sorted output results, the cumulative logit probabilities are computed based on the topK sampling output, and sampling is truncated according to top_p:
    $$
    topPMask[b] =
    \begin{cases}
    0, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] > p[b] \\
    1, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] \leq p[b]
    \end{cases}
    $$
  * If regular topP sampling is performed but the preceding topK stage is skipped, the top-p mask is computed:
    $$
    topPMask[b] =
    \begin{cases}
    topKMask[b][0:GuessK], & if \space \sum_{\text{GuessK}}^{} probValue[b][*] \ge p[b] \\
    probSum[b][v] \le 1 - p[b], & if \space \text{others}
    \end{cases}
    $$
  * Set the positions to be filtered to the default invalid value defLogit to obtain logits_sort, denoted as sortedValue[b][v]:
  $$
  sortedValue[b][v] =
  \begin{cases}
  defLogit & \quad \text{topPMask}[b][v] = \text{false} \\
  logit\_sortProb[b][v] & \quad \text{topPMask}[b][v] = \text{true}
  \end{cases}
  $$
  * Take the first topK elements in each row of the filtered sortedValue[b][v], look up their original indices in the input, and combine them into logits_idx:
  $$
  logitsIdx[b][v] = Index(sortedValue[b][v] \in Logits)
  $$
  * Use the truncated sortedValue as logitsSortMasked:
  $
  logitsSortMasked[b,:] = sortedValue[b]
  $
  min-p sampling
  * Based on the value of the input `minPs[b]`, the processing strategy of this module is as follows:

    | Type | ≤0 | Valid Range | Invalid Range |
    | :-------:| :------:|:-------:|:-------:|
    |`minPs[b]`|Skips minP sampling and keeps the previous result unchanged.|0<minPs<1, performs minP sampling.|minPs≥1, keeps only 1 maximum token.|

  * If min_ps[b] ∈ [0,1], perform min-p sampling:
    $$
    \text{logitsMax}[b] = \text{Max}(\text{logitsSortMasked}[b])
    $$
    $$
    \text{minPThd} = \text{logitsMax}[b] * \text{minPs}[b]
    $$

    $$
    \text{minPMask}[b] =
    \begin{cases}
    0, & \text{logitsSortMasked}[b][:] < \text{minPThd} \\
    1, & \text{logitsSortMasked}[b][:] \geq \text{minPThd}
    \end{cases}
    $$

    Performs the gather operation to densely arrange minP sampling results and indexes:

    $$
    minPCnt = Sum(minPMask[b]) \\
    minPFiltIdx = Index(minPMask[b]=1)
    $$

    Fills the remaining elements:
    $$
    \text{logitsMinpFiltered}[b,:minPCnt] = logitsSortMasked[b][minPFiltIdx] \\
    \text{logitsMinpFiltered}[b,minPCnt:] = defLogit
    $$

  * Other cases:
    $$
    \text{logitsMinpFiltered}[b, :] =
    \begin{cases}
        \text{logitsSortMasked}[b, :], & \text{if } minPs[b] \leq 0 \\
        \max(\text{logitsSortMasked}[b, :]), & \text{if } minPs[b] \geq 1
    \end{cases}
    $$
    When min_ps[b]≥1, each batch takes only 1 maximum token, and the remaining positions are filled with defLogit.

  Optional Outputs

  * If the input attribute IsNeedLogits is set to True, logitsIndexMasked after combined topK-topP-minP sampling is used for `logits_top_kp_select` output.
    $$
    \text{logitsIndex}[b][v] = \text{Index}(\text{logitsMinpFiltered}[b][v] \in \text{Logits})
    $$
    $$
    \text{logitsIndexMasked}[b,:] = \text{logitsIndex}[b,:] * \text{topKMask}[b] * \text{topPMask}[b] * \text{minPMask}[b]
    $$
    where, if the topK, topP, or minP sampling step is skipped, the corresponding mask is all 1s.
  * Next, logitsIndexMasked is used to select from the input Logits, filtering high-frequency tokens in the input Logits as the `logits_top_kp_select` output:
    $$
    \text{logitsTopKpSelect}[b][v] =
    \begin{cases}
    \text{logits}[b][v], & \text{if } logitsIndexMasked[b,v] = \text{True} \\
    \text{defLogit}, & \text{if } logitsIndexMasked[b,v] = \text{False}
    \end{cases}
    $$

  Subsequent Processing

  * The input of this stage is logitsSortMasked, the joint result of the preceding topK-topP-minP sampling.<br>If none of the preceding sampling steps are enabled, the normalized result of the input logits is directly used:

    $$
    logitsMinpFiltered =
    \begin{cases}
    logits, & if \space inputIsLogits=False \\
    safe\_softmax(logits), & if \space inputIsLogits=True
    \end{cases}
    $$

  * The input here must satisfy logitsSortMasked ∈ [0,1]. Configure the input constraint attribute inputIsLogits based on the actual input Logits, as follows:

    $$
    \text{inputIsLogits} =
    \begin{cases}
    True, & \text{Logits} \notin [0,1] \\
    False, & \text{Logits} \in [0,1]
    \end{cases}
    $$

    such that

    $$
    \text{probs}[b] = \text{logitsMinpFiltered}[b, :]
    $$

    There are three modes: None, QSample, and outputting intermediate results, controlled by the constraint attribute isNeedSampleResult of the input parameter and whether q is input.
  * None:
  * This mode is used when isNeedSampleResult is false and q is not provided. In this mode, the maximum element and index are directly obtained for each batch through Argmax and output via gatherOut.
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b]\left[\text{ArgMax}(\text{probs}[b][:])\right]
    $$
  * QSample:
  * This mode is used when isNeedSampleResult is false and q is provided. In this mode, exponential distribution sampling is first performed on probs:
    $$
    qCnt = \text{Sum}(\text{MinPMask} == 1)
    $$
    $$
    \text{probsOpt}[b] = \frac{\text{probs}[b]}{q[b, :qCnt] + \text{eps}}
    $$
  * Then perform Argmax-GatherOut to output the result:
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{ArgMax}(\text{probsOpt}[b][:])]
    $$
  * Output intermediate results:
  * When isNeedSampleResult is true, this mode is used. In this case, the sampled logitsSortMasked and its original index logitsIdx in the input are output:

    $$
    \text{logitsMinpFiltered}[b, v] =
    \begin{cases}
        \text{logitsMinpFiltered}[b, v], & \text{if } \text{minPMask}[b, v] = 1 \\
        0, & \text{if } \text{minPMask}[b, v] = 0
    \end{cases}
    $$

    $$
    logitsIdx[b][v] = Index(logitsMinpFiltered[b][v])
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnTopKTopPSampleV2GetWorkspaceSize` API first to obtain the workspace size required for the computation and the executor that contains the computation flow of the operator. Then call the `aclnnTopKTopPSampleV2` API to perform the computation.

```Cpp
aclnnStatus aclnnTopKTopPSampleV2GetWorkspaceSize(
  const aclTensor *logits,
  const aclTensor *topK,
  const aclTensor *topP,
  const aclTensor *q,
  const aclTensor *minPs,
  double           eps,
  bool             isNeedLogits,
  int64_t          topKGuess,
  int64_t          ksMax,
  bool             inputIsLogits,
  bool             isNeedSampleResult,
  const aclTensor *logitsSelectIdx,
  const aclTensor *logitsTopKPSelect,
  const aclTensor *logitsIdx,
  const aclTensor *logitsSortMasked,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)

```

```Cpp
aclnnStatus aclnnTopKTopPSampleV2(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)

```

## aclnnTopKTopPSampleV2GetWorkspaceSize

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
          <th>Shape</th>
          <th>Non-contiguous Tensor</th>
        </tr></thead>
      <tbody>
      <tr>
        <td>logits</td>
        <td>Input</td>
        <td>Indicates the input word frequency to be sampled. The word frequency index is fixed to the last dimension, corresponding to `logits` in the formula.</td>
        <td><ul><li>Does not support empty tensors.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>Input</td>
        <td>Indicates the k value sampled for each batch, corresponding to `topK[b]` in the formula.</td>
        <td><ul><li>Does not support empty tensor.</li><li>The shape needs to remain consistent with the first n-1 dimensions of `logits`.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topP</td>
        <td>Input</td>
        <td>Indicates the p value of each batch sampling. Corresponds to `topP[b]` in the formula.</td>
        <td><ul><li>Does not support empty tensor.</li><li>The shape needs to remain consistent with the first n-1 dimensions of `logits`, and the data type needs to remain consistent with `logits`.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>q</td>
        <td>Input</td>
        <td>Indicates the exponential sampling matrix output by topK-topP-minP sampling. Corresponds to `q` in the formula.</td>
        <td><ul><li>shape needs to remain consistent with `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>minPs</td>
        <td>Input</td>
        <td>Indicates the minP value for sampling in each batch. It is `minPs[b]` in the corresponding formula.</td>
        <td><ul><li>The shape needs to remain consistent with the first n-1 dimensions of `logits`, and the data type needs to remain consistent with `logits`.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Input</td>
        <td>Prevents division by zero in softmax and weighted sampling. It is recommended to set this parameter to 1e-8.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedLogits</td>
        <td>Input</td>
        <td>Indicates the condition that controls the output of logitsTopKPSelect. It is recommended to set this parameter to 0.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topKGuess</td>
        <td>Input</td>
        <td>Indicates each batch's candidate logits size when attempting to traverse and sample logits in the topP part.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ksMax</td>
        <td>Input</td>
        <td>Indicates the maximum topK value for each batch during topK sampling. It must be a positive integer.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputIsLogits</td>
        <td>Input</td>
        <td>Indicates whether the input logits are not normalized. Defaults to true.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedSampleResult</td>
        <td>Input</td>
        <td>Indicates whether to output the intermediate computation result. Defaults to false.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsSelectIdx</td>
        <td>Output</td>
        <td>Indicates the position index of the element with the maximum word frequency max(probsOpt[batch, :]) in each batch within the input logits after the topK-topP-minP-sample computation flow.</td>
        <td><ul><li>Does not support empty tensors.</li><li>The shape needs to be consistent with the first n-1 dimensions of `logits`.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logitsTopKPSelect</td>
        <td>Output</td>
        <td>Indicates the remaining unfiltered logits in the input logits after the topK-topP-minP computation flow.</td>
        <td><ul><li>Does not support empty tensors.</li><li>The shape needs to be consistent with that of `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logitsIdx</td>
        <td>Output</td>
        <td>Indicates the position index in the input logits of the intermediate sampling result of each batch after the topK-topP-minP computation flow.</td>
        <td><ul><li>Does not support empty Tensor.</li><li>The shape needs to be consistent with `logits`.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logitsSortMasked</td>
        <td>Output</td>
        <td>Indicates the intermediate sampling result of each batch after the topK-topP-minP computation flow.</td>
        <td><ul><li>Does not support empty Tensor.</li><li>Shape needs to be consistent with `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be side-applied on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 300px">
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
      <td>Any of the input parameters logits, topK, or topP is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of logits, topK, topP, q, and minPs are not within the supported range.</td>
    </tr>
    <tr>
      <td>The sizes of logits and q are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimensions of topK, topP, and minPs are inconsistent with the first n-1 dimensions of logits.</td>
    </tr>
    <tr>
      <td>The data types of logits, topP, and minPs are inconsistent.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The dimensions of logits and q are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnTopKTopPSampleV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Address of the workspace memory side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory side-applied on the Device, obtained by the first-phase API aclnnTopKTopPSampleV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - Deterministic computation:
    - aclnnTopKTopPSampleV2 defaults to a deterministic implementation.
  - Input value range restrictions:
    - For all sampling parameters, their dimensions must satisfy batch > 0 and 0 < vocSize <= 2^20.
    - For logits, ensure that the value of each element after the $exp(*)$ operation is not smaller than the minimum valid value of FLOAT32. Otherwise, the corresponding Safe_softmax result will be 0, which may cause unexpected output of the operator.
    - topK accepts only non-negative values as valid input. Passing 0 or a negative value skips sampling for the corresponding batch.
  - Input shape constraints:
    - The sizes and dimensions of logits, q, logitsTopKPselect, logitsIdx, and logitsSortMasked must be exactly the same.
    - All dimensions except the last one of logits, topK, topP, minPs, logitsSelectIdx, logitsIdx, and logitsSortMasked must be exactly the same in order and size. Currently, logits can only be 2-dimensional, and topK, topP, and logitsSelectIdx must be 1-dimensional non-empty tensors. logits, topK, and topP do not allow empty tensors as input. To skip the corresponding module, set the input according to the corresponding rules.

  - Q-Sample constraints:
      - The indices of input tensor q are always aligned to the corresponding batch element by element as input to the $\text{probs}[b]$ for subsequent processing.
      - The $\text{probs}[b]$ input must be compactly arranged, that is, valid elements are concentrated at the front of the current batch.
      - If at least one of topK or topP sampling is enabled in the preceding step of the current batch, the elements of $\text{probs}[b]$ are also arranged in descending order.
  - Other constraints:
    - To skip the topK module individually, pass a tensor of size [batch, 1] with each element set to an invalid value.
    - If min(ksMaxAligned, 1024) < topK[batch] < vocSize[batch], all valid elements of the current batch are selected and topK sampling is skipped. ksMaxAligned is ksMax rounded up to the nearest multiple of 8, and the value range of ksMax is [1, 1024].
    - To skip the topP module individually, pass a tensor of size [batch, 1] with each element ≥ 1.
    - To skip the minP module, pass `minPs=nullptr` or pass a tensor of size [batch, 1] with each element ≤ 0.
    - To skip the sample module, pass `q=nullptr`. To use the sample module, pass a tensor of size [batch, vocSize].
    - To output intermediate results, set isNeedSampleResult to true and pass `q=nullptr`. In this case, logitsSelectIdx is not output.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_top_k_top_p_sample_v2.h"

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
  // Call aclrtMalloc to allocate device-side memory.
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
    // 1. (Boilerplate) Initialize device/stream by referring to the acl API.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Customize the construction according to the API.
    std::vector<int64_t> logitsShape = {48, 131072};
    std::vector<int64_t> topKPShape = {48};
    long long vocShapeSize = GetShapeSize(logitsShape);
    long long batchShapeSize = GetShapeSize(topKPShape);

    void* logitsDeviceAddr = nullptr;
    void* topKDeviceAddr = nullptr;
    void* topPDeviceAddr = nullptr;
    void* qDeviceAddr = nullptr;
    void* minPsDeviceAddr = nullptr;
    void* logitsSelectIdxDeviceAddr = nullptr;
    void* logitsTopKPSelectDeviceAddr = nullptr;
    void* logitsIdxDeviceAddr = nullptr;
    void* logitsSortMaskedDeviceAddr = nullptr;

    aclTensor* logits = nullptr;
    aclTensor* topK = nullptr;
    aclTensor* topP = nullptr;
    aclTensor* q = nullptr;
    aclTensor* minPs = nullptr;
    aclTensor* logitsSelectIdx = nullptr;
    aclTensor* logitsTopKPSelect = nullptr;
    aclTensor* logitsIdx = nullptr;
    aclTensor* logitsSortMasked = nullptr;
    std::vector<int16_t> logitsHostData(48 * 131072, 1);
    std::vector<int32_t> topKHostData(48, 128);
    std::vector<int16_t> topPHostData(48, 1);
    std::vector<float> qHostData(48 * 131072, 1.0f);
    std::vector<int16_t> minPsHostData(48, 1);

    std::vector<int64_t> logitsSelectIdxHostData(48, 0);
    std::vector<float> logitsTopKPSelectHostData(48 * 131072, 0);
    std::vector<int64_t> logitsIdxHostData(48 * 131072, 0);
    std::vector<float> logitsSortMaskedHostData(48 * 131072, 0);

    float eps = 1e-8;
    bool isNeedLogits = false;
    int64_t topKGuess = 32;
    int64_t ksMax = 1024;
    bool inputIsLogits = true;
    bool isNeedSampleResult = false;

    // Create logitsaclTensor.
    ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_BF16, &logits);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create topKaclTensor.
    ret = CreateAclTensor(topKHostData, topKPShape, &topKDeviceAddr, aclDataType::ACL_INT32, &topK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create topPaclTensor.
    ret = CreateAclTensor(topPHostData, topKPShape, &topPDeviceAddr, aclDataType::ACL_BF16, &topP);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create q aclTensor.
    ret = CreateAclTensor(qHostData, logitsShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create minps aclTensor.
    ret = CreateAclTensor(minPsHostData, topKPShape, &minPsDeviceAddr, aclDataType::ACL_BF16, &minPs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create logitsSelectIdx aclTensor.
    ret = CreateAclTensor(logitsSelectIdxHostData, topKPShape, &logitsSelectIdxDeviceAddr, aclDataType::ACL_INT64, &logitsSelectIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create logitsTopKPSelect aclTensor.
    ret = CreateAclTensor(logitsTopKPSelectHostData, logitsShape, &logitsTopKPSelectDeviceAddr, aclDataType::ACL_FLOAT, &logitsTopKPSelect);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create logitsIdx aclTensor.
    ret = CreateAclTensor(logitsIdxHostData, logitsShape, &logitsIdxDeviceAddr, aclDataType::ACL_INT64, &logitsIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the logitsSortMasked aclTensor.
    ret = CreateAclTensor(logitsSortMaskedHostData, logitsShape, &logitsSortMaskedDeviceAddr, aclDataType::ACL_FLOAT, &logitsSortMasked);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnTopKTopPSampleV2.
    ret = aclnnTopKTopPSampleV2GetWorkspaceSize(logits, topK, topP, q, minPs, eps, isNeedLogits, topKGuess, ksMax, inputIsLogits,
      isNeedSampleResult, logitsSelectIdx, logitsTopKPSelect, logitsIdx, logitsSortMasked, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnTopKTopPSampleV2.
    ret = aclnnTopKTopPSampleV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait synchronously for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
    auto size = GetShapeSize(topKPShape);
    std::vector<int64_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), logitsSelectIdxDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
    }

    // 6. Release the aclTensor. Modify it based on the specific API definition.
    aclDestroyTensor(logits);
    aclDestroyTensor(topK);
    aclDestroyTensor(topP);
    aclDestroyTensor(q);
    aclDestroyTensor(logitsSelectIdx);
    aclDestroyTensor(logitsTopKPSelect);
    // 7. Release device resources. Modify it based on the specific API definition.
    aclrtFree(logitsDeviceAddr);
    aclrtFree(topKDeviceAddr);
    aclrtFree(topPDeviceAddr);
    aclrtFree(qDeviceAddr);
    aclrtFree(logitsSelectIdxDeviceAddr);
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
