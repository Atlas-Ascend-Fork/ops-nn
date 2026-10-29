# aclnnCtcLossBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:19:13.414Z -->

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

- Description: performs backward propagation of [aclnnCtcLoss](../../ctc_loss_v2/docs/aclnnCtcLoss.md) and calculates the CTC loss gradient.
- Formula:
Define $y_{k}^{t}$ as the probability that the true character is $k$ at time $t$. (Generally, $y_{k}^{t}$ is an element of the output matrix after softmax.) The set of all sequences that can be formed from the character set $L^{'}$ is denoted as $L^{'T}$. Any sequence in $L^{'T}$ is called a path, denoted as $π$. The distribution of $π$ is formula (1):

  $$
  p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
  $$

  Define a many-to-one mapping B: $L^{'T} \to L^{\leq T}$. The conditional probability of $l \in L^{\leq T}$ obtained through mapping B is equal to the sum of probabilities of all possible paths corresponding to $l$, as shown in formula (2):

  $$
  p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
  $$

  The task of finding the path of $l$ that maximizes $p(l|x)$ is called decoding, as shown in formula (3):

  $$
  h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
  $$

  When zeroInfinity is True

  $$
  h(x)=\begin{cases}0,&h(x) == inf \text{ or } h(x) == -inf \\h(x),&\text { else }\end{cases}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnCtcLossBackwardGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnCtcLossBackward" to perform the computation.

```Cpp
aclnnStatus aclnnCtcLossBackwardGetWorkspaceSize(
    const aclTensor*     gradOut,
    const aclTensor*     logProbs,
    const aclTensor*     targets,
    const aclIntArray*   inputLengths,
    const aclIntArray*   targetLengths,
    const aclTensor*     negLogLikelihood,
    const aclTensor*     logAlpha,
    int64_t              blank,
    bool                 zeroInfinity,
    aclTensor*           out,
    uint64_t*            workspaceSize,
    aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnCtcLossBackward(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnCtcLossBackwardGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
    <col style="width: 173px">
    <col style="width: 120px">
    <col style="width: 222px">
    <col style="width: 338px">
    <col style="width: 156px">
    <col style="width: 104px">
    <col style="width: 162px">
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
      <td>gradOut(aclTensor*)</td>
      <td>Input</td>
      <td>Represents the gradient update coefficient.</td>
      <td>The data type must be consistent with logProbs.</td>
      <td>FLOAT16, BFLOAT16, FLOAT, DOUBLE</td>
      <td>ND</td>
      <td>(N)</td>
      <td>√</td>
      </tr>
      <tr>
      <td>logProbs (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the log probabilities of the output.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT, DOUBLE</td>
      <td>ND</td>
      <td>(T,N,C)<br>T is the input length, N is the batch size, and C is the number of classes, including the blank symbol.</td>
      <td>√</td>
      </tr>
      <tr>
      <td>targets (aclTensor*)</td>
      <td>Input</td>
      <td>Labels containing the target sequences.</td>
      <td>When the shape is (N,S), S is not less than the maximum value in targetLengths; or the shape is (SUM(targetLengths)), assuming that targets is unpadded and concatenated in 1 dimension. The values must be less than C and greater than or equal to 0.</td>
      <td>INT64, INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
      </tr>
      <tr>
      <td>inputLengths (aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the actual lengths of the input sequences.</td>
      <td>The array length is N. Each value in the array must be less than or equal to T.</td>
      <td>INT64, INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>targetLengths(aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the actual lengths of target sequences.</td>
      <td>The array length is N. When the shape of targets is (N,S), each value in the array must be less than or equal to S, and the maximum value is maxTargetLength.</td>
      <td>INT64, INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>negLogLikelihood(aclTensor*)</td>
      <td>Input</td>
      <td>Represents the loss value differentiable with respect to each input node.</td>
      <td>The data type must be consistent with that of logProbs.</td>
      <td>Consistent with logProbs</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
      </tr>
      <tr>
      <td>logAlpha (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the probability of possible alignments from input to target.</td>
      <td>The data type must be consistent with logProbs. The shape must be a non-empty 3D tensor, with shape (N,T,alpha), satisfying (alpha−1)/2>=maxTargetLength.</td>
      <td>Consistent with logProbs</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
      </tr>
      <tr>
      <td>blank (int64_t)</td>
      <td>Input</td>
      <td>Indicates the blank identifier.</td>
      <td>The value must be less than C and greater than or equal to 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>zeroInfinity (bool)</td>
      <td>Input</td>
      <td>Indicates whether to zero infinite losses and the associated gradients.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the CTC loss gradient.</td>
      <td>The data type must be consistent with that of gradOut.</td>
      <td>Consistent with gradOut</td>
      <td>ND</td>
      <td>(T,N,C)</td>
      <td>√</td>
      </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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
    </tbody></table>

- **Return value**

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
      <td>gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, or out is not within the supported range.</td>
      </tr>
      <tr>
      <td>The data types of gradOut, negLogLikelihood, logAlpha, and out are different from that of logProbs.</td>
      </tr>
      <tr>
      <td>The tensors of gradOut, logProbs, targets, negLogLikelihood, logAlpha, or out do not meet the corresponding shape requirements, or the length of the ArrayList of inputLengths or targetLengths does not meet the requirements.</td>
      </tr>
      <tr>
      <td>blank is out of the value range.</td>
      </tr>
    </tbody>
    </table>

## aclnnCtcLossBackward

- **Parameters:**

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
      <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnCtcLossBackwardGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that specifies the task to be executed.</td>
      </tr>
      </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

None.

## Example

The following sample code is for reference only. For details about compiling and running, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ctc_loss_backward.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: initialize resources.
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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to the device memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Process the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> gradOutShape = {4};
  // logProbsShape (T, N, C)
  std::vector<int64_t> logProbsShape = {12, 4, 5};
  std::vector<int64_t> targetsShape = {4, 7};
  std::vector<int64_t> negLoglikelihoodShape = {4};
  // logAlphaShape (N, T, X)  X = ((max(targetLengths) * 2 + 1) + 7) / 8 * 8;
  std::vector<int64_t> logAlphaShape = {4, 12, 16};
  std::vector<int64_t> outShape = {12, 4, 5};
  void* gradOutDeviceAddr = nullptr;
  void* logProbsDeviceAddr = nullptr;
  void* targetsDeviceAddr = nullptr;
  void* negLoglikelihoodDeviceAddr = nullptr;
  void* logAlphaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* logProbs = nullptr;
  aclTensor* targets = nullptr;
  aclIntArray* inputLengths = nullptr;
  aclIntArray* targetLengths = nullptr;
  aclTensor* negLoglikelihood = nullptr;
  aclTensor* logAlpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutHostData = {1, 1, 1, 1};
  std::vector<float> logProbsHostData = {
    -1.0894, -2.7162, -0.9764, -1.9126, -2.6162,
    -2.0684, -2.4871, -2.0866, -1.7205, -0.7187,
    -2.4423, -1.2017, -1.4653, -1.1821, -2.5942,
    -2.4670, -2.7257, -1.4135, -2.1042, -0.7248,

    -3.7759, -1.3742, -1.2549, -1.5807, -1.4562,
    -1.3826, -1.8995, -1.8527, -0.9493, -2.8895,
    -1.6316, -2.6603, -2.5014, -0.6992, -1.8609,
    -1.9269, -2.2350, -0.8073, -1.8906, -1.8947,

    -0.3468, -2.5855, -2.0723, -2.7147, -3.6668,
    -0.9541, -1.7258, -2.0693, -1.6378, -2.1531,
    -3.5386, -3.4830, -0.2532, -2.0557, -3.3261,
    -1.1480, -1.8080, -0.8244, -3.2414, -3.1909,

    -0.8866, -0.7540, -4.4312, -3.4634, -2.6000,
    -1.2785, -1.8347, -3.3122, -0.7620, -2.8349,
    -1.4975, -1.3865, -0.9645, -3.8171, -2.0939,
    -2.3536, -2.0773, -1.4981, -0.8372, -2.0938,

    -1.2186, -0.8285, -2.9399, -2.1159, -2.3620,
    -2.3139, -0.6503, -2.7249, -1.2340, -3.7927,
    -0.7143, -2.5084, -3.2826, -2.6651, -1.1334,
    -1.6965, -1.9728, -2.3849, -1.6052, -0.9554,

    -1.6384, -1.2596, -2.1680, -1.8476, -1.3866,
    -3.0455, -0.5737, -2.5339, -2.1118, -1.6681,
    -2.4675, -2.8842, -0.4329, -3.6266, -1.6925,
    -3.1023, -2.7696, -1.2755, -0.6470, -2.4143,

    -2.0107, -2.0912, -1.3053, -0.8557, -3.0683,
    -1.2872, -3.6523, -1.6703, -2.7596, -0.8063,
    -2.4633, -1.2959, -1.6153, -2.3072, -1.0705,
    -3.0543, -0.6473, -1.1650, -2.9025, -2.7710,

    -3.5519, -2.0400, -1.8667, -1.4289, -0.8050,
    -1.4602, -0.7452, -1.5754, -3.1624, -3.1247,
    -1.4677, -1.2725, -2.9575, -1.8883, -1.2513,
    -1.2164, -1.5894, -2.2217, -2.3714, -1.2110,

    -2.0843, -0.6515, -1.4252, -2.9402, -2.7964,
    -1.5261, -2.5471, -1.7167, -1.9846, -0.9488,
    -1.4847, -1.7093, -1.4095, -1.7293, -1.7675,
    -0.9203, -4.2299, -1.8740, -1.4076, -1.6671,

    -1.9052, -0.8330, -2.1839, -2.2459, -1.6193,
    -2.9108, -1.2114, -1.4616, -1.7297, -1.4330,
    -2.2656, -0.7878, -1.8533, -1.8711, -2.0349,
    -2.2457, -2.1395, -1.4509, -0.7538, -2.6381,

    -0.8078, -2.1054, -2.6703, -1.1108, -3.3867,
    -1.7774, -1.8426, -1.9473, -1.3293, -1.3273,
    -1.3490, -1.9842, -2.5357, -2.2161, -0.8800,
    -1.5412, -1.8003, -2.7603, -0.8606, -2.0066,

    -1.8342, -2.2741, -1.8348, -1.5833, -0.9877,
    -3.5196, -2.3361, -0.9124, -0.9307, -2.5531,
    -1.4862, -1.2153, -1.4453, -3.4462, -1.5625,
    -2.6455, -1.4153, -1.3079, -1.1568, -2.2897};
  std::vector<int64_t> targetsHostData = {
    1, 2, 1, 1, 2, 4, 1,
    2, 2, 2, 2, 2, 2, 3,
    4, 2, 1, 4, 3, 1, 4,
    4, 1, 4, 2, 2, 2, 3};

  std::vector<float> negLoglikelihoodHostData = {10.1999, 16.1340, 14.9006,  9.3596};
  std::vector<float> logAlphaHostData = {
    -1.0894, -2.7162, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999,
    -4.8653, -2.2842, -6.4921, -3.9711, -99999, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -5.2121, -4.7967, -2.6162, -4.1742, -4.3179, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -6.0987, -5.0438, -3.3957, -6.7671, -4.4369, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -7.3173, -5.5735, -4.4384, -6.1313, -5.5627, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -8.9557, -6.6720, -5.7981, -6.1973, -6.7523, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -10.9664, -8.6661, -7.4600, -6.3671, -7.7544, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -14.5183, -10.6106, -10.7501, -7.8722, -9.6961, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -16.6026, -11.2422, -12.0691, -9.1833, -9.8069, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -18.5078, -12.0705, -12.7846, -11.1988, -10.6593, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,

    -2.0684, -2.0866, -99999, -99999, -99999, -99999, -99999,-99999, -99999, -99999, -99999, -99999, -99999, -99999,-99999, -99999,
    -3.4510, -3.2370, -3.4692, -99999, -99999, -99999, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -4.4051, -4.7144, -3.6073, -5.5385, -99999, -99999, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -5.6836, -7.1669, -4.6003, -6.7841, -6.8170, -99999, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -7.9975, -8.2040, -6.8402, -7.2185, -8.4212, -9.5419, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -11.0430, -9.9362, -9.6580, -8.8523, -10.0013, -10.6729, -12.5874,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -12.3302, -11.3209, -10.3815, -10.1533, -9.8642, -11.2589, -11.8226,-14.2577, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -13.7904, -12.5855, -11.5118, -11.1431, -10.7654, -11.2181, -12.2686,-13.3140, -15.7179, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -15.3165, -14.0400, -12.7439, -12.3341, -11.7695, -11.9899, -12.4443,-13.6840, -14.7536, -17.4346, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -18.2273, -15.2555, -15.4129, -13.2866, -14.2301, -12.6421, -14.4091,-13.6517, -16.2998, -16.1490, -20.3454, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,

    -2.4423, -2.5942, -99999, -99999, -99999, -99999, -99999,-99999, -99999, -99999, -99999, -99999, -99999, -99999,-99999, -99999,
    -4.0739, -3.6831, -4.2258, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -7.6125, -6.4925, -6.7635, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -9.1100, -8.3040, -7.4232, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -9.8243, -9.0682, -7.7908, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -12.2918, -10.3758, -10.0124, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -14.7551, -11.3089, -11.9478, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -16.2228, -12.5289, -12.3528, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -17.7075, -14.2718, -13.2285, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -19.9731, -16.2750, -15.1923, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,

    -2.4670, -0.7248, -99999, -99999, -99999, -99999, -99999,-99999, -99999, -99999, -99999, -99999, -99999, -99999,-99999, -99999,
    -4.3939, -2.4581, -2.6517, -2.9598, -99999, -99999, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -5.5419, -5.5142, -3.0051, -3.3784, -4.1078, -6.1507, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -7.8955, -6.9286, -5.2805, -4.5115, -5.3385, -5.0374, -8.5043,-7.6488, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -9.5920, -7.5617, -6.8010, -6.0444, -5.8452, -4.7597, -6.7031,-7.3228, -9.3453, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -12.6943, -9.8527, -9.5199, -8.2901, -8.3490, -6.6950, -7.7281,-5.8361, -10.3008, -10.6208, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -15.7486, -12.5670, -12.0336, -8.5306, -10.6803, -9.1337, -9.4448,-6.5472, -8.8790, -10.9199, -13.6751, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -16.9650, -13.7373, -12.7884, -10.0734, -9.6368, -9.2326, -9.8004,-8.6463, -7.6709, -10.9785, -12.0746, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -17.8853, -15.3655, -13.3814, -14.2154, -10.0586, -10.1583, -9.7039,-9.8935, -8.2713, -9.5090, -11.6105, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -20.1310, -17.9262, -15.4983, -15.0687, -12.2888, -12.0440, -11.4581,-10.2538, -10.3368, -9.4675, -11.6393, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000};

  std::vector<float> outHostData = {
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
  };

  // Create the gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the logProbs aclTensor.
  ret = CreateAclTensor(logProbsHostData, logProbsShape, &logProbsDeviceAddr, aclDataType::ACL_FLOAT, &logProbs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the targets aclTensor.
  ret = CreateAclTensor(targetsHostData, targetsShape, &targetsDeviceAddr, aclDataType::ACL_INT64, &targets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> inputLengthsSizeData = {10,10,10,10};
  inputLengths = aclCreateIntArray(inputLengthsSizeData.data(), 4);
  CHECK_RET(inputLengths != nullptr, return ACL_ERROR_BAD_ALLOC);
  std::vector<int64_t> targetLengthsSizeData = {2, 3, 1, 5};
  targetLengths = aclCreateIntArray(targetLengthsSizeData.data(), 4);
  CHECK_RET(targetLengths != nullptr, return ACL_ERROR_BAD_ALLOC);

  // Create the negLoglikelihood aclTensor.
  ret = CreateAclTensor(negLoglikelihoodHostData, negLoglikelihoodShape, &negLoglikelihoodDeviceAddr, aclDataType::ACL_FLOAT, &negLoglikelihood);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the logAlpha aclTensor.
  ret = CreateAclTensor(logAlphaHostData, logAlphaShape, &logAlphaDeviceAddr, aclDataType::ACL_FLOAT, &logAlpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnCtcLossBackward.
  ret = aclnnCtcLossBackwardGetWorkspaceSize(gradOut, logProbs, targets, inputLengths, targetLengths, negLoglikelihood, logAlpha, 0, false, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnCtcLossBackward.
  ret = aclnnCtcLossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossBackward failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value of out and copy the result from the device side to the host side. Modify it according to the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and IntArray. Modify it according to the definition of the specific API.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(logProbs);
  aclDestroyTensor(targets);
  aclDestroyIntArray(inputLengths);
  aclDestroyIntArray(targetLengths);
  aclDestroyTensor(negLoglikelihood);
  aclDestroyTensor(logAlpha);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(logProbsDeviceAddr);
  aclrtFree(targetsDeviceAddr);
  aclrtFree(negLoglikelihoodDeviceAddr);
  aclrtFree(logAlphaDeviceAddr);
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
