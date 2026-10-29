# aclnnGRU

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:49:42.066Z -->

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

- Description: Implements the computation of the Gated Recurrent Unit (GRU). Supports multi-layer stacking, bidirectional computation, and two input modes: fixed-length sequences and variable-length sequences (PackedSequence). In training mode, the intermediate results of each gate (r, z, n, hn, h) can be output for backward propagation.

- Input and output shapes:
  - Fixed-length mode (batchSizes is nullptr): If the shape of the input tensor input is (T, B, I) (batchFirst=false) or (B, T, I) (batchFirst=true), the shape of the output tensor output is (T, B, D *H) (batchFirst=false) or (B, T, D* H) (batchFirst=true), and the shape of the final hidden state hy is (L * D, B, H).
  - Variable-length mode (batchSizes is not nullptr): the shape of the input tensor input is (sum(batch_size), I), the shape of the output tensor output is (sum(batch_size), D *H), and the shape of the final hidden state hy is (L* D, B, H).

- Formula:

  $$
  r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr})
  $$

  $$
  z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz})
  $$

  $$
  n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)} + b_{hn}))
  $$

  $$
  h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
  $$

  Where:
  - $\sigma$ is the sigmoid activation function.
  - $\odot$ indicates element-wise multiplication (Hadamard product).
  - $r_t$, $z_t$, and $n_t$ are the reset gate, update gate, and new gate, respectively.
  - $h_t$ is the hidden state at the current time step.
  - $W_{ir}$, $W_{iz}$, and $W_{in}$ are the input weight matrices with shape $[H, I]$ (stored in transposed form as $[3H, I]$).
  - $W_{hr}$, $W_{hz}$, $W_{hn}$ are hidden state weight matrices, with shape $[H, H]$ (stored in transposed form as $[3H, H]$).
  - If hasBias=true, $b_{ir}$, $b_{iz}$, $b_{in}$ and $b_{hr}$, $b_{hz}$, $b_{hn}$ are the input biases and hidden state biases respectively, both with shape $[H]$ (stored as $[3H]$).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnGRUGetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the computation flow, and then call "aclnnGRU" to perform the computation.

```cpp
aclnnStatus aclnnGRUGetWorkspaceSize(
  const aclTensor     *input,
  const aclTensorList *params,
  const aclTensor     *hx,
  const aclTensor     *batchSizes,
  bool                 hasBias,
  int64_t              numLayers,
  double               dropout,
  bool                 train,
  bool                 bidirection,
  bool                 batchFirst,
  aclTensor           *output,
  aclTensor           *hy,
  aclTensorList       *rOut,
  aclTensorList       *zOut,
  aclTensorList       *nOut,
  aclTensorList       *hnOut,
  aclTensorList       *hOut,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```cpp
aclnnStatus aclnnGRU(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGRUGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1650px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 430px">
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
      <td>Input sequence of GRU, corresponding to x in the formula.</td>
      <td><ul>
      <li>If passing valid batchSizes, the shape is [sum(batch_size), input_size];</li>
      <li>If passing a null pointer for batchSizes, the shape is [time_step, batch_size, input_size] or [batch_size, time_step, input_size] (when batchFirst=true)</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>params (aclTensorList*)</td>
      <td>Input</td>
      <td>Weight and bias tensor list of each GRU layer, corresponding to w and b in the formula.</td>
      <td><ul><li>If bidirection is True, `D = 2`; otherwise, `D = 1`. If hasBias is True, `B = 2`; otherwise, `B = 1`. The list length is 2 * B * D * num_layers (2 is the number of weights W_ih and W_hh per layer per direction).</li><li>If both bidirection and hasBias are True, the layout is: [weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0].</li>
      <li>If hasBias is False, no bias item exists; if bidirection is False, no reverse item exists.</li><li>For multiple layers, the layout is layer by layer.</li><li>The data type is consistent with that of input.</li><li>weight_ih: [3*hidden_size, cur_input_size] (cur_input_size=input_size for the first layer, cur_input_size=D*hidden_size for other layers)<br>weight_hh: [3*hidden_size, hidden_size]<br>bias_ih: [3*hidden_size]<br>bias_hh: [3*hidden_size]</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hx (aclTensor*)</td>
      <td>Optional input</td>
      <td>Initial hidden state for each GRU layer. Corresponds to h(t-1) at time step 0.</td>
      <td><ul><li>Optional parameter. Passing a null pointer indicates that this parameter is not used. In this case, the initial hidden state is a zero vector.</li><li>In multi-layer bidirectional mode, the data of each tensor is arranged along dimension 0, bidirectional first and then layer by layer.</li><li>The data type is consistent with that of input.</li><li>[numLayers * D, batch_size, hidden_size].</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchSizes (aclTensor*)</td>
      <td>Optional input</td>
      <td>Number of valid sequence batches at each time step in the variable-length GRU input sequence, corresponding to the PackedSequence mode.</td>
      <td><ul><li>Optional parameter. Passing a null pointer indicates using the fixed-length mode.</li><li>When valid batchSizes are passed, the PackedSequence mode is used, and the input shape must be 2D [sum(batch_size), input_size].</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hasBias (bool)</td>
      <td>Input</td>
      <td>Indicates whether there is a bias b, corresponding to $b_{ir}$, $b_{iz}$, $b_{in}$, $b_{hr}$, $b_{hz}$, $b_{hn}$ in the corresponding formula.</td>
      <td>If hasBias is True, the params list contains bias parameters. Otherwise, it does not.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numLayers (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of stacked GRU layers, which corresponds to $L$ in the corresponding formula.</td>
      <td>Value range: integers greater than or equal to 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>`dropout (double)`</td>
      <td>Input</td>
      <td>Indicates the dropout probability.</td>
      <td>The current version does not implement dropout. Passing any value does not affect the computation result.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>train (bool)</td>
      <td>Input</td>
      <td>Indicates whether training mode is used.</td>
      <td>When train is True, rOut, zOut, nOut, hnOut, and hOut must be passed for gated output; when train is False, pass a null pointer for the corresponding parameters.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bidirection (bool)</td>
      <td>Input</td>
      <td>Indicates whether it is a bidirectional GRU.</td>
      <td>When bidirection is True, forward and backward are computed simultaneously at each time step. The output dimension is 2*H.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchFirst(bool)</td>
      <td>Input</td>
      <td>Indicates whether the batch dimension of input and output is the first dimension.</td>
      <td>When batchFirst is True, the shapes of input and output are (B, T, I) and (B, T, D*H). When batchFirst is False, the shapes are (T, B, I) and (T, B, D*H).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output of all time steps, corresponding to $h_t$ in the corresponding formula.</td>
      <td><ul><li>If passing a valid batchSizes, the shape is [sum(batch_size), hidden_size * D];</li><li>If passing a null pointer batchSizes, the shape is [time_step, batch_size, hidden_size * D] or [batch_size, time_step, hidden_size * D] (when batchFirst=true)</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hy (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the final hidden state of all directions in the last layer, corresponding to $h_t$ at the last time step in the corresponding formula.</td>
      <td>The shape is [numLayers * D, batch_size, hidden_size].</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rOut (aclTensorList*)</td>
      <td>Optional output</td>
      <td>Indicates the reset gate output in training mode, corresponding to $r_t$ in the formula.</td>
      <td><ul><li>Must be passed in training mode; passing a null pointer in inference mode.</li><li>List length = dScale * numLayers.</li><li>Fixed-length mode: each shape in the list is [time_step, batch_size, hidden_size];</li><li>Variable-length mode: each shape in the list is [sum(batch_size), hidden_size].</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zOut (aclTensorList*)</td>
      <td>Optional output</td>
      <td>Indicates the update gate output in training mode, corresponding to $z_t$ in the formula.</td>
      <td><ul><li>Must be passed in training mode; passing a null pointer in inference mode.</li><li>List length = dScale * numLayers.</li><li>Fixed-length mode: each in the list has a shape of [time_step, batch_size, hidden_size];</li><li>Variable-length mode: each in the list has a shape of [sum(batch_size), hidden_size].</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>nOut (aclTensorList*)</td>
      <td>Optional output</td>
      <td>Indicates the new gate output in training mode, corresponding to $n_t$ in the formula.</td>
      <td><ul><li>Must be passed in training mode; pass a null pointer in inference mode.</li><li>List length = dScale * numLayers.</li><li>Fixed-length mode: each shape in the list is [time_step, batch_size, hidden_size];</li><li>Variable-length mode: each shape in the list is [sum(batch_size), hidden_size].</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hnOut (aclTensorList*)</td>
      <td>Optional output</td>
      <td>Indicates the hidden state - new gate output $W_{hn} h_{(t-1)} + b_{hn}$ in the training mode.</td>
      <td><ul><li>Must be passed in the training mode; pass a null pointer in the inference mode.</li><li>List length = dScale * numLayers.</li><li>Fixed-length mode: each in the list has a shape of [time_step, batch_size, hidden_size];</li><li>Variable-length mode: each in the list has a shape of [sum(batch_size), hidden_size].</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hOut (aclTensorList*)</td>
      <td>Optional output</td>
      <td>Indicates the hidden state output in the training mode, that is, $h_t$ at each time step.</td>
      <td><ul><li>Must be passed in training mode; pass a null pointer in inference mode.</li><li>List length = dScale * numLayers.</li><li>Fixed-length mode: each shape in the list is [time_step, batch_size, hidden_size];</li><li>Variable-length mode: each shape in the list is [sum(batch_size), hidden_size].</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
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
      <td>Returns the op executor, which contains the operator computation process.</td>
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
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>input, params, output, or hy is a null pointer, or rOut, zOut, nOut, hnOut, or hOut is a null pointer in training mode.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of input, elements in params, hx, output, hy, or the gated output in training mode is not in the supported range (only FLOAT32 and FLOAT16 are supported).</td>
    </tr>
    <tr>
      <td>The data types of input, elements in params, hx, output, hy, and the gated output in training mode are inconsistent.</td>
    </tr>
    <tr>
      <td>The list length of params is incorrect (should be 2 * bScale * dScale * numLayers).</td>
    </tr>
    <tr>
      <td>In training mode, the list lengths of rOut, zOut, nOut, hnOut, and hOut are incorrect (should be dScale * numLayers).</td>
    </tr>
    <tr>
      <td>The dimension of input is not 2D or 3D.</td>
    </tr>
    <tr>
      <td>The dimension of the weight in params is not 2D, or the dimension of the bias is not 1D.</td>
    </tr>
    <tr>
      <td>The dimension of hx is not 3D, or the dimensions of output and hy are not 3D (in fixed-length mode).</td>
    </tr>
    <tr>
      <td>The shapes of weights or biases in params do not meet the expectation: $W_{ih}$ should be $[3H, I]$ for the first layer and $[3H, D*H]$ for non-first layers, $W_{hh}$ should be $[3H, H]$, and biases should be $[3H]$.</td>
    </tr>
  </tbody></table>

## aclnnGRU

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnGRUGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. Contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: aclnnGRU defaults to a deterministic implementation.

  <!-- end id7 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gru.h"

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

void PrintOutResult(const std::string& name, const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name.c_str(), ret); return);
    LOG_PRINT("=== %s shape=[", name.c_str());
    for (size_t i = 0; i < shape.size(); i++) {
        LOG_PRINT("%ld%s", shape[i], (i + 1 < shape.size()) ? "," : "");
    }
    LOG_PRINT("] ===\n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
    }
    LOG_PRINT("\n");
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate: AscendCL initialization
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(
    const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType, aclTensorList** tensor,
    T initVal = 1)
{
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        std::vector<T> hostData(GetShapeSize(shapes[i]), initVal);
        int ret = CreateAclTensor<float>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) device/stream initialization
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs.
    int64_t timeStep = 2;
    int64_t batchSize = 3;
    int64_t inputSize = 4;
    int64_t hiddenSize = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    bool isbias = false;
    bool batchFirst = false;
    bool bidirection = false;
    bool isTraining = true;
    double dropout = 0.0;
    int64_t dScale = bidirection ? 2 : 1;
    int64_t ldScale = numLayers * dScale;

    std::vector<int64_t> inputShape = {timeStep, batchSize, inputSize};
    std::vector<int64_t> outputShape = {timeStep, batchSize, dScale * hiddenSize};
    std::vector<int64_t> hyShape = {ldScale, batchSize, hiddenSize};
    std::vector<int64_t> hxShape = {ldScale, batchSize, hiddenSize};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    auto curLayerInputSize = inputSize;
    for (int i = 0; i < numLayers; i++) {
        for (int64_t j = 0; j < dScale; j++) {
            paramsListShape.push_back({hiddenSize * gateNum, curLayerInputSize});
            paramsListShape.push_back({hiddenSize * gateNum, hiddenSize});
            if (isbias) {
                paramsListShape.push_back({hiddenSize * gateNum});
                paramsListShape.push_back({hiddenSize * gateNum});
            }
        }
        curLayerInputSize = dScale * hiddenSize;
    }

    std::vector<std::vector<int64_t>> rOutListShape;
    std::vector<std::vector<int64_t>> zOutListShape;
    std::vector<std::vector<int64_t>> nOutListShape;
    std::vector<std::vector<int64_t>> hnOutListShape;
    std::vector<std::vector<int64_t>> hOutListShape;

    if (isTraining) {
        for (int64_t i = 0; i < ldScale; i++) {
            rOutListShape.push_back({timeStep, batchSize, hiddenSize});
            zOutListShape.push_back({timeStep, batchSize, hiddenSize});
            nOutListShape.push_back({timeStep, batchSize, hiddenSize});
            hnOutListShape.push_back({timeStep, batchSize, hiddenSize});
            hOutListShape.push_back({timeStep, batchSize, hiddenSize});
        }
    }

    void* inputDeviceAddr = nullptr;
    std::vector<void*> paramsListDeviceAddr(paramsListShape.size(), nullptr);
    void* outputDeviceAddr = nullptr;
    void* hyDeviceAddr = nullptr;
    void* hxDeviceAddr = nullptr;

    std::vector<void*> rOutDeviceAddr;
    std::vector<void*> zOutDeviceAddr;
    std::vector<void*> nOutDeviceAddr;
    std::vector<void*> hnOutDeviceAddr;
    std::vector<void*> hOutDeviceAddr;

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;
    aclTensor* output = nullptr;
    aclTensor* hy = nullptr;
    aclTensor* hx = nullptr;

    aclTensorList* rOut = nullptr;
    aclTensorList* zOut = nullptr;
    aclTensorList* nOut = nullptr;
    aclTensorList* hnOut = nullptr;
    aclTensorList* hOut = nullptr;

    std::vector<float> inputHostData(GetShapeSize(inputShape), 1.0);
    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &params, 1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outputHostData(GetShapeSize(outputShape), 0.0);
    ret = CreateAclTensor<float>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hyHostData(GetShapeSize(hyShape), 0.0);
    ret = CreateAclTensor<float>(hyHostData, hyShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    bool useHx = false;
    if (useHx) {
        std::vector<float> hxHostData(GetShapeSize(hxShape), 0.5);
        ret = CreateAclTensor<float>(hxHostData, hxShape, &hxDeviceAddr, aclDataType::ACL_FLOAT, &hx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    if (isTraining) {
        rOutDeviceAddr.resize(ldScale, nullptr);
        zOutDeviceAddr.resize(ldScale, nullptr);
        nOutDeviceAddr.resize(ldScale, nullptr);
        hnOutDeviceAddr.resize(ldScale, nullptr);
        hOutDeviceAddr.resize(ldScale, nullptr);

        ret = CreateAclTensorList<float>(rOutListShape, rOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &rOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(zOutListShape, zOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &zOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(nOutListShape, nOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &nOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(hnOutListShape, hnOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &hnOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(hOutListShape, hOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &hOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // 3. Call the first-phase API of aclnnGRU.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGRUGetWorkspaceSize(
        input, params, (useHx ? hx : nullptr), nullptr, isbias, numLayers, dropout, isTraining, bidirection, batchFirst, output, hy,
        isTraining ? rOut : nullptr,
        isTraining ? zOut : nullptr,
        isTraining ? nOut : nullptr,
        isTraining ? hnOut : nullptr,
        isTraining ? hOut : nullptr,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. Call the second-phase API of aclnnGRU.
    ret = aclnnGRU(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRU failed. ERROR: %d\n", ret); return ret);

    // 6. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. Obtain the output values.
    PrintOutResult("output", outputShape, &outputDeviceAddr);
    PrintOutResult("hy", hyShape, &hyDeviceAddr);
    if (isTraining) {
        for (int64_t i = 0; i < ldScale; i++) {
            PrintOutResult("rOut[" + std::to_string(i) + "]", rOutListShape[i], &rOutDeviceAddr[i]);
            PrintOutResult("zOut[" + std::to_string(i) + "]", zOutListShape[i], &zOutDeviceAddr[i]);
            PrintOutResult("nOut[" + std::to_string(i) + "]", nOutListShape[i], &nOutDeviceAddr[i]);
            PrintOutResult("hnOut[" + std::to_string(i) + "]", hnOutListShape[i], &hnOutDeviceAddr[i]);
            PrintOutResult("hOut[" + std::to_string(i) + "]", hOutListShape[i], &hOutDeviceAddr[i]);
        }
    }

    // Release resources.
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(output);
    aclDestroyTensor(hy);
    if (hx != nullptr) {
        aclDestroyTensor(hx);
    }
    if (isTraining) {
        aclDestroyTensorList(rOut);
        aclDestroyTensorList(zOut);
        aclDestroyTensorList(nOut);
        aclDestroyTensorList(hnOut);
        aclDestroyTensorList(hOut);
    }
    aclrtFree(inputDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        aclrtFree(paramsListDeviceAddr[i]);
    }
    aclrtFree(outputDeviceAddr);
    aclrtFree(hyDeviceAddr);
    if (hxDeviceAddr != nullptr) {
        aclrtFree(hxDeviceAddr);
    }
    if (isTraining) {
        for (size_t i = 0; i < rOutDeviceAddr.size(); i++) {
            aclrtFree(rOutDeviceAddr[i]);
            aclrtFree(zOutDeviceAddr[i]);
            aclrtFree(nOutDeviceAddr[i]);
            aclrtFree(hnOutDeviceAddr[i]);
            aclrtFree(hOutDeviceAddr[i]);
        }
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
