# aclnnMultiScaleDeformableAttentionGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:00:16.866Z -->

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

  The forward operator of MultiScaleDeformableAttention mainly traverses different sampling points of feature maps of different sizes through parameters such as sample locations, attention weights, mapped value features, the starting index positions of multi-scale features, and the spatial sizes of multi-scale feature maps (to convert sample locations from normalized values to absolute positions). The backward operator computes the gradients of the inputs based on the contribution of the forward inputs to the outputs and the initial gradients.
- Formula:

    Let $b \in [0, bs)$, $q \in [0, num\_queries)$, $h \in [0, num\_heads)$, $\ell \in [0, num\_levels)$, $p \in [0, num\_points)$, and $c \in [0, channels)$ be the indices of batch, query, head, feature map, sampling point, and channel, respectively. The height and width of the $\ell$-th level feature map are $H_\ell$ and $W_\ell$, i.e., $\mathrm{spatialShape}[\ell] = (H_\ell, W_\ell)$; the flattened index of the pixel $(y, x)$ in the $\ell$-th level feature map along the $num\_keys$ dimension of $\mathrm{value}$ is $k_\ell(y, x) = \mathrm{levelStartIndex}[\ell] + y \cdot W_\ell + x$. The sampling point $\mathrm{location}[b, q, h, \ell, p] = (u, v)$ (the 0th and 1st elements in the last dimension correspond to the $x$ and $y$ directions, respectively) is mapped to the pixel coordinates $x = u \cdot W_\ell - 0.5$ and $y = v \cdot H_\ell - 0.5$; let $x_0 = \lfloor x \rfloor$, $x_1 = x_0 + 1$, $y_0 = \lfloor y \rfloor$, $y_1 = y_0 + 1$, $\alpha_x = x - x_0$, and $\alpha_y = y - y_0$; the bilinear interpolation weights are $w_{00} = (1-\alpha_y)(1-\alpha_x)$, $w_{10} = (1-\alpha_y)\alpha_x$, $w_{01} = \alpha_y(1-\alpha_x)$, and $w_{11} = \alpha_y\alpha_x$; and the feature vectors of the four neighboring points are $V_{y_i, x_j} := \mathrm{value}[b,\; k_\ell(y_i, x_j),\; h,\; :]$ ($i, j \in \{0, 1\}$, with out-of-bounds neighboring points treated as 0).

    - The forward output is:

      $$
      \mathrm{output}[b,\; q,\; h \times channels + c] =
      \sum_{\ell=0}^{num\_levels-1} \sum_{p=0}^{num\_points-1}
      \mathrm{attnWeight}[b, q, h, \ell, p] \cdot
      \sum_{i,j \in \{0,1\}} w_{ij} \cdot V_{y_i, x_j}[c]
      $$

    - Based on the gradient $\mathrm{gradOutput}$ (shape $(bs, num\_queries, num\_heads \times channels)$, with the last dimension laid out as $h \times channels + c$), the backward operator computes the gradients of the three inputs $\mathrm{value}$, $\mathrm{location}$, and $\mathrm{attnWeight}$. Denote the slice of head $h$ in $\mathrm{gradOutput}$ as $G_{b,q,h,c} := \mathrm{gradOutput}[b,\; q,\; h \times channels + c]$.
      1. Expand by attention weights:

         $$
         \tilde{G}_{b,q,h,\ell,p,c} = \mathrm{attnWeight}[b, q, h, \ell, p] \cdot G_{b,q,h,c}
         $$

      2. Compute $\mathrm{gradValue}$, the gradient of $\mathrm{value}$ (shape consistent with value): For each sampling point $(b, q, h, \ell, p)$, accumulate its expanded gradient to the four neighboring positions according to the bilinear weights (accumulate over all $q$ and $p$; out-of-bounds neighbors are not accumulated):

         $$
         \begin{aligned}
         \mathrm{gradValue}[b,\; k_\ell(y_0, x_0),\; h,\; c] &+= w_{00} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_0, x_1),\; h,\; c] &+= w_{10} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_1, x_0),\; h,\; c] &+= w_{01} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_1, x_1),\; h,\; c] &+= w_{11} \cdot \tilde{G}_{b,q,h,\ell,p,c}
         \end{aligned}
         $$

      3. Compute $\mathrm{gradAttnWeight}$, the gradient of $\mathrm{attnWeight}$ (shape consistent with attnWeight), which is the inner product of the output gradient and the sampled feature vectors:

         $$
         \mathrm{gradAttnWeight}[b, q, h, \ell, p] =
         \sum_{c} G_{b,q,h,c} \cdot
         \left( \sum_{i,j \in \{0,1\}} w_{ij} \cdot V_{y_i, x_j}[c] \right)
         $$

      4. Compute $\mathrm{gradLocation}$, the gradient of $\mathrm{location}$ (shape consistent with location). The gradient of the pixel coordinates $(x, y)$ of the sampling point is obtained from the first-order partial derivatives of bilinear interpolation:

         $$
         \nabla x_{b,q,h,\ell,p} =
         \sum_{c} \tilde{G}_{b,q,h,\ell,p,c} \,
         \big[ (V_{y_0, x_1}[c] - V_{y_0, x_0}[c])(1-\alpha_y)
             + (V_{y_1, x_1}[c] - V_{y_1, x_0}[c])\alpha_y <!-- __BS_129 -->
         $$

         $$
         \nabla y_{b,q,h,\ell,p} =
         \sum_{c} \tilde{G}_{b,q,h,\ell,p,c} \,
         \big[ (V_{y_1, x_0}[c] - V_{y_0, x_0}[c])(1-\alpha_x)
             + (V_{y_1, x_1}[c] - V_{y_0, x_1}[c])\alpha_x \big]
         $$

      5. Based on $x = u \cdot W_\ell - 0.5$ and $y = v \cdot H_\ell - 0.5$, scale the gradient of the normalized regression coordinates by the chain rule and write the result to $\mathrm{gradLocation}$ (in the last dimension, the 0th and 1st elements correspond to the $x$ and $y$ directions respectively, consistent with location):

         $$
         \mathrm{gradLocation}[b, q, h, \ell, p] = (\nabla u, \nabla v), \qquad
         \nabla u = W_\ell \cdot \nabla x_{b,q,h,\ell,p}, \qquad
         \nabla v = H_\ell \cdot \nabla y_{b,q,h,\ell,p}
         $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation flow. Then, call the "aclnnMultiScaleDeformableAttentionGrad" API to perform the computation.

```Cpp
aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(
    const aclTensor* value,
    const aclTensor* spatialShape,
    const aclTensor* levelStartIndex,
    const aclTensor* location,
    const aclTensor* attnWeight,
    const aclTensor* gradOutput,
    aclTensor*       gradValue,
    aclTensor*       gradLocation,
    aclTensor*       gradAttnWeight,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnMultiScaleDeformableAttentionGrad(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 100%"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
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
      <td>value</td>
      <td>Input</td>
      <td>Feature values of the feature map. Corresponding to `value` in the formula.</td>
      <td>The shape is (bs, num_keys, num_heads, channels), where: <ul><li>bs is the batch size.</li><li>num_keys is the size of the feature map.</li><li>num_heads is the quantity of heads.</li><li>channels is the dimension of the feature map.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>spatialShape</td>
      <td>Input</td>
      <td>Stores the height and width of the feature map at each scale. Corresponding to `spatialShape` in the formula.</td>
      <td>shape is (num_levels, 2), where: <ul><li>num_levels is the quantity of feature maps.</li><li>2 represents H and W respectively.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>levelStartIndex</td>
      <td>Input</td>
      <td>Start index of each feature map. It corresponds to `levelStartIndex` in the corresponding formula.</td>
      <td>shape is (num_levels).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>location</td>
      <td>Input</td>
      <td>Sampling point location tensor, which stores the coordinate position of each sampling point. Corresponds to `location` in the formula.</td>
      <td>Shape (bs, num_queries, num_heads, num_levels, num_points, 2), where:<ul><li>num_queries is the quantity of queries.</li><li>num_points is the quantity of sampling points.</li><li>2 represents y and x respectively.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnWeight</td>
      <td>Input</td>
      <td>Weight tensor of sampling points. Corresponds to `​attnWeight` in the corresponding formula.</td>
      <td>Shape: (bs, num_queries, num_heads, num_levels, num_points).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient of the forward output, and also the initial gradient of the backward operator. Corresponds to `gradOutput` in the formula.</td>
      <td>Shape: (bs, num_queries, num_heads*channels).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradValue</td>
      <td>Output</td>
      <td>Gradient of the input value. Corresponds to `gradValue` in the formula.</td>
      <td>shape remains consistent with value</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradLocation</td>
      <td>Output</td>
      <td>Gradient of the input location. Corresponds to `gradLocation` in the corresponding formula.</td>
      <td>shape remains consistent with location</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradAttnWeight</td>
      <td>Output</td>
      <td>Corresponding gradient of the input attnWeight. It is `gradAttnWeight` in the corresponding formula.</td>
      <td>Shape remains consistent with attnWeight</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the Device side-applied workspace.</td>
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

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The input or output is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of the input or output is not supported.</td>
    </tr>
    <tr>
      <td>The constraints in the Constraints section are not met.</td>
    </tr>
  </tbody>
  </table>

## aclnnMultiScaleDeformableAttentionGrad

- **Parameters**:

  <table><thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the Device side-applied workspace memory.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the Device side-applied workspace, obtained by calling the first-phase API aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>opExecutor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>
- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation
  - aclnnMultiScaleDeformableAttentionGrad defaults to a non-deterministic implementation. You can call aclrtCtxSetSysParamOpt to enable deterministic computation.

- The number of channels satisfies channels % 8 = 0 and channels <= 256.
- The number of queries satisfies num_queries < 500000.
- The number of feature maps: num_levels <= 16.
- The number of heads: num_heads <= 16.
- The number of sampling points: num_points <= 16.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multi_scale_deformable_attention_grad.h"

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
    // Boilerplate: resource initialization
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
    // Call aclrtMalloc to apply for device-side memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
       strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1.(Boilerplate) Initialize device/stream, refer to the acl external API list
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as needed
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs. Custom construction is required based on the API

    std::vector<int64_t> valueShape = {1, 1, 1, 8};
    std::vector<int64_t> spatialShapeShape = {1, 2};
    std::vector<int64_t> levelStartIndexShape = {1};
    std::vector<int64_t> locationShape = {1, 32, 1, 1, 1, 2};
    std::vector<int64_t> attnWeightShape = {1, 32, 1, 1, 1};
    std::vector<int64_t> gradOutputShape = {1, 32, 8};
    std::vector<int64_t> gradValueShape = {1, 1, 1, 8};
    std::vector<int64_t> gradLocationShape = {1, 32, 1, 1, 1, 2};
    std::vector<int64_t> gradAttnWeightShape = {1, 32, 1, 1, 1};
    void* valueDeviceAddr = nullptr;
    void* spatialShapeDeviceAddr = nullptr;
    void* levelStartIndexDeviceAddr = nullptr;
    void* locationDeviceAddr = nullptr;
    void* attnWeightDeviceAddr = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    void* gradValueDeviceAddr = nullptr;
    void* gradLocationDeviceAddr = nullptr;
    void* gradAttnWeightDeviceAddr = nullptr;
    aclTensor* value = nullptr;
    aclTensor* spatialShape = nullptr;
    aclTensor* levelStartIndex = nullptr;
    aclTensor* location = nullptr;
    aclTensor* attnWeight = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* gradValue = nullptr;
    aclTensor* gradLocation = nullptr;
    aclTensor* gradAttnWeight = nullptr;
    std::vector<float> valueHostData = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int32_t> spatialShapeHostData = {1, 1};
    std::vector<float> levelStartIndexHostData = {0};
    std::vector<float> gradValueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gradLocationHostData(GetShapeSize(gradLocationShape), 0);
    std::vector<float> gradAttnWeightHostData(GetShapeSize(gradAttnWeightShape), 0);
    std::vector<float> locationHostData(GetShapeSize(locationShape), 0);
    std::vector<float> attnWeightHostData(GetShapeSize(attnWeightShape), 1);
    std::vector<float> gradOutputHostData(GetShapeSize(gradOutputShape), 1);

    // value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the spatialShape aclTensor
    ret = CreateAclTensor(spatialShapeHostData, spatialShapeShape, &spatialShapeDeviceAddr, aclDataType::ACL_INT32, &spatialShape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create levelStartIndex aclTensor.
    ret = CreateAclTensor(levelStartIndexHostData, levelStartIndexShape, &levelStartIndexDeviceAddr, aclDataType::ACL_INT32, &levelStartIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create location aclTensor.
    ret = CreateAclTensor(locationHostData, locationShape, &locationDeviceAddr, aclDataType::ACL_FLOAT, &location);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create attnWeight aclTensor.
    ret = CreateAclTensor(attnWeightHostData, attnWeightShape, &attnWeightDeviceAddr, aclDataType::ACL_FLOAT, &attnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // gradOutput aclTensor
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradValue aclTensor.
    ret = CreateAclTensor(gradValueHostData, gradValueShape, &gradValueDeviceAddr, aclDataType::ACL_FLOAT, &gradValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradLocation aclTensor.
    ret = CreateAclTensor(gradLocationHostData, gradLocationShape, &gradLocationDeviceAddr, aclDataType::ACL_FLOAT, &gradLocation);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradAttnWeight aclTensor.
    ret = CreateAclTensor(gradAttnWeightHostData, gradAttnWeightShape, &gradAttnWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradAttnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMultiScaleDeformableAttentionGrad.
    ret = aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(value, spatialShape, levelStartIndex, location, attnWeight, gradOutput, gradValue, gradLocation, gradAttnWeight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMultiScaleDeformableAttentionGrad.
    ret = aclnnMultiScaleDeformableAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttentionGrad failed. ERROR: %d\n", ret); return ret);
    // 4.(boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values, and copy the result in device memory to the host. Modify according to the specific API definition.
    auto size = GetShapeSize(gradValueShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradValueDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify according to the specific API definition.
    aclDestroyTensor(value);
    aclDestroyTensor(spatialShape);
    aclDestroyTensor(levelStartIndex);
    aclDestroyTensor(location);
    aclDestroyTensor(attnWeight);
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradValue);
    aclDestroyTensor(gradLocation);
    aclDestroyTensor(gradAttnWeight);

    // 7. Release device resources. Modify according to the specific API definition.
    aclrtFree(valueDeviceAddr);
    aclrtFree(spatialShapeDeviceAddr);
    aclrtFree(levelStartIndexDeviceAddr);
    aclrtFree(locationDeviceAddr);
    aclrtFree(attnWeightDeviceAddr);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradValueDeviceAddr);
    aclrtFree(gradLocationDeviceAddr);
    aclrtFree(gradAttnWeightDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
