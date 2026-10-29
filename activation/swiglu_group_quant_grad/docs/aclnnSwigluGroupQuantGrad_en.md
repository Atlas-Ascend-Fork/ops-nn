# aclnnSwigluGroupQuantGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T15:56:35.280Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: The SwigluGroupQuantGrad operator implements the backward gradient computation of group quantization for the SwiGLU activation function. It computes the input gradient `grad_x` and the weight gradient `grad_weight`.
- Operator support scope: Supports the MoE use case (groupIndex passed in) and the non-MoE use case (groupIndex left empty). Supports the optional Clamp backward propagation mask and the optional Weight gradient computation.
- Calculation Process:
  - Step 0: GroupIndex processing (optional) → Compute trunc
  - Step 1: Input splitting (split x into x0 and x1)
  - Step 2: Clamp processing (optional)
  - Step 3: SwiGLU backward propagation computation
  - Step 4: Weight gradient computation (optional)
  - Step 5: Gradient concatenation output

- GroupIndex processing formula in the MoE use case:
  $$
    \text{trunc} = \sum_{g=0}^{G-1} \text{groupIndex}[g]
  $$
  where $G$ is the number of MoE expert groups, and all subsequent steps process only the first $\text{trunc}$ rows of data.

- Input splitting formula:
  $$
    \mathbf{x}_0[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)
  $$

  $$
    \mathbf{x}_1[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)
  $$

- Clamp processing formula (when clamp_limit > 0):
  $$
    \mathbf{x}_0'[t, h] = \min(\mathbf{x}_0[t, h], c)
  $$

  $$
    \mathbf{x}_1'[t, h] = \min(\max(\mathbf{x}_1[t, h], -c), c)
  $$
  where $c$ is `clamp_limit`.

- SiLU gradient formula:
  $$
    \frac{d\text{SiLU}}{d\mathbf{x}_0'} = \sigma(\mathbf{x}_0') \cdot \left(1 + \mathbf{x}_0' \cdot (1 - \sigma(\mathbf{x}_0'))\right)
  $$
  where: $\sigma(\mathbf{x}_0') = \frac{1}{1 + e^{-\mathbf{x}_0'}}$

- Input gradient computation formula:
  $$
    \mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \mathbf{x}_1'[t, h] \cdot \frac{d\text{SiLU}}{d\mathbf{x}_0'}[t, h]
  $$

  $$
    \mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \text{SiLU}(\mathbf{x}_0'[t, h])
  $$
  where: if weight is provided, then $\mathbf{grad}_{y_0} = \mathbf{grad}_{y} \cdot \mathbf{weight}$; if weight is not provided, then $\mathbf{grad}_{y_0} = \mathbf{grad}_{y}$

- Weight gradient computation formula (optional):
   $$
     \mathbf{grad}_{\text{weight}}[t] = \sum_{h=0}^{H-1} \mathbf{grad}_{y}[t, h] \cdot \mathbf{y}_{\text{origin}}[t, h]
   $$
   where: $\mathbf{y}_{\text{origin}}$ is the raw activation value output of the SwiGLU forward propagation, summed along the last dimension (H dimension).

   $$
     \mathbf{grad}_{\text{weight}}[t] = \mathbf{grad}_{\text{weight}}[t] \cdot \mathbb{I}(t < \text{trunc})
   $$

- Clamp backward propagation mask formula (when clamp_limit > 0):
  $$
    \mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{x_0}[t, h] \cdot \mathbb{I}(\mathbf{x}_0[t, h] < c)
  $$

  $$
    \mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{x_1}[t, h] \cdot \mathbb{I}(-c < \mathbf{x}_1[t, h] < c)
  $$
  where $\mathbb{I}$ is the indicator function.

- Gradient concatenation and GroupIndex processing formula:
  $$
    \mathbf{grad}_x[t, h] = \begin{cases}
    \mathbf{grad}_{x_0}[t, h] & h \in [0, H) \\
    \mathbf{grad}_{x_1}[t, h-H] & h \in [H, 2H)
    \end{cases}
  $$

  $$
    \mathbf{grad}_x[t, :] = \mathbf{grad}_x[t, :] \cdot \mathbb{I}(t < \text{trunc})
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSwigluGroupQuantGradGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnSwigluGroupQuantGrad" API to perform computation.

```Cpp
aclnnStatus aclnnSwigluGroupQuantGradGetWorkspaceSize(
    const aclTensor   *gradY,
    const aclTensor   *x,
    const aclTensor   *weightOptional,
    const aclTensor   *yOriginOptional,
    const aclTensor   *groupIndexOptional,
    double             clampLimit,
    const aclTensor   *gradXOut,
    const aclTensor   *gradWeightOutOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuantGrad(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnSwigluGroupQuantGradGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>gradY (aclTensor*)</td>
      <td>Input</td>
      <td>Output gradient tensor. The gradient comes from the downstream layer.</td>
      <td><ul><li>shape=[..., H]. Common shapes are [T, H] or [B, S, H].</li><li>T is the number of tokens, B is the batch size, S is the sequence length, and H is the hidden size.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2~8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor for forward propagation.</td>
      <td><ul><li>shape=[..., 2H]. Common shapes are [T, 2H] or [B, S, 2H]. The last dimension must be an even number.</li><li>H in the last dimension corresponds to H in gradY.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2~8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input (optional)</td>
      <td>MoE weight tensor.</td>
      <td><ul><li>If weight is not empty, the number of elements must equal the product of the dimensions of x or gradY excluding the last dimension. Common shapes are [T] or [B, S].</li><li>If weight is provided, yOrigin must also be provided to compute gradWeight.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1~8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yOriginOptional (aclTensor*)</td>
      <td>Input (optional)</td>
      <td>Raw activation output of SwiGLU forward propagation.</td>
      <td><ul><li>shape=[..., H]. Common shapes are [T, H] or [B, S, H], and the shape must be the same as that of gradY.</li><li>If weight is provided, yOrigin must also be provided to compute gradWeight.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2~8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input (optional)</td>
      <td>GroupIndex tensor for dynamic kernel allocation.</td>
      <td><ul><li>shape=[G], dtype=INT64.</li><li>G is the number of MoE expert groups.</li><li>Elements in groupIndex are group token counts in count mode.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>clampLimit (float)</td>
      <td>Input</td>
      <td>Clamp threshold.</td>
      <td><ul><li>The value range is -1.0 or >0.0.</li><li>clampLimit=-1.0 indicates that the Clamp backward propagation mask is disabled; when the mask is enabled, clampLimit must be >0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut (aclTensor*)</td>
      <td>Output</td>
      <td>Input gradient tensor.</td>
      <td><ul><li>shape=[..., 2H]. Common shapes are [T, 2H] or [B, S, 2H]. Same as x.</li><li>Data type is the same as gradY/x.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2~8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradWeightOutOptional (aclTensor*)</td>
      <td>Output (optional)</td>
      <td>Weight gradient tensor.</td>
      <td><ul><li>Output when weight is provided. The shape is the same as that of the input weight.</li><li>Data type is FLOAT.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1~8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
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

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The required parameter gradY/x/gradXOut is nullptr.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>161002</td>
      <td>The data type of the input or output is out of the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shapes of multiple input tensors do not match, or an input attribute is out of the value range (see Parameters).</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuantGrad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 173px">
  <col style="width: 124px">
  <col style="width: 852px">
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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnSwigluGroupQuantGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSwigluGroupQuantGrad defaults to a deterministic implementation.

- Input shape constraints:
  - The last dimension of x must be an even number ($2H$).
  - The last dimension of gradY is $H$, corresponding to half of the last dimension of x.
  - The shapes of the first n-1 dimensions of gradY and x must be the same.

- Optional parameter constraints:
  - If weight is provided, yOrigin must also be provided to compute gradWeight.
  - The number of elements in weight must equal the product of the element counts of x or gradY excluding the last dimension.
  - The shape of yOrigin must be the same as that of gradY.

- Data type constraints:
  - The data types of gradY, x, yOrigin, and gradX must be the same (FLOAT, FLOAT16, or BFLOAT16).
  - weight and gradWeight must be of the FLOAT type.
  - groupIndex must be of the INT64 type.

- Clamp constraints:
  - The value range of clampLimit is -1.0 or >0.0.
  - clampLimit=-1.0 indicates that the Clamp backward propagation mask is disabled. When the mask is enabled, clampLimit must be >0.0.

- Specification constraints:

  | Specification Item | Specification | Specification |
  |--------|------|----------|
  | B | 1~31 | - |
  | S | 0~128K | - |
  | H | 512, 768, 1024, 1536, 1792, 2048, 2560, 4096 | - |

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_quant_grad.h"

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
                    aclDataType dataType, aclTensor** tensor)
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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorWithValue(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                             aclTensor** tensor, T value)
{
    int64_t shapeSize = GetShapeSize(shape);
    std::vector<T> hostData(shapeSize, value);
    return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> gradYShape = {512, 512};
    std::vector<int64_t> xShape = {512, 1024};
    std::vector<int64_t> weightShape = {512};
    std::vector<int64_t> yOriginShape = {512, 512};
    std::vector<int64_t> groupIndexShape = {256};
    std::vector<int64_t> gradXShape = {512, 1024};
    std::vector<int64_t> gradWeightShape = {512};

    void* gradYDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* yOriginDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;

    aclTensor* gradYTensor = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* weightTensor = nullptr;
    aclTensor* yOriginTensor = nullptr;
    aclTensor* groupIndexTensor = nullptr;
    aclTensor* gradXTensor = nullptr;
    aclTensor* gradWeightTensor = nullptr;

    int64_t gradYSize = GetShapeSize(gradYShape);
    std::vector<float> gradYHostData(gradYSize, 1.0f);
    for (int64_t i = 0; i < gradYSize; i++) {
        gradYHostData[i] = static_cast<float>(i % 10) * 0.1f;
    }

    int64_t xSize = GetShapeSize(xShape);
    std::vector<float> xHostData(xSize, 1.0f);
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = static_cast<float>((i % 20) - 10) * 0.5f;
    }

    int64_t weightSize = GetShapeSize(weightShape);
    std::vector<float> weightHostData(weightSize, 1.0f);
    for (int64_t i = 0; i < weightSize; i++) {
        weightHostData[i] = static_cast<float>((i % 5) + 1) * 0.2f;
    }

    int64_t yOriginSize = GetShapeSize(yOriginShape);
    std::vector<float> yOriginHostData(yOriginSize, 1.0f);
    for (int64_t i = 0; i < yOriginSize; i++) {
        yOriginHostData[i] = static_cast<float>((i % 8) + 1) * 0.3f;
    }

    int64_t groupIndexSize = GetShapeSize(groupIndexShape);
    std::vector<int64_t> groupIndexHostData(groupIndexSize, 0);
    int64_t groupStride = 512 / 256;
    for (int64_t i = 0; i < groupIndexSize; i++) {
        groupIndexHostData[i] = i * groupStride;
    }

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradYTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weightTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(yOriginHostData, yOriginShape, &yOriginDeviceAddr, aclDataType::ACL_FLOAT16, &yOriginTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64,
                          &groupIndexTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<float>(gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT16, &gradXTensor, 0.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<float>(gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT,
                                          &gradWeightTensor, 0.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float clampLimit = 1.0f;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnSwigluGroupQuantGradGetWorkspaceSize(gradYTensor, xTensor, weightTensor, yOriginTensor, groupIndexTensor,
                                                    clampLimit, gradXTensor, gradWeightTensor, &workspaceSize,
                                                    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSwigluGroupQuantGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto gradXResultSize = GetShapeSize(gradXShape);
    std::vector<float> gradXResultData(gradXResultSize, 0);
    ret = aclrtMemcpy(gradXResultData.data(), gradXResultData.size() * sizeof(float), gradXDeviceAddr,
                      gradXResultSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradX result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("gradX output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < gradXResultSize; i++) {
        LOG_PRINT("gradX[%ld] = %f\n", i, gradXResultData[i]);
    }

    auto gradWeightResultSize = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightResultData(gradWeightResultSize, 0);
    ret = aclrtMemcpy(gradWeightResultData.data(), gradWeightResultData.size() * sizeof(float), gradWeightDeviceAddr,
                      gradWeightResultSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradWeight result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("gradWeight output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < gradWeightResultSize; i++) {
        LOG_PRINT("gradWeight[%ld] = %f\n", i, gradWeightResultData[i]);
    }

    aclDestroyTensor(gradYTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(weightTensor);
    aclDestroyTensor(yOriginTensor);
    aclDestroyTensor(groupIndexTensor);
    aclDestroyTensor(gradXTensor);
    aclDestroyTensor(gradWeightTensor);

    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(yOriginDeviceAddr);
    aclrtFree(groupIndexDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
