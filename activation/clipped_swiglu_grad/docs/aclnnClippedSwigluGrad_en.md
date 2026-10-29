# aclnnClippedSwigluGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T08:55:53.455Z -->

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

- Description: The backward operator of aclnnClippedSwiglu. It computes the gradient of input x, `gradXOut`.

- Calculation Process:
  - Step 1: Axis merging and splitting (merge the axes of x and split it into x0 and x1)
  - Step 2: GroupIndex processing (optional) → Compute the number of valid rows
  - Step 3: Recompute the forward intermediate quantities
  - Step 4: Clamp backward propagation mask computation
  - Step 5: SwiGLU backward gradient computation
  - Step 6: Concatenate gradients for output
  - Step 7: GroupIndex zeroing (optional)

- Formula for axis merging and splitting:

  Merge the axes of x based on the input parameter `dim`. The dimensions after merging are [pre, cut, after], where the cut axis is the axis to be split, and pre and after can be 1. Since the elements along the after axis are stored contiguously and the computation is element-wise, merge the cut axis with the after axis to obtain x with dimensions [pre, cut]. The dimensions of `gradY` after merging are [pre, cut/2].

  Split x based on the `interleaved` parameter:

  When `interleaved` is true, it represents odd-even splitting:

  $$
    \mathbf{x}_0[p, h] = \mathbf{x}[p, 2h], \quad h \in [0, \text{cut}/2)
  $$

  $$
    \mathbf{x}_1[p, h] = \mathbf{x}[p, 2h+1], \quad h \in [0, \text{cut}/2)
  $$

  When `interleaved` is false, it indicates front-back split:

  $$
    H = \text{cut} / 2
  $$

  $$
    \mathbf{x}_0[p, h] = \mathbf{x}[p, h], \quad h \in [0, H)
  $$

  $$
    \mathbf{x}_1[p, h] = \mathbf{x}[p, h + H], \quad h \in [0, H)
  $$

  Where $p$ is the pre-dimension index, and $h$ is the half-width index after splitting.

- GroupIndex processing formula (when `groupIndex` is provided):

  $$
    \text{trunc} = \sum_{g=0}^{G-1} \text{groupIndex}[g]
  $$

  $$
    \text{validRows} = \min(\text{trunc}, \text{pre})
  $$

  Where: $G$ is the number of elements in groupIndex, and all subsequent steps process only the first $\text{validRows}$ rows of data. When `groupIndex` is not provided, $\text{validRows} = \text{pre}$.

- Formula for recomputing forward intermediate quantities:

  $$
    \mathbf{x}_0' = \min(\mathbf{x}_0, l)
  $$

  $$
    \mathbf{x}_1' = \min(\max(\mathbf{x}_1, -l), l)
  $$

  $$
    s = \sigma(\alpha \times \mathbf{x}_0') = \frac{1}{1 + e^{-\alpha \times \mathbf{x}_0'}}
  $$

  Where: $\mathbf{x}_0'$ represents $\mathbf{x}_0$ with only the upper bound clamped, $\mathbf{x}_1'$ represents $\mathbf{x}_1$ with both bounds clamped, $s$ is the sigmoid activation value, $\alpha$ is the `alpha` parameter, and $l$ is the `limit` parameter.

- Clamp backward propagation mask formula:

  $$
    \text{mask}_{x_0} = \mathbb{I}(\mathbf{x}_0 \leq l)
  $$

  $$
    \text{mask}_{x_1} = \mathbb{I}(-l \leq \mathbf{x}_1 \leq l)
  $$

  Where $\mathbb{I}$ is the indicator function, which is 1 if the condition is met and 0 otherwise.

- SwiGLU backward gradient calculation formula:

  $$
    \mathbf{grad}_{x_0} = \mathbf{grad}_{y} \times (\mathbf{x}_1' + \text{bias}) \times s \times (1 + \alpha \times \mathbf{x}_0' \times (1 - s)) \times \text{mask}_{x_0}
  $$

  $$
    \mathbf{grad}_{x_1} = \mathbf{grad}_{y} \times \mathbf{x}_0' \times s \times \text{mask}_{x_1}
  $$

  Where: $\mathbf{grad}_{x_0}$ is the gradient of $\mathbf{x}_0$, $\mathbf{grad}_{x_1}$ is the gradient of $\mathbf{x}_1$, and $\mathbf{grad}_{y}$ is `gradY`.

- Gradient concatenation output formula:

  When `interleaved` is true:

  $$
    \mathbf{grad}_{x}[p, 2h] = \mathbf{grad}_{x_0}[p, h]
  $$

  $$
    \mathbf{grad}_{x}[p, 2h+1] = \mathbf{grad}_{x_1}[p, h]
  $$

  When `interleaved` is false:

  $$
    \mathbf{grad}_{x}[p, h] = \mathbf{grad}_{x_0}[p, h], \quad h \in [0, H)
  $$

  $$
    \mathbf{grad}_{x}[p, h+H] = \mathbf{grad}_{x_1}[p, h], \quad h \in [0, H)
  $$

  where $\mathbf{grad}_{x}$ is `gradXOut`.

- GroupIndex zeroing formula (when `groupIndex` is provided):

  $$
    \mathbf{grad}_{x}[p, :] = \mathbf{grad}_{x}[p, :] \times \mathbb{I}(p < \text{validRows})
  $$

  Finally, reshape the output tensor `gradXOut` so that the number of its dimensions is the same as that of `x` before axis merging, and the size of each dimension is the same as that of `x`.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnClippedSwigluGradGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation flow, and then call the "aclnnClippedSwigluGrad" API to perform the computation.

```Cpp
aclnnStatus aclnnClippedSwigluGradGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *groupIndexOptional,
    int64_t          dim,
    double           alpha,
    double           limit,
    double           bias,
    bool             interleaved,
    const aclTensor *gradXOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnClippedSwigluGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnClippedSwigluGradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 270px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 90px">
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
      <td>gradY in the formula, which represents the gradient of forward output y.</td>
      <td>The dimensions must be greater than 0, and the size in the dimension corresponding to parameter dim must be half of that of x. Other dimensions must be consistent with those of x. The data type must be consistent with those of x and gradXOut. gradY does not support empty tensors, and the input does not support containing ±inf or nan.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>x in the formula, which is the input of the forward operator. It is used to recompute forward intermediate values.</td>
      <td>The dimension must be greater than 0 and must be an even number in the dimension corresponding to the input parameter dim. Empty tensors are not supported, and the input does not support ±inf or nan.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Input groupIndex in the formula, which represents grouping.</td>
      <td>Supports null pointers. If the pointer is not null, the tensor must be 1D, and each element must be greater than or equal to 0. The i-th element represents the number of x batches to be processed by the i-th group, and the sum of elements must be less than or equal to the product of all dimensions before the dim dimension prior to axis merging. Empty tensors are not supported, and the input does not support ±inf or nan.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>Input dim in the formula, representing the dimension index for axis merging and splitting of x.</td>
      <td>Value range: [-x.dim(), x.dim()-1].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha (double)</td>
      <td>Input</td>
      <td>Input alpha in the formula, representing the parameter used by the SwiGlu variant.</td>
      <td>Recommended to be 1.702.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>limit (double)</td>
      <td>Input</td>
      <td>Input limit in the formula, representing the threshold value used by the SwiGlu variant.</td>
      <td>Must be greater than 0, recommended to be 7.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias (double)</td>
      <td>Input</td>
      <td>Input bias in the formula, representing the bias parameter used by the SwiGlu variant.</td>
      <td>Recommended to be 1.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>interleaved (bool)</td>
      <td>Input</td>
      <td>Input interleaved in the formula, representing whether x is split by odd/even.</td>
      <td>true indicates that x is split by odd/even; false indicates that x is split into front and back halves.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut (aclTensor*)</td>
      <td>Output</td>
      <td>The output gradXOut in the formula.</td>
      <td>Does not support empty pointers. The shape is consistent with x. Does not support empty tensors.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
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
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

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
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Required parameters gradY/x/gradXOut are nullptr.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is out of the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shapes of multiple input tensors do not match, or input attributes are out of the valid range (see Parameters for details).</td>
    </tr>
  </tbody></table>

## aclnnClippedSwigluGrad

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
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnClippedSwigluGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnClippedSwigluGrad defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clipped_swiglu_grad.h"

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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // x shape [2, 32], grad_y shape [2, 16], grad_x shape [2, 32]
    std::vector<int64_t> xShape = {2, 32};
    std::vector<int64_t> gradYShape = {2, 16};
    std::vector<int64_t> gradXOutShape = {2, 32};
    void* xDeviceAddr = nullptr;
    void* gradYDeviceAddr = nullptr;
    void* gradXOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* gradY = nullptr;
    aclTensor* gradXOut = nullptr;

    std::vector<float> xHostData(64, 1.0f);
    std::vector<float> gradYHostData(32, 1.0f);
    std::vector<float> gradXOutHostData(64, 0.0f);

    int dim = -1;
    float alpha = 1.702;
    float limit = 7.0;
    float bias = 1.0;
    bool interleaved = true;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXOutHostData, gradXOutShape, &gradXOutDeviceAddr, aclDataType::ACL_FLOAT, &gradXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnClippedSwigluGradGetWorkspaceSize(gradY, x, nullptr, dim, alpha, limit, bias, interleaved, gradXOut,
                                                 &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClippedSwigluGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnClippedSwigluGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClippedSwigluGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(gradXOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXOutDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    aclDestroyTensor(x);
    aclDestroyTensor(gradY);
    aclDestroyTensor(gradXOut);
    aclrtFree(xDeviceAddr);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(gradXOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
