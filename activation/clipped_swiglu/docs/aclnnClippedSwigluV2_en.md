# aclnnClippedSwigluV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T08:42:28.122Z -->

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

- Description: Clipped Swish gated linear unit activation function, which performs SwiGlu computation on x. Compared with aclnnClippedSwiglu, this API adds an input parameter: clamp_mode, to support use cases where some new models require the clamp operation to be performed after the silu activation.

- Formula:

  For a given input tensor x with dimensions [a,b,c,d,e,f,g...], aclnnClippedSwigluV2 performs the following computation:

  1. Concatenate the axes of x based on the input parameter dim. After concatenation, the dimensions become [pre, cut, after]. The cut axis is the axis that needs to be split into two tensors after concatenation. The split method can be front-back split or odd-even split. pre and after can be 1. For example, when dim is 3, the dimensions of x after concatenation are [a*b*c, d, e*f*g*…]. In addition, since the elements of the after axis are stored contiguously and the computation is element-wise, the cut axis and the after axis are merged, resulting in x with dimensions [pre, cut].

  2. Filters the pre axis of x based on the input parameter group_index. The formula is as follows:

     $$
     sum = \text{Sum}(group\_index)
     $$

     $$
     x = x[ : sum, : ]
     $$

where sum represents the sum of all elements in group_index. This step is skipped when group_index is not provided.

  3. Split x according to the input parameter interleaved. The formula is as follows:

     When interleaved is true, an odd-even split is performed:

     $$
     A = x[ : , : : 2]
     $$

     $$
     B = x[ : , 1 : : 2]
     $$

     When interleaved is false, a front-back split is performed:

     $$
     h = x.shape[1] // 2
     $$

     $$
     A = x[ : , : h]
     $$

     $$
     B = x[ : , h : ]
     $$
  4. Perform variant SwiGlu computation based on the input parameters alpha, limit, bias, and clamp_mode. The formula is as follows:

     When clampMode is 0, the clamp operation is performed before silu:

     $$
     A = A.clamp(min=None, max=limit)
     $$

     $$
     B = B.clamp(min=-limit, max=limit)
     $$

     $$
     y\_glu = A * sigmoid(alpha * A)
     $$

     $$
     y = y\_glu * (B + bias)
     $$

     When clampMode is 1, the clamp operation is performed after silu:

     $$
     y\_glu = A * sigmoid(A)
     $$

     $$
     y\_glu = y\_glu.clamp(min=None, max=limit)
     $$

     $$
     B = B.clamp(min=-limit, max=limit)
     $$

     $$
     y = y\_glu * B
     $$

  5. Reshape the output tensor y to have the same number of dimensions as x before concatenation, with the size on the dim axis being half of that of x, and other dimensions identical to x.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnClippedSwigluV2GetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnClippedSwigluV2" to perform the computation.

```Cpp
aclnnStatus aclnnClippedSwigluV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *groupIndexOptional,
    int64_t          dim,
    double           alpha,
    double           limit,
    double           bias,
    bool             interleaved,
    int64_t          clampMode,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnClippedSwigluV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnClippedSwigluV2GetWorkspaceSize

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
      <td>x(aclTensor*)</td>
      <td>Input</td>
      <td>Input x in the formula.</td>
      <td><ul><li>Does not support empty pointers. The dimension must be greater than 0, and the shape must be an even number on the dimension corresponding to input parameter dim.</li><li>Supports empty tensors.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Input group_index in the formula, representing the grouping case.</td>
      <td><ul><li>Supports null pointers.</li><li>If not a null pointer, it must be 1D, and its elements must be greater than or equal to 0. The i-th element represents the number of x batches to be processed by the i-th group.</li><li>The number of supported groups is determined by the UB space size.</li><li>Does not support empty tensor.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>dim in the formula, indicating the dimension index along which axis merging and splitting are performed on x.</td>
      <td>Value range: [-x.dim(), x.dim()-1].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha (double)</td>
      <td>Input</td>
      <td>Input alpha in the formula, represents the parameter used by variant SwiGlu.</td>
      <td>The recommended value is 1.702.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>limit (double)</td>
      <td>Input</td>
      <td>Input limit in the formula, representing the threshold value used by variant SwiGlu.</td>
      <td>Must be greater than 0. 7.0 is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias (double)</td>
      <td>Input</td>
      <td>Input bias in the formula, which represents the bias parameter used by variant SwiGlu.</td>
      <td>The recommended value is 1.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>interleaved (bool)</td>
      <td>Input</td>
      <td>Input interleaved in the formula, which indicates whether x is split in the odd-even manner.</td>
      <td>Set to true to perform an odd-even split on x; set to false to perform a front-back split on x.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampMode (int64_t)</td>
      <td>Input</td>
      <td>Input clampMode in the formula, indicating the execution order of the clamp operation and the silu operation.</td>
      <td>Set to 1 to move the clamp operation on A after the silu activation, that is, clamp the result of silu(A); set to 0 to perform the clamp operation on A before the silu activation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>The output y in the formula.</td>
      <td><ul><li>Does not support empty pointers. The shape is half of x on the dimension corresponding to the input dim, and the same as x on other dimensions.</li><li>Supports empty tensors.</li></ul></td>
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

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>The passed-in x or out is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="5">561002</td>
      <td>The shape of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data type or shape size of groupIndexOptional is not in the supported range.</td>
    </tr>
    <tr>
      <td>dim is not within the dimension range supported by input x.</td>
    </tr>
    <tr>
      <td>limit is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>clampMode is neither 0 nor 1.</td>
    </tr>
  </tbody>
  </table>

## aclnnClippedSwigluV2

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
      <td>Address of the workspace memory applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnClippedSwigluV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnClippedSwigluV2 defaults to a deterministic implementation. Non-deterministic implementation is not currently supported; even if deterministic computation is configured, the configuration does not take effect.
- The portion of output out beyond the sum of all elements in groupIndexOptional is not cleared, and the memory of this portion contains garbage data.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clipped_swiglu_v2.h"

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
    // Boilerplate: acl initialization.
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
    // Call aclrtMalloc to apply for memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
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

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> xShape = {2, 32};
    std::vector<int64_t> groupIndexShape = {1};
    std::vector<int64_t> outShape = {2, 16};
    void* xDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> xHostData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
    std::vector<int64_t> groupIndexData = {1};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int dim = -1;
    float alpha = 1.0;
    float limit = 7.0;
    float bias = 1.702;
    bool interleaved = true;
    int64_t clampMode = 1;
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the groupIndex aclTensor.
    ret = CreateAclTensor(groupIndexData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64, &groupIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnClippedSwigluV2.
    ret = aclnnClippedSwigluV2GetWorkspaceSize(x, groupIndex, dim, alpha, limit, bias, interleaved, clampMode, out,
                                               &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClippedSwigluV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnClippedSwigluV2.
    ret = aclnnClippedSwigluV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClippedSwigluV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device side to the host side. Modify it according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(groupIndex);
    aclDestroyTensor(out);
    // 7. Release device resources. Modify it according to the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(groupIndexDeviceAddr);
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
