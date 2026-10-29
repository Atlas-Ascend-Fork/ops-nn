# aclnnClippedSwiglu

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T08:33:43.868Z -->

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

- Description: A Swish gated linear unit activation function with clipping, which implements the SwiGlu computation on x. Compared with aclnnSwiGlu, this API adds the following input parameters: groupIndex, alpha, limit, bias, and interleaved, to support the SwiGlu variant used by the GPT-OSS model and the grouping scenario used by the MoE model.

- Formula:

  For the given input tensor x with dimensions [a,b,c,d,e,f,g...], aclnnClippedSwiglu performs the following computation:

  1. Concatenate the axes of x based on the input parameter dim. After concatenation, the dimensions become [pre, cut, after]. The cut axis is the axis that needs to be split into two tensors after concatenation. The split method can be front-back split or odd-even split. pre and after can be 1. For example, when dim is 3, the dimensions of x after concatenation are [a*b*c, d, e*f*g*…]. In addition, since the elements of the after axis are stored contiguously and the computation is element-wise, the cut axis and the after axis are merged, resulting in x with dimensions [pre, cut].

  2. Based on the input parameter group_index, filter the pre-axis of x. The formula is as follows:

     $$
     sum = \text{Sum}(group\_index)
     $$

     $$
     x = x[ : sum, : ]
     $$

     where sum is the sum of all elements in group_index. This step is skipped if group_index is not provided.

  3. Split x based on the input parameter interleaved. The formula is as follows:

     When interleaved is true, it indicates odd-even split:

     $$
     A = x[ : , : : 2]
     $$

     $$
     B = x[ : , 1 : : 2]
     $$

     When interleaved is false, it indicates front-back split:

     $$
     h = x.shape[1] // 2
     $$

     $$
     A = x[ : , : h]
     $$

     $$
     B = x[ : , h : ]
     $$
  4. Perform variant SwiGlu calculation based on the input parameters alpha, limit, and bias. The formula is as follows:

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
  5. Reshape the output tensor y to have the same number of dimensions as x before concatenation, with the size on the dim axis being half of that of x, and other dimensions identical to x.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call the "aclnnClippedSwigluGetWorkspaceSize" API first to obtain the workspace size required for the calculation and the executor that includes the calculation process of the operator. Then call the "aclnnClippedSwiglu" API to perform the calculation.

```Cpp
aclnnStatus aclnnClippedSwigluGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *groupIndexOptional,
    int64_t          dim,
    double           alpha,
    double           limit,
    double           bias,
    bool             interleaved,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnClippedSwiglu(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnClippedSwigluGetWorkspaceSize

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x(aclTensor*)</td>
      <td>Input</td>
      <td>The input x in the formula.</td>
      <td>Null pointers are not supported. The dimension must be greater than 0, and the shape size on the dimension corresponding to input parameter dim must be an even number. Empty tensors are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>See Constraints.</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>The input group_index in the formula, indicating the grouping case.</td>
      <td>Empty supported. If not a null pointer, the tensor must be 1-dimensional, and elements must be greater than or equal to 0. The i-th element indicates the number of x batches to be processed by the i-th group.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>See Constraints.</td>
    </tr>
    <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>In the formula, the input dim indicates the index of the dimension for merging and splitting x.</td>
      <td>The value range is [-x.dim(), x.dim()-1].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha (double)</td>
      <td>Input</td>
      <td>The input alpha in the formula indicates the parameter used by the SwiGlu variant.</td>
      <td>Recommended to be 1.702.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>limit (double)</td>
      <td>Input</td>
      <td>The input limit in the formula indicates a variant threshold value used by SwiGlu.</td>
      <td>Must be greater than 0, recommended to be 7.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias (double)</td>
      <td>Input</td>
      <td>The input bias in the formula indicates a variant of the bias parameter used by SwiGlu.</td>
      <td>Recommended to be 1.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>interleaved (bool)</td>
      <td>Input</td>
      <td>The input interleaved in the formula indicates whether x is split in an odd-even manner.</td>
      <td>Setting it to true indicates odd-even split of x, and setting it to false indicates front-back split of x.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>The output y in the formula.</td>
      <td>Null pointer not supported. The shape is half of x on the dimension specified by the input dim, and the same as x on other dimensions. Empty tensor supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>See Constraints.</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator calculation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - For input parameter `groupIndexOptional`, empty tensor not supported. The number of supported groups is determined by the UB space size.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - For input parameter `groupIndexOptional`, empty tensor supported.
  <!-- end id8 -->

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
      <td>The input x or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimensions of the input or output parameters are not within the supported range.</td>
    </tr>
    <tr>
      <td>The value of dim is not within the specified range.</td>
    </tr>
  </tbody>
  </table>

## aclnnClippedSwiglu

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
      <td>Workspace memory address applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the Device side, obtained by calling the first-phase API aclnnClippedSwigluGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which includes the calculation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnClippedSwiglu defaults to a deterministic implementation. Non-deterministic implementation is not supported currently. Even if the deterministic computation configuration is set, it does not take effect.
- Non-contiguity of tensor parameters:
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - All input tensor parameters support non-contiguous tensors, while output tensor parameters do not support non-contiguous tensors.
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - All input/output tensor parameters do not support non-contiguous tensors.
  <!-- end id10 -->

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clipped_swiglu.h"

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
    // Boilerplate: ACL initialization
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
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the ACL API documentation.
    // Set deviceId based on the actual device in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API definition.
    std::vector<int64_t> xShape = {2, 32};
    std::vector<int64_t> groupIndexShape = {1};
    std::vector<int64_t> outShape = {2, 16};
    void* xDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> xHostData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                             22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
    std::vector<int64_t> groupIndexData = {1};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int dim = -1;
    float alpha = 1.702;
    float limit = 7.0;
    float bias = 1.0;
    bool interleaved = true;
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the groupIndex aclTensor.
    ret = CreateAclTensor(groupIndexData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64, &groupIndex);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnClippedSwiglu.
    ret = aclnnClippedSwigluGetWorkspaceSize(
        x, groupIndex, dim, alpha, limit, bias, interleaved, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClippedSwigluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnClippedSwiglu.
    ret = aclnnClippedSwiglu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClippedSwiglu failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify it based on the definition of the specific API.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // 6. Release aclTensor and aclScalar. Modify based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(groupIndex);
    aclDestroyTensor(out);
    // 7. Release device resources. Modify based on the specific API definition.
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
