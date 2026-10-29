# aclnnLeakyReluBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T11:56:52.938Z -->

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

Description: Backward propagation of the [aclnnLeakyRelu](../../leaky_relu/docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu.md) activation function.
Formula:

$$
output =
\begin{cases}
gradOutput, &if\ self \gt 0 \\
gradOutput*negativeSlope, &if\ self \le 0
\end{cases}
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnLeakyReluBackwardGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnLeakyReluBackward" to perform the computation.

```Cpp
aclnnStatus aclnnLeakyReluBackwardGetWorkspaceSize(
  const aclTensor *gradOutput,
  const aclTensor *self,
  const aclScalar *negativeSlope,
  bool             selfIsResult,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnLeakyReluBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLeakyReluBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1410px"><colgroup>
  <col style="width: 211px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the gradient.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type must satisfy the data type deduction rule with the data type of self (see also <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input tensor.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type must meet the data type deduction rule with the data type of gradOutput (see also <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li><li>The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with gradOutput.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>negativeSlope (aclScalar*)</td>
      <td>Input</td>
      <td>Slope when self < 0.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, BOOL, INT16, UINT8, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>selfIsResult (bool)</td>
      <td>Input</td>
      <td>-</td>
      <td>When selfIsResult is true, negativeSlope cannot be negative.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the calculation output.</td>
      <td><ul><li>No extra memory needs to be allocated. The data type must be a type to which gradOutput and self can be converted after deduction (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li><li>Other data types (INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX64, COMPLEX128) are supported through the automatic cast capability but require extra memory allocation.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

   <!-- npu="910,310p" id7 -->
   - <term>Atlas inference products</term>, <term>Atlas training products</term>: The data types of gradOutput, self, and out support FLOAT, FLOAT16, and DOUBLE. The data type of negativeSlope supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, BOOL, INT16, and UINT8.
   <!-- end id7 -->

- **Return value**

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
      <td>The passed-in gradOutput, self, negativeSlope, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of gradOutput, self, or negativeSlope is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of gradOutput, self, or out exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput and self do not meet the broadcast relationship, or the shape of out is inconsistent with the broadcast result.</td>
    </tr>
    <tr>
      <td>When selfIsResult is true, negativeSlope is negative.</td>
    </tr>
  </tbody></table>

## aclnnLeakyReluBackward

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
      <td>Address of the workspace memory requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnLeakyReluBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

 aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnLeakyReluBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_leaky_relu_backward.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
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
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API reference.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> gradShape = {4};
    std::vector<int64_t> selfShape = {4};
    std::vector<int64_t> outShape = {4};
    void* gradDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* grad = nullptr;
    aclTensor* self = nullptr;
    aclScalar* negativeSlope = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> gradHostData = {2, 3, 4, 5};
    std::vector<float> selfHostData = {1, 2, 3, 4};
    std::vector<float> outHostData = {0, 0, 0, 0};
    float negativeSlopeValue = 0.01f;
    bool selfIsResultValue = true;
    // Create the grad aclTensor.
    ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the negativeSlope aclScalar.
    negativeSlope = aclCreateScalar(&negativeSlopeValue, aclDataType::ACL_FLOAT);
    CHECK_RET(negativeSlope != nullptr, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace the name with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnLeakyReluBackward.
    ret = aclnnLeakyReluBackwardGetWorkspaceSize(
        grad, self, negativeSlope, selfIsResultValue, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLeakyReluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnLeakyReluBackward.
    ret = aclnnLeakyReluBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLeakyReluBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronously wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device side to the host side. Modify the code according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
    aclDestroyTensor(grad);
    aclDestroyTensor(self);
    aclDestroyScalar(negativeSlope);
    aclDestroyTensor(out);

    // 7. Release device resources.
    aclrtFree(gradDeviceAddr);
    aclrtFree(selfDeviceAddr);
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
