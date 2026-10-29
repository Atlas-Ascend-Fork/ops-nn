# aclnnGeGluV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T09:58:10.098Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Gaussian error linear unit activation gating function. Compared with aclnnGeGlu, this API extends the capability to set the direction of the data block for the activation function operation.
- Formula:
  If activateLeft is true, the activation is applied to the left half of $self$.

    $$
    out_{i}=GeGlu(self_{i}) = Gelu(A) \cdot B
    $$

  If activateLeft is false, it indicates that the right half of $self$ is activated.

    $$
    out_{i}=GeGlu(self_{i}) = A \cdot Gelu(B)
    $$

  Where $A$ represents the left half of $self$, and $B$ represents the right half of $self$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGeGluV3GetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnGeGluV3" API to perform the computation.

```Cpp
aclnnStatus aclnnGeGluV3GetWorkspaceSize(
  const aclTensor   *self,
  int64_t            dim,
  int64_t            approximate,
  bool               activateLeft,
  aclTensor         *out,
  aclTensor         *outGelu,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnGeGluV3(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  aclrtStream        stream)
```

## aclnnGeGluV3GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 280px">
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
      <th>Dimensions</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input parameter for GeGluV3 computation, self in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>Optional input parameter.</td>
      <td><ul><li>The specified slice axis, which is used to split the corresponding axis of self in half.</li><li>The axis of self corresponding to dim must be even.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>approximate (int64_t)</td>
      <td>Input</td>
      <td>Optional input.</td>
      <td>Index of the activation function used in GeGluV3 computation. 0 indicates none and 1 indicates tanh.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Specifies the direction of the data block on which the activation function is performed.</td>
      <td>false indicates activating the right part, and true indicates activating the left part.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output parameter of the GeGluV3 computation, which is out_i in the formula.</td>
      <td><ul><li>The shape of out must be the same as the shape of self except for the axis specified by dim. The size of the axis specified by dim is half of the corresponding axis in the shape of self.</li><li>The data type is the same as that of self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outGelu (aclTensor*)</td>
      <td>Output</td>
      <td>Output parameter of the GeGluV3 computation.</td>
      <td><ul><li>The shape of outGelu must be the same as the shape of self except for the axis specified by dim. The size of the axis specified by dim is half of that of the corresponding axis in the shape of self.</li><li>The data type must be the same as that of self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The supported data types are FLOAT and FLOAT16.
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
      <td>Parameters self, out, and outGelu are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of self, out, or outGelu are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of out and outGelu are inconsistent with that of self.</td>
    </tr>
    <tr>
      <td>The number of dimensions of self, out, or outGelu is greater than 8.</td>
    </tr>
    <tr>
      <td>If self.dim()=0, dim is not in the range [-1, 0]; if self.dim()>0, dim is not in the range [-self.dim, self.dim()-1].</td>
    </tr>
    <tr>
      <td>The shapes of out and outGelu must be consistent with the shape of self except for the axis specified by dim, and the axis specified by dim is half the size of the corresponding axis in the shape of self.</td>
    </tr>
  </tbody></table>

## aclnnGeGluV3

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained by the first-phase API aclnnGeGluV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor. Contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGeGluV3 defaults to a deterministic implementation.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_geglu.h"

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
    // Boilerplate for resource initialization
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
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
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
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Customize the construction as required by the API.
    std::vector<int64_t> selfShape = {2, 2};
    std::vector<int64_t> outShape = {2, 1};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* outGeluDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclTensor* outGelu = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3};
    std::vector<float> outHostData = {0, 0};
    std::vector<float> outGeluHostData = {0, 0};
    int dim = -1;
    int approximate = 1;
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the outGelu aclTensor.
    ret = CreateAclTensor(outGeluHostData, outShape, &outGeluDeviceAddr, aclDataType::ACL_FLOAT, &outGelu);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnGeGluV3.
    bool activateLeft = false;
    ret = aclnnGeGluV3GetWorkspaceSize(self, dim, approximate, activateLeft, out, outGelu, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeGluV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGeGluV3.
    ret = aclnnGeGluV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeGluV3 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device side to the host side. Modify it according to the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    std::vector<float> resultGeluData(size, 0);
    ret = aclrtMemcpy(
        resultGeluData.data(), resultGeluData.size() * sizeof(resultGeluData[0]), outGeluDeviceAddr,
        size * sizeof(resultGeluData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("copy resultGelu from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultGeluData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it according to the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclDestroyTensor(outGelu);

    // 7. Release device resources. Modify it according to the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(outGeluDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
