# aclnnAscendAntiQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:05:19.759Z -->

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

- Description: Dequantizes the input x based on the input scale and offset.
- Formula:
  - If sqrtMode is true and offset is None, the formula is as follows:

    $$
    y = cast\_to\_dst\_type((x) * scale * scale)
    $$

  - If sqrtMode is true and offset is not None, the formula is as follows:

    $$
    y = cast\_to\_dst\_type((x + offset) * scale * scale)
    $$

  - If sqrtMode is false and offset is None, the calculation formula is as follows:

    $$
    y = cast\_to\_dst\_type((x) * scale)
    $$

  - If sqrtMode is false and offset is not None, the calculation formula is as follows:

    $$
    y = cast\_to\_dst\_type((x + offset) * scale)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnAscendAntiQuantGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnAscendAntiQuant" API to perform the computation.

```Cpp
aclnnStatus aclnnAscendAntiQuantGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* scale,
  const aclTensor* offset,
  int64_t          dstType,
  bool             sqrtMode,
  const aclTensor* y,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAscendAntiQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAscendAntiQuantGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Denotes the input to be anti-quantized. It corresponds to x in the formula.</td>
      <td><ul><li>Supports empty tensors;</li><li>When the data type is INT32, each element is used as 8 INT4 values;</li><li>When the data type is INT4, the last dimension of shape must be an even number.</li></ul></td>
      <td>INT4, INT8, INT32, HIFLOAT8, FLOAT8_E4M3, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>Denotes the scale value in anti-quantization. It corresponds to scale in the formula.</td>
      <td><ul><li>Supports empty tensors;</li><li>The number of dimensions of scale must be the same as that of x, or be 1;</li><li>If x is 1-dimensional, the shape of scale must be [1] or the same as x;</li><li>If scale is 1-dimensional, its size must be 1, x[-1], or x[-2];</li><li>If scale is multi-dimensional, at most one dimension can be non-1, and the non-1 dimension can only be axis -1 or -2;</li><li>When the input x is of the INT32 type and the dequantization axis is the last axis, the last dimension of scale must be 8 times that of x;</li><li>When x is of type HIFLOAT8, FLOAT8_E4M3, or FLOAT8_E5M2, scale supports only FLOAT32.</li></ul></td>
      <td>FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset(aclTensor*)</td>
      <td>Input</td>
      <td>Offset value in dequantization. Corresponds to offset in the formula.</td>
      <td><ul><li>Supports empty tensors;</li><li>The data type and shape must be the same as those of scale.</li></ul></td>
      <td>FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the output data type.</td>
      <td>Supported values: 1 and 27, indicating FLOAT16 and BFLOAT16 respectively.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sqrtMode (bool)</td>
      <td>Input</td>
      <td>Specifies how scale participates in the computation. Corresponds to sqrtMode in the formula.</td>
      <td>When the data type of x is HIFLOAT8, FLOAT8_E4M3, or FLOAT8_E5M2, sqrtMode is false.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Computation output of dequantization. Corresponds to y in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When input x is of the INT32 type, the size of the last dimension of the shape is 8 times that of the last dimension of input x. In other cases, the shape is the same as that of input x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>:
    - Data types:
      - The input `x` supports only INT8.
      - The inputs `scale` and `offset`, and the output `y` do not support BFLOAT16.
    - The input `dstType` supports only the value 1, which indicates FLOAT16.
    - Input parameters `scale` and `offset` support only 1D tensors, and their sizes can only be equal to the size of the last dimension of x or 1. If x is of the int32 type, the size must be 8 times the size of the last dimension of x.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Input parameter `x` supports only INT4, INT8, and INT32.
    - Input parameters `scale` and `offset` support only 1D tensors, and their sizes can only be equal to the size of the last dimension of x or 1. If x is of the int32 type, the size must be 8 times the size of the last dimension of x.
  <!-- end id8 -->

- **Return Value**

  aclnnStatus: Returns the status code. For details, see [aclnnReturnCode](../../../docs/en/context/aclnn_return_code.md).

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
      <td>The input x, scale, or y is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type or data format of x, scale, offset, or y is not within the supported range.</td>
    </tr>
    <tr>
      <td>dstType is not within the valid value range.</td>
    </tr>
    <tr>
      <td>If the data type of x is INT4, the antiquantization axis size of x is not an even number.</td>
    </tr>
    <tr>
      <td>If the data type of x is INT4, x is a non-contiguous tensor.</td>
    </tr>
    <tr>
      <td>If the data type of x is INT32, the dequantization axis of y is not 8 times the size of the dequantization axis of x, or the sizes of the other axes of x and y are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnAscendAntiQuant

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnAscendAntiQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When the data type of input x is INT4, x does not support non-contiguous tensors.
- Deterministic computation:
  - aclnnAscendAntiQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ascend_anti_quant.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate, resource initialization
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
    // Call aclrtMalloc to allocate device-side memory
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Custom construction is required based on the API
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> offsetShape = {1};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<int8_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> scaleHostData = {1};
    std::vector<float> offsetHostData = {1};
    std::vector<short> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // Create the self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT8, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t dstType = 1;
    bool sqrtMode = false;

    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API aclnnAscendAntiQuant.
    ret = aclnnAscendAntiQuantGetWorkspaceSize(self, scale, offset, dstType, sqrtMode, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendAntiQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnAscendAntiQuant.
    ret = aclnnAscendAntiQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendAntiQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output. Copy the result from the device memory to the host memory. Modify the code according to the specific API definition and check the data in resultData.
    auto size = GetShapeSize(outShape);
    std::vector<short> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(short),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %hd\n", i, resultData[i]);
    }

    // 6. Release the aclTensors. Modify the code according to the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the code as required by the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
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
