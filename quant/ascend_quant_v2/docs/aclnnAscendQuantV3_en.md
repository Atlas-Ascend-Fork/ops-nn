# aclnnAscendQuantV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:07:26.625Z -->

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

- Description: Performs quantization on the input x. You can set axis to specify the axis corresponding to scale and offset. The shape of scale and offset must be equal to the axis of x specified by axis, or equal to 1.
- Formula:
  - When sqrtMode is false, the formula is:

    $$
    y = round((x * scale) + offset)
    $$

  - When sqrtMode is true, the formula is:

    $$
    y = round((x * scale * scale) + offset)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnAscendQuantV3GetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnAscendQuantV3" API to execute the computation.

```Cpp
aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *scale,
  const aclTensor *offset,
  bool             sqrtMode,
  const char      *roundMode,
  int32_t          dstType,
  int32_t          axis,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAscendQuantV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAscendQuantV3GetWorkspaceSize

- **Parameters**

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
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input to be quantized. Corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>For the ND data format, if `dstType` is 3, the last dimension of the shape must be divisible by 8. If `dstType` is 29, the last dimension of the shape must be divisible by 2.</li><li>For the NZ data format, only 3D shapes are supported, and the last dimension of the shape must be divisible by 8.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>Scale value used for quantization. Corresponds to `scale` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>`scale` supports 1D or multi-dimensional tensors. The shape depends on the input `x` and the attribute `axis` (when the shape of `scale` is a 1D tensor, dimension 0 of `scale` must equal dimension `axis` of x or 1; when the shape of `scale` is a multi-dimensional tensor, the number of dimensions of `scale` must keep consistent with that of `x`, dimension `axis` of `scale` must equal dimension `axis` of x or 1, and the other dimensions of `scale` are 1).</li><li>The data type and data format must be consistent with those of `x`.</li><li>When the data format is NZ, all elements of `scale` are 1.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset (aclTensor*)</td>
      <td>Input</td>
      <td>Optional parameter, the offset value in quantization. Corresponds to `offset` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape must keep consistent with those of `scale`.</li><li>When the data format is NZ, the value is empty, and the data type of offset keeps consistent with that of x.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sqrtMode (bool)</td>
      <td>Input</td>
      <td>Specifies the logic of scale participating in the computation. Corresponds to `sqrtMode` in the formula.</td>
      <td>If the value is true, the formula is y = round((x * scale * scale) + offset). If the value is false, the formula is y = round((x * scale) + offset).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>Input</td>
      <td>Specifies the conversion method for casting to the output data type.</td>
      <td><ul><li>Supported values: round/ceil/trunc/floor/hybrid.</li><li>If the data format of input `x` is NZ, the supported value is round.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int32_t)</td>
      <td>Input</td>
      <td>Specifies the data type of the output.</td>
      <td><ul><li>Supported values are 2, 3, 29, 34, 35, and 36, indicating INT8, INT32, INT4, HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN respectively.</li><li>When the data format of input `x` is NZ, the supported value is 3, indicating INT32.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis (int32_t)</td>
      <td>Input</td>
      <td>Specifies the dimension of `x` corresponding to `scale` and `offset`. When the data format of input `x` is NZ, the value is -1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Output of the quantization computation. Corresponds to `y` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When the type is INT32, the last dimension of the shape is 1/8 of the last dimension of `x`, and the other dimensions are the same as those of `x`; for other types, the shape is the same as that of `x`.</li></ul></td>
      <td>INT8, INT32, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND, NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    - When the data format of parameters `x`, `scale`, and `offset` is NZ, the data type only supports FLOAT32.
    - The output parameter `y` supports only INT8, INT32, and INT4 data types. When the data format is NZ, the data type supports INT32.
    - The input parameter `roundMode` supports the values round, ceil, trunc, and floor. When the data format of input `x` is NZ, it supports the value round.
    - The input parameter `dstType` supports the values 2, 3, and 29, which represent INT8, INT32, and INT4 respectively. When the data format of input `x` is NZ, it supports the value 3, which represents INT32.
    - The input parameter `axis` supports specifying the last two dimensions of x (assuming the input x has xDimNum dimensions, the value range of axis is [-2, -1] or [xDimNum-2, xDimNum-1]).
  <!-- end id7 -->

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The data format of `x`, `scale`, and `offset` does not support NZ.
    - Input parameter `roundMode`: When `dstType` is FLOAT8_E5M2 or FLOAT8_E4M3FN, only round is supported. When `dstType` is HIFLOAT8, round and hybrid are supported. When `dstType` is another type, round, ceil, trunc, and floor are supported.
    - The input parameter `axis` supports specifying the last two dimensions of x (assuming the input x has xDimNum dimensions, the value range of axis is [-2, -1] or [xDimNum-2, xDimNum-1]).
  <!-- end id8 -->

  <!-- npu="310p" id9 -->
  - <term>Atlas inference products</term>:
    - The data type of input parameters `x`, `scale`, and `offset` does not support BFLOAT16, and the data format does not support NZ.
    - The output parameter `y` only supports the INT8 data type and does not support the NZ data format.
    - The input parameter `roundMode` supports the values round, ceil, trunc, and floor.
    - The input parameter `dstType` only supports the value 2, which indicates INT8.
    - The input parameter `axis` only supports specifying the last dimension of x (assuming the input x has xDimNum dimensions, the value of axis is -1 or xDimNum-1).
  <!-- end id9 -->

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
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, scale, or y is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type or data format of x, scale, offset, or y is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x, scale, offset, or y does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The number of dimensions of x is not within 1 to 8.</td>
    </tr>
    <tr>
      <td>roundMode is not in the valid value range.</td>
    </tr>
    <tr>
      <td>dstType is not in the valid value range.</td>
    </tr>
    <tr>
      <td>axis is not in the valid value range.</td>
    </tr>
    <tr>
      <td>If the data type of y is INT4, the tail axis size of the shape of x is not even.</td>
    </tr>
    <tr>
      <td>If the data type of y is INT32, the tail axis size of the shape of y is not 1/8 of that of the shape of x, or the sizes of the non-tail axes of the shapes of x and y are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnAscendQuantV3

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnAscendQuantV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnAscendQuantV3 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ascend_quant_v3.h"

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
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> scaleShape = {2};
    std::vector<int64_t> offsetShape = {2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> scaleHostData = {1, 2};
    std::vector<float> offsetHostData = {1, 2};
    std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int32_t dstType = 2;
    const int32_t axis = -1;
    bool sqrtMode = false;
    const char* roundMode = "round";

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAscendQuantV3.
    ret = aclnnAscendQuantV3GetWorkspaceSize(
        self, scale, offset, sqrtMode, roundMode, dstType, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAscendQuantV3.
    ret = aclnnAscendQuantV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendQuantV3 failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host memory. Modify the code based on the specific API definition and view the data in resultData.
    auto size = GetShapeSize(outShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release the aclTensor. Modify the code based on the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(out);

    // 7. Release the device resources. Modify the code based on the specific API definition.
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
