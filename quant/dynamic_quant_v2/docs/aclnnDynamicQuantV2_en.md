# aclnnDynamicQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:25:18.849Z -->

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

- Description: Performs per-token symmetric/asymmetric dynamic quantization on the input tensor. In MOE scenarios, the smoothScalesOptional of each expert is different and is distinguished based on the input groupIndexOptional.

- Formula:
  - Symmetric quantization:
    - If smoothScalesOptional is not provided:

      $$
        scaleOut=row\_max(abs(x))/dtypeMax
      $$

      $$
        yOut=round(x/scaleOut)
      $$

    - If smoothScalesOptional is provided:

      $$
        input = x\cdot smoothScalesOptional
      $$

      $$
        scaleOut=row\_max(abs(input))/dtypeMax
      $$

      $$
        yOut=round(input/scaleOut)
      $$

  - Asymmetric quantization:
    - If smoothScalesOptional is not provided:

      $$
        scaleOut=(row\_max(x) - row\_min(x))/(dtypeMax - dtypeMin)
      $$

      $$
        offset=dtypeMax-row\_max(x)/scaleOut
      $$

      $$
        yOut=round(x/scaleOut+offset)
      $$

    - If smoothScalesOptional is provided:

      $$
        input = x\cdot smoothScalesOptional
      $$

      $$
        scaleOut=(row\_max(input) - row\_min(input))/(dtypeMax - dtypeMin)
      $$

      $$
        offset=dtypeMax-row\_max(input)/scaleOut
      $$

      $$
        yOut=round(input/scaleOut+offset)
      $$

  where row\_max denotes the maximum value of each row, row_min denotes the minimum value of each row, dtypeMax is the maximum value of the output data type, and dtypeMin is the minimum value of the output data type.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDynamicQuantV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnDynamicQuantV2" API to execute the computation.

```Cpp
aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScalesOptional,
  const aclTensor *groupIndexOptional,
  int64_t          dstType,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  const aclTensor *offsetOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDynamicQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicQuantV2GetWorkspaceSize

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Tensor of the operator input. Corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>Greater than 1 dimension</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>smoothScales of the operator input. Corresponds to `smoothScalesOptional` in the formula description.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type must be the same as that of x.</li><li>When groupIndexOptional is not specified, the shape dimension is the same as the last dimension of x. When groupIndexOptional is specified, the shape is two-dimensional. The first dimension is the number of experts, not greater than 1024. The second dimension is the last dimension of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>GroupIndex of the operator input.</td>
      <td><ul><li>Supports empty tensors.</li><li>Only one-dimensional shapes are supported, and the dimension size must be equal to the first dimension of smoothScalesOptional.</li><li>When groupIndexOptional is not nullptr, smoothScalesOptional must not be nullptr.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Enum value corresponding to the type of output y.</td>
      <td><ul><li>If the type of output y is INT8, the value is 2; if the type of y is INT4, the value is 29; if the type of y is INT32, the value is 3.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor after quantization. Corresponds to `yOut` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When the type is INT32, the last dimension of the shape is 1/8 of the last dimension of x, and the other dimensions are the same as those of x. For other types, the shape is the same as that of x.</li></ul></td>
      <td>INT4, INT8, INT32, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
      <td>Greater than 1 dimension</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Scale used for quantization. Corresponds to `scaleOut` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape dimensions are the shape of x with the last dimension removed.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>N-dimension (N > 0)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOut (aclTensor*)</td>
      <td>Output</td>
      <td>Offset used for quantization. Corresponds to `offset` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape dimensions are the shape of x with the last dimension removed.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>Greater than 0 dimensions</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor**)</td>
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

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type of output parameter `yOut` supports only INT4, INT8, and INT32.
  <!-- end id7 -->
  <!-- npu="910,310p" id8 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>:
    - The input parameter `x` only supports FLOAT16.
    - The input parameters `smoothScalesOptional` and `groupIndexOptional` are reserved parameters and do not participate in computation in the current version.
    - The input parameter `dstType` only supports being set to 2.
    - The output parameter `yOut` only supports INT8.
  <!-- end id8 -->

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
      <td>The input parameter x or yOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type, data format, or dimension of a parameter is not in the supported range.</td>
    </tr>
    <tr>
      <td>When the data type of yOut is INT4, the last dimension of x or yOut is not an even number.</td>
    </tr>
    <tr>
      <td>When the data type of yOut is INT32, the last dimension of x is not divisible by 8.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561001</td>
      <td>Failed to create aclOpExecutor internally.</td>
    </tr>
  </tbody></table>

## aclnnDynamicQuantV2

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the device. The value is obtained by the first-phase API aclnnDynamicQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="950,A3,910b" id9 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
  - If the data type of `yOut` is INT4, the last dimensions of `x` and `yOut` must be divisible by 2.
  - If the data type of `yOut` is INT32, the last dimension of `x` must be divisible by 8.
  - When groupIndexOptional is used, the number of experts must not exceed the product of all dimensions of x excluding the last dimension. The values of groupIndexOptional must be a non-decreasing array, with each value no less than 0, and the last value must equal the product of all dimensions of x excluding the last dimension. Otherwise, the result is meaningless.
<!-- end id9 -->
<!-- npu="910,310p" id10 -->
- <term>Atlas inference products</term> and <term>Atlas training products</term>: The last dimension of input x must be divisible by 32. Currently, only symmetric quantization is supported, and the BFLOAT16 data type is not supported.
<!-- end id10 -->
- Deterministic computation:
  - aclnnDynamicQuantV2 defaults to a deterministic implementation.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_quant_v2.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. You need to customize the construction based on the API.
    int rowNum = 4;
    int rowLen = 32;
    int groupNum = 4;
    std::vector<int64_t> xShape = {4, 32};
    std::vector<int64_t> smoothShape = {groupNum, rowLen};
    std::vector<int64_t> groupShape = {groupNum};
    std::vector<int64_t> yShape = {4, 32};
    std::vector<int64_t> scaleShape = {4};
    std::vector<int64_t> offsetShape = {4};

    void* xDeviceAddr = nullptr;
    void* smoothDeviceAddr = nullptr;
    void* groupDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* smooth = nullptr;
    aclTensor* group = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;

    std::vector<float> xHostData;
    std::vector<float> smoothHostData;
    // groupIndex stores the cumulative row count for which each expert is responsible. The last element must be equal to the total number of rows in x
    std::vector<int32_t> groupHostData = {1, 2, 3, rowNum};
    std::vector<int8_t> yHostData;
    std::vector<float> scaleHostData;
    std::vector<float> offsetHostData;
    for (int i = 0; i < rowNum; ++i) {
        for (int j = 0; j < rowLen; ++j) {
            float value1 = i * rowLen + j;
            xHostData.push_back(aclFloatToFloat16(value1));
            yHostData.push_back(0);
        }
        scaleHostData.push_back(0);
        offsetHostData.push_back(0);
    }

    for (int m = 0; m < groupNum; ++m) {
        for (int n = 0; n < rowLen; ++n) {
            float value2 = m * rowLen + n;
            smoothHostData.push_back(aclFloatToFloat16(value2));
        }
    }

    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create smooth aclTensor.
    ret = CreateAclTensor(smoothHostData, smoothShape, &smoothDeviceAddr, aclDataType::ACL_FLOAT16, &smooth);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create group aclTensor.
    ret = CreateAclTensor(groupHostData, groupShape, &groupDeviceAddr, aclDataType::ACL_INT32, &group);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call a CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnDynamicQuantV2.
    ret = aclnnDynamicQuantV2GetWorkspaceSize(x, smooth, group, 2, y, scale, offset, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnDynamicQuantV2.
    ret = aclnnDynamicQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value. Copy the result from device memory to host memory. Modify the code based on the specific API definition.
    PrintOutResult(yShape, &yDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(smooth);
    aclDestroyTensor(group);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(smoothDeviceAddr);
    aclrtFree(groupDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
