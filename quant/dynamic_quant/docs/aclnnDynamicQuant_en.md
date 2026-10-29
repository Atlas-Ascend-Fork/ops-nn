# aclnnDynamicQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:19:57.536Z -->

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

- Description: Performs per-token symmetric dynamic quantization on the input tensor.

- Formula:
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

  Where row_max represents the maximum value of each row, and dtypeMax is the maximum value of the output data type.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnDynamicQuantGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnDynamicQuant" API to perform the computation.

```Cpp
aclnnStatus aclnnDynamicQuantGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* smoothScalesOptional,
  const aclTensor* yOut,
  const aclTensor* scaleOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnDynamicQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicQuantGetWorkspaceSize

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
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Tensor input of the operator. Corresponds to `x` in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>smoothScales input of the operator. Corresponds to `smoothScalesOptional` in the formula description.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type must be the same as that of x.</li><li>The shape dimension is the same as the last dimension of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor after quantization. Corresponds to `yOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>When the type is INT32, the last dimension of the shape is 1/8 of the last dimension of x, and the other dimensions are the same as those of x. For other types, the shape is the same as that of x.</li></ul></td>
      <td>INT4, INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Scale used for quantization. Corresponds to `scaleOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is the shape of x with the last dimension removed.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the device.</td>
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

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type of the output parameter `yOut` supports only INT4, INT8, and INT32.
  <!-- end id7 -->
  <!-- npu="910,310p" id8 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>:
    - The data type of the input `x` and `smoothScalesOptional` supports only FLOAT16.
    - The data type of the output parameter `yOut` only supports INT8.
    - The input parameter `smoothScalesOptional` is a reserved parameter and does not participate in computation in the current version.
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
      <td>The x or out parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type, data format, dimension, etc. of the parameter is out of the supported range.</td>
    </tr>
    <tr>
      <td>When the data type of yOut is INT4, the last dimension of x or yOut is not even.</td>
    </tr>
    <tr>
      <td>If the data type of yOut is INT32, the last dimension of x must not be divisible by 8.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561001</td>
      <td>Failed to create aclOpExecutor internally.</td>
    </tr>
  </tbody></table>

## aclnnDynamicQuant

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
      <td>Address of the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnDynamicQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="950,A3,910b" id9 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
  - When the data type of yOut is INT4, the last dimension of x and yOut must be divisible by 2. Non-contiguous tensors are not supported.
  - When the data type of yOut is INT32, the last dimension of x must be divisible by 8.
<!-- end id9 -->
<!-- npu="910,310p" id10 -->
- <term>Atlas inference products</term> and <term>Atlas training products</term>: The last dimension of input x must be divisible by 32. Currently, only symmetric quantization is supported, and the BFLOAT16 data type is not supported.
- deterministic computation:
  - aclnnDynamicQuant defaults to a deterministic implementation.
<!-- end id10 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_quant.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, (int)resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output, which must be customized based on the API.
    int rowNum = 4;
    int rowLen = 32;
    std::vector<int64_t> xShape = {rowNum, rowLen};
    std::vector<int64_t> smoothShape = {rowLen};
    std::vector<int64_t> yShape = {rowNum, rowLen};
    std::vector<int64_t> scaleShape = {rowNum};

    void* xDeviceAddr = nullptr;
    void* smoothDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* smooth = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scale = nullptr;

    std::vector<aclFloat16> xHostData;
    std::vector<aclFloat16> smoothHostData;
    std::vector<int8_t> yHostData;
    std::vector<float> scaleHostData;
    for (int i = 0; i < rowNum; ++i) {
        for (int j = 0; j < rowLen; ++j) {
            float value1 = i * rowLen + j;
            xHostData.push_back(aclFloatToFloat16(value1));
            yHostData.push_back(0);
        }
        scaleHostData.push_back(0);
    }

    for (int k = 0; k < rowLen; ++k) {
        float value2 = k * rowLen + 1;
        smoothHostData.push_back(aclFloatToFloat16(value2));
    }

    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the smooth aclTensor.
    ret = CreateAclTensor(smoothHostData, smoothShape, &smoothDeviceAddr, aclDataType::ACL_FLOAT16, &smooth);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnDynamicQuant.
    ret = aclnnDynamicQuantGetWorkspaceSize(x, smooth, y, scale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnDynamicQuant.
    ret = aclnnDynamicQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuant failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the result from the device side to the host side. Change it according to the specific API definition.
    PrintOutResult(yShape, &yDeviceAddr);

    // 6. Release the aclTensor. Change it according to the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(smooth);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(smoothDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
