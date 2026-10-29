# aclnnDynamicQuantV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:25:15.250Z -->

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

- Description: Performs dynamic quantization on the input tensor. In the MOE scenario, smoothScalesOptional differs for each expert and is distinguished by the input groupIndexOptional. Supports symmetric/asymmetric quantization. Supports pertoken/pertensor/perchannel quantization modes. Compared with aclnnDynamicQuantV3, the optional input parameter dstTypeMax is added to specify the range of quantization results.

- Formula:
  - Symmetric quantization:
    - If smoothScalesOptional is not provided:
      $$
        scaleOut=\max_{t}(abs(x))/DTYPE_{MAX}
      $$
      $$
        yOut=round(x/scaleOut)
      $$
    - If smoothScalesOptional is provided:
      $$
        input = x\cdot smoothScalesOptional
      $$
      $$
        scaleOut=\max_{t}(abs(input))/DTYPE_{MAX}
      $$
      $$
        yOut=round(input/scaleOut)
      $$
  - Asymmetric quantization:
    - If smoothScalesOptional is not provided:
      $$
        scaleOut=(\max_{t}(x) - \min_{t}(x))/(DTYPE_{MAX} - DTYPE_{MIN})
      $$
      $$
        offset=DTYPE_{MAX}-\max_{t}(x)/scaleOut
      $$
      $$
        yOut=round(x/scaleOut+offset)
      $$
    - If smoothScalesOptional is provided:
      $$
        input = x\cdot smoothScalesOptional
      $$
      $$
        scaleOut=(\max_{t}(input) - \min_{t}(input))/(DTYPE_{MAX} - DTYPE_{MIN})
      $$
      $$
        offset=DTYPE_{MAX}-\max_{t}(input)/scaleOut
      $$
      $$
        yOut=round(input/scaleOut+offset)
      $$
  where $\max_{t}$/$\min_{t}$ indicates the mode of computing the maximum/minimum value: if quantMode is "pertoken", $t=row$, indicating that the maximum/minimum value is computed for each token; if quantMode is "pertensor", $t=all$, indicating that the maximum/minimum value of the entire tensor is computed; if quantMode is "perchannel", $t=col$, indicating that the maximum/minimum value is computed for each channel. $DTYPE_{MAX}$ is the maximum value of the output data type, and $DTYPE_{MIN}$ is the minimum value of the output data type. In the HiFloat8 scenario, custom input of $DTYPE_{MAX}$ ($DTYPE_{MIN}$) is supported.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnDynamicQuantV4GetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the computation process of the operator, and then call "aclnnDynamicQuantV4" to execute the computation.

```Cpp
aclnnStatus aclnnDynamicQuantV4GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScalesOptional,
  const aclTensor *groupIndexOptional,
  int64_t          dstType,
  bool             isSymmetrical,
  const char      *quantMode,
  float            dstTypeMax,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  const aclTensor *offsetOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDynamicQuantV4(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicQuantV4GetWorkspaceSize

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
      <td>Tensor of the operator input. Corresponds to x in the formula.</td>
      <td><ul>When the data type of the corresponding yOut is INT4, the size of the last dimension of x must be divisible by 2.</ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional(aclTensor*)</td>
      <td>Input</td>
      <td>smoothScales of the operator input. Corresponds to smoothScalesOptional in the formula description.</td>
      <td><ul><li>In the pertoken/pertensor scenario, when groupIndexOptional is not provided, the shape dimension is the last dimension of x.</li><li>When groupIndexOptional is provided, the shape is two-dimensional: the first dimension corresponds to the number of experts and cannot exceed 1024, and the second dimension equals the size of the last dimension of x.</li><li>In the perchannel scenario, the shape size is the size of the second-to-last dimension of x.</li></ul></td>
      <td>Same as x.</td>
      <td>ND</td>
      <td>1, 2</td>
      <td>√</td>
    </tr>
       <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>groupIndex of the operator input. Corresponds to groupIndexOptional in the formula description.</td>
      <td><ul><li>Only 1-D shape is supported, and the dimension size must equal the first dimension of smoothScalesOptional.</li><li>If groupIndexOptional is not nullptr, smoothScalesOptional must not be nullptr.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the type of yOut after data conversion, corresponding to DType in the formula.</td>
      <td><ul><li>The value range is {2, 3, 29, 34, 35, 36}, corresponding to the data types of output yOut: {2:INT8, 3:INT32, 29:INT4, 34:HIFLOAT8, 35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN}.</li><li>INT32 actually consists of 8 concatenated INT4 values.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isSymmetrical (bool)</td>
      <td>Input</td>
      <td>Specifies whether symmetric quantization is used.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (char*) </td>
      <td>Input</td>
      <td>Specifies the quantization mode.</td>
      <td><ul><li>Currently, "pertoken", "pertensor", and "perchannel" are supported.</li><li>If quantMode is "pertensor" or "perchannel", groupIndexOptional must be nullptr.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (float) </td>
      <td>Input</td>
      <td>Specifies the range of quantization results.</td>
      <td><ul>Currently, values from 0 to 32768 are supported. Takes effect only in the scenario where dstType is 34 (hifloat8).</ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output tensor. Its data type is specified by dstType. It corresponds to yOut in the corresponding formula.</td>
      <td><ul><li>When the data type is INT4, the size of the last dimension must be divisible by 2.</li><li>When the data type is INT32, the last dimension is 1/8 of the last dimension of x.</li><li>For other data types, the shape is the same as that of x.</li></ul></td>
      <td>INT4, INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Scale used for quantization. It is scaleOut in the corresponding formula.</td>
      <td><ul><li>If quantMode is pertoken, the shape is the shape of x with the last dimension removed.</li><li>If quantMode is pertensor, the shape is (1,).</li><li>If quantMode is perchannel, the shape is the shape of x with the second-to-last dimension removed.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOut (aclTensor*)</td>
      <td>Output</td>
      <td>Offset used for asymmetric quantization. It corresponds to offsetOut in the formula.</td>
      <td><ul>Supported only when isSymmetrical is false. If isSymmetrical is true, offsetOut must be set to nullptr.</ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
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
      <td>The input x or output parameter is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type, data format, dimensions, or other attributes of the parameter are not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561001</td>
      <td>Failed to create aclOpExecutor internally.</td>
    </tr>
  </tbody></table>

## aclnnDynamicQuantV4

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
        <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnDynamicQuantV4GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream used to execute the task.</td>
      </tr>
    </tbody>
    </table>

  - **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnDynamicQuantV4 defaults to a deterministic implementation.

- If the data type of yOut is INT4, the last dimension of x and yOut must be divisible by 2. Non-contiguous tensors are not supported.
- When the data type of yOut is INT32, the last dimension of x must be divisible by 8.
- When groupIndexOptional is used, the number of experts must not exceed the product of all dimensions of x after removing the last dimension. The values of groupIndexOptional must form a non-decreasing array of values not less than zero, and the last value must be equal to the product of all dimensions of x after removing the last dimension. If this condition is not met, the result is meaningless.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_quant_v4.h"

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
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for resource initialization.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  int rowNum = 4;
  int rowLen = 2;
  int groupNum = 2;
  std::vector<int64_t> xShape = {4, 2};
  std::vector<int64_t> smoothShape = {groupNum, rowLen};
  std::vector<int64_t> groupShape = {groupNum};
  std::vector<int64_t> yShape = {4, 2};
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

  std::vector<aclFloat16> xHostData;
  std::vector<aclFloat16> smoothHostData;
  std::vector<int32_t> groupHostData = {2, rowNum};
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

  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the smooth aclTensor.
  ret = CreateAclTensor(smoothHostData, smoothShape, &smoothDeviceAddr, aclDataType::ACL_FLOAT16, &smooth);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the group aclTensor.
  ret = CreateAclTensor(groupHostData, groupShape, &groupDeviceAddr, aclDataType::ACL_INT32, &group);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the offset aclTensor.
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API and replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  const char* quantMode = "pertoken";
  float dstTypeMax = 0.0;

  // Call the first-phase API of aclnnDynamicQuantV4.
  ret = aclnnDynamicQuantV4GetWorkspaceSize(x, smooth, group, 2, false, quantMode, dstTypeMax, y, scale, offset, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnDynamicQuantV4.
  ret = aclnnDynamicQuantV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV4 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host memory. Modify this code according to the specific API definition.
  PrintOutResult(yShape, &yDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify this code according to the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(smooth);
  aclDestroyTensor(y);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);

  // 7. Release device resources.
  aclrtFree(xDeviceAddr);
  aclrtFree(smoothDeviceAddr);
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
