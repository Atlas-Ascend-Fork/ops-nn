# aclnnSwiGluQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:40:19.773Z -->

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

- Description: Adds a quant operation after the SwiGlu activation function to perform SwiGluQuant computation on input x, supporting int8 or int4 quantized output.
- Description of API differences: Compared with the aclnnSwiGluQuant API, aclnnSwiGluQuantV2 newly supports the cumsum mode and count mode for groupIndexOptional, with different modes controlled by groupListType; newly supports non-MoE scenarios (groupIndexOptional passes null); and newly supports int8 or int4 quantized output yOut, with different quantized output data types controlled by dstType.
- Operator support scope: Currently, SwiGluQuant supports MoE scenarios (with groupIndexOptional passed) and non-MoE scenarios (groupIndexOptional passed as null). The inputs x and group_index of SwiGluQuant come from the outputs of the GroupedMatMul operator and MoeInitRouting. The group_index parameter enables MoE group-based dynamic quantization, static per-tensor quantization, and static per-channel quantization.
- Dynamic quantization formula for MoE scenarios:
  $$
    Act = SwiGlu(x) = Swish(A)*B \\
    Y_{tmp}^0 = Act[0\colon g[0],\colon] * smooth\_scales[0\colon g[0],\colon], i=0 \\
    Y_{tmp}^i = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[g[i]\colon g[i+1], \colon], i \in (0, G) \cap \mathbb{Z}\\
    scale=dstTypeScale/row\_max(abs(Y_{tmp}))
  $$

  $
    Y = Cast(Mul(Y_{tmp}, Scale))
  $
     Where A represents the first half of input x, B represents the second half of input x, g represents group_index, and G is the number of groups in group_index. For int8 quantization, $dstTypeScale = 127$ (127 is the maximum value of int8); for int4 quantization, $dstTypeScale = 7$ (7 is the maximum value of int4).

- Static quantization formula for MoE scenarios:
  $
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}^0 = Act(0\colon g[0],\colon) * smooth\_scales[0\colon g[0],\colon] + offsets[0\colon g[0],\colon], i=0 \\
    Y_{tmp}^i = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[g[i]\colon g[i+1], \colon] + offsets[g[i]\colon g[i+1], \colon], i \in (0, G) \cap \mathbb{Z}\\
  $
  $
    Y = Cast(Y_{tmp})
  $
  Where A represents the first half of input x, B represents the second half of input x, g represents group_index, and G is the number of groups in group_index.

- Non-MoE scenario (groupIndexOptional is passed null) dynamic quantization formula:
  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp} = Act* smooth\_scales(0,\colon)\\
    scale=dstTypeScale/row\_max(abs(Y_{tmp}))
  $$

  $
    Y = Cast(Mul(Y_{tmp}, Scale))
  $
     Where A represents the first half of input x, B represents the second half of input x. For int8 quantization, $dstTypeScale = 127$ (127 is the maximum value of int8); for int4 quantization, $dstTypeScale = 7$ (7 is the maximum value of int4).

- Non-MoE scenario (groupIndexOptional is passed null) static quantization formula:
  $
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp} = Act * smooth\_scales(0,\colon) + offsets(0,\colon) \\
  $
  $
    Y = Cast(Y_{tmp})
  $
  Where A represents the first half of input x, and B represents the second half of input x.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSwiGluQuantV2GetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnSwiGluQuantV2" to execute the computation.

```Cpp
aclnnStatus aclnnSwiGluQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScalesOptional,
  const aclTensor *offsetsOptional,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *quantModeOptional,
  int64_t          groupListType,
  int64_t          dstType,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwiGluQuantV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnSwiGluQuantV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input data to be processed, that is, x in the formula.</td>
      <td><ul><li>The last dimension of x must be a multiple of 2, and the number of dimensions of x must be greater than 1.</li><li>Currently only supports a last-dimension length of input x not greater than 8192.</li><li>When dstType is set to 29 (indicating that the output yOut is the INT4 quantized output), the last dimension of x must be a multiple of 4.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>smooth_scales for quantization, that is, smooth_scales in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Shapes [G, N] and [G, ] are supported, where G indicates the number of groups in groupIndex, and N is half the size of the last dimension of compute input x.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
     <tr>
      <td>offsetsOptional (aclTensor*)</td>
      <td>Input</td>
      <td>offsets in the formula.</td>
      <td><ul><li>This parameter does not take effect in the dynamic quantization use case. Pass a null pointer.</li><li>In the static quantization use case: FLOAT is supported for the data type.</li><li>Shape [G, N] is supported in per_channel mode.</li><li>Shape [G, ] is supported in per_tensor mode, and the data type and shape must be kept consistent with those of smoothScalesOptional.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional(aclTensor*)</td>
      <td>Input</td>
      <td>group_index required for MoE grouping, which is the group_index in the formula.</td>
      <td>The shape supports [G, ]. Elements in group_index must be non-decreasing, and the maximum value cannot exceed the product of the sizes of all dimensions of input x except the last dimension; the value of G cannot exceed the product of the sizes of all dimensions of input x except the last dimension.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
      <tr>
      <td>activateLeft(bool)</td>
      <td>Input</td>
      <td>Indicates whether the left matrix participates in the computation. The user must pass parameters.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>quantModeOptional(char*)</td>
      <td>Input</td>
      <td>Compute input.</td>
      <td>The user must pass parameters. "static" indicates static quantization, "dynamic" indicates dynamic quantization, and "dynamic_msd" indicates dynamic MSD quantization. Currently only supports "dynamic" dynamic quantization and "static" static quantization. Static quantization only supports per_tensor quantization and per_channel quantization.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>groupListType (int64_t)</td>
      <td>Input</td>
      <td>Compute input.</td>
      <td>The user must pass parameters. 0 indicates the cumsum mode, and 1 indicates the count mode. Currently, only 0 (cumsum mode) and 1 (count mode) are supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Compute input.</td>
      <td>The user must pass parameters. 2 indicates that yOut is the int8 quantized output, and 29 indicates that yOut is the int4 quantized output. Currently, only 2 and 29 are supported as inputs, and the default value is 2.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The size of the last dimension of the output yOut is half that of the last dimension of compute input x, and the other dimensions keep consistent with x.</td>
      <td>INT8, INT4</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
      <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The shape of output scaleOut is the same as that of compute input x except that the last dimension is removed, and the remaining dimensions keep consistent with compute input x.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device side.</td>
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
      <td>The passed-in x or yOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>The parameter dimension of the input or output is not within the supported range.</td>
    </tr>
      <tr>
      <td>quantModeOptional is not in the specified value range.</td>
    </tr>
    <tr>
      <td>groupListType is not in the specified value range.</td>
    </tr>
      <tr>
      <td>dstType is not in the specified value range.</td>
    </tr>
  </tbody></table>

## aclnnSwiGluQuantV2

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
      <td>Memory address of the workspace applied for on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied for on the Device side, obtained by calling the first-phase API aclnnSwiGluQuantV2GetWorkspaceSize.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnSwiGluQuantV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swi_glu_quant_v2.h"

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
  // Call aclrtMalloc to apply for device-side memory
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API
  // Fill in deviceId based on the actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Custom construction is required based on the API
  std::vector<int64_t> xShape = {3, 32};
  std::vector<int64_t> smoothScalesShape = {2, 16};
  std::vector<int64_t> groupIndexShape = {2};
  std::vector<int64_t> outShape = {3, 16};
  std::vector<int64_t> scaleShape = {3};
  void* xDeviceAddr = nullptr;
  void* smoothScalesDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* smoothScales = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scale = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                  43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                  63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                                  83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  std::vector<float> smoothScalesHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> groupIndexHostData = {1, 3};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleHostData = {0, 0, 0};

  // Create the x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the smoothScales aclTensor
  ret = CreateAclTensor(smoothScalesHostData, smoothScalesShape, &smoothScalesDeviceAddr, aclDataType::ACL_FLOAT, &smoothScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the groupIndex aclTensor
  ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT32, &groupIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSwiGluQuantV2.
  ret = aclnnSwiGluQuantV2GetWorkspaceSize(x, smoothScales, nullptr, groupIndex, false, "dynamic", 0, 2, out, scale, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSwiGluQuantV2.
  ret = aclnnSwiGluQuantV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluQuantV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(smoothScales);
  aclDestroyTensor(groupIndex);
  aclDestroyTensor(out);
  aclDestroyTensor(scale);
  // 7. Release device resources. Modify it according to the specific API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(smoothScalesDeviceAddr);
  aclrtFree(groupIndexDeviceAddr);
  aclrtFree(outDeviceAddr);
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
