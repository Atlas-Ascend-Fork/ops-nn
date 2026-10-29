# aclnnWeightQuantBatchMatmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:13:29.572Z -->

**Note: This API will be deprecated in later versions. Use the latest aclnnWeightQuantBatchMatmulV3 API instead.**

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: Pseudo-quantization is used to quantize mat2 in self * mat2 (matmul/batchmatmul).
- Formula:

  $$
  result = self@mat2+bias
  $$

## Function Prototype

- [Two-phase API](../../../docs/en/context/two_phase_api.md). Call the aclnnWeightQuantBatchMatmulGetWorkspaceSize API first to obtain the input parameters and calculate the required workspace size based on the calculation process, and then call the aclnnWeightQuantBatchMatmul API to execute the calculation.

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *diagonalMatrix,
  const aclTensor *deqOffset,
  const aclTensor *deqScale,
  const aclTensor *addOffset,
  const aclTensor *mulScale,
  const aclTensor *bias,
  bool             transposeX1,
  bool             transposeX2,
  float            antiquantScale,
  float            antiquantOffset,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmul(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnWeightQuantBatchMatmulGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed;width: 1555px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 350px">
    <col style="width: 150px">
    <col style="width: 120px">
    <col style="width: 200px">
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
      <td>x1</td>
      <td>Input</td>
      <td>Input self in the formula.</td>
      <td>Only 2 dimensions are supported, the batch axis is not supported. Must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Processed to obtain mat2 in the formula.</td>
      <td>Only two dims are supported and the batch axis is not supported, but it must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x1.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diagonalMatrix</td>
      <td>Input</td>
      <td>x2 is dequantized to obtain the input mat2 in the formula.</td>
      <td>The shape is (32, 32), which is an identity matrix. When m > 64, it does not participate in the calculation and can be empty.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2D, shape (32, 32)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqOffset</td>
      <td>Input</td>
      <td>Input mat2 in the formula, dequantized from x2. Calculated from addOffset, antiquantOffset, and antiquantScale. For the calculation method, see the sample code.</td>
      <td>Supported shapes: 1, n, (1, 1), (1, n), or (n, 1), which must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2. When m > 64, this parameter is not involved in the calculation and can be empty.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>n, (1, 1), (1, n), or (n, 1)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>Input</td>
      <td>Dequantized from x2 to obtain mat2 in the formula. It is calculated by the aclnnTransQuantParam API. For the calculation method, see the sample code.</td>
      <td>Shapes 1, n, (1, 1), (1, n), and (n, 1) are supported. It must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2. If m > 64, this parameter is not involved in the calculation and can be empty.</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>1, n, (1, 1), (1, n), or (n, 1)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>addOffset</td>
      <td>Input</td>
      <td>Dequantizes x2 to obtain input mat2 in the formula.</td>
      <td>shape supports 1, n, (1, 1), (1, n), or (n, 1), and must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2. Not involved in the calculation when m < 64; can be empty in any case.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1, n, (1, 1), (1, n), or (n, 1)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mulScale</td>
      <td>Input</td>
      <td>Dequantizes x2 to obtain input mat2 in the formula.</td>
      <td>Shape supports 1, n, (1, 1), (1, n), or (n, 1), and must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2. Not involved in calculation when m < 64, and can be empty in any case.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1, n, (1, 1), (1, n), or (n, 1)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Input bias in the formula.</td>
      <td>The dimension is 1D and the value equals N. Can be empty.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Specifies whether x1 is transposed.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Indicates whether x2 is transposed.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScale</td>
      <td>Input</td>
      <td>Obtains mat2 in the formula by dequantizing x2.</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffset</td>
      <td>Input</td>
      <td>Obtains mat2 in the formula by dequantizing x2.</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The result in the formula.</td>
      <td>The data type must be a type that x1 and x2 can be converted to after deduction. The shape must be the shape of x1 and x2 after broadcasting.</td>
      <td>FLOAT16,INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output parameter</td>
      <td>Size of the workspace to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator calculation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x1 or x2, diagonalMatrix (when m < 64), deqOffset (when m < 64), or deqScale (when m < 64) is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of a non-empty tensor is out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnWeightQuantBatchMatmul

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1143px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 825px">
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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device side, obtained by the first-phase API aclnnWeightQuantBatchMatmulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the calculation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: aclnnWeightQuantBatchMatmul defaults to a deterministic implementation.

  <!-- end id7 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul.h"

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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> x1Shape = {128, 4};
  std::vector<int64_t> x2Shape = {4, 4};
  std::vector<int64_t> addOffsetShape = {4};
  std::vector<int64_t> mulScaleShape = {4};
  std::vector<int64_t> diagonalMatrixShape = {32, 32};
  std::vector<int64_t> deqOffsetShape = {4};
  std::vector<int64_t> deqScaleShape = {4};
  std::vector<int64_t> outShape = {128, 4};

  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* addOffsetDeviceAddr = nullptr;
  void* mulScaleDeviceAddr = nullptr;
  void* diagonalMatrixDeviceAddr = nullptr;
  void* deqOffsetDeviceAddr = nullptr;
  void* deqScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* x1Fp16DeviceAddr = nullptr;
  void* addOffsetFp16DeviceAddr = nullptr;
  void* mulScaleFp16DeviceAddr = nullptr;
  void* outFp16DeviceAddr = nullptr;

  std::vector<float> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<float> x2HostData = {1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1};

  std::vector<float> outHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  bool transposeX1 = false;
  bool transposeX2 = false;
  float antiquantOffset = 0;
  float antiquantScale = 1;

  std::vector<float> addOffsetHostData = {1.0, 1.0, 1.0, 0.0};
  float* addOffsetData = addOffsetHostData.data();
  uint64_t addOffsetSize = 4;
  std::vector<float> mulScaleHostData = {2.0, 2.0, 1.0, 1.0};
  float* mulScaleData = mulScaleHostData.data();
  uint64_t mulScaleSize = 4;

  // diagonalMatrixData
  uint64_t n = 32;
  uint64_t diagonalMatrixSize = n*n;
  int8_t *diagonalMatrixData = (int8_t *)calloc(diagonalMatrixSize, sizeof(int32_t));
  for (int64_t i = 0; i < n; i++) {
    diagonalMatrixData[i * n + i] = 1;
  }
  std::vector<int8_t> diagonalMatrixHostData(diagonalMatrixData, diagonalMatrixData + diagonalMatrixSize);

  // Get deqOffset
  uint64_t deqOffsetSize = addOffsetSize;
  int32_t *deqOffsetData = (int32_t *)calloc(deqOffsetSize, sizeof(int32_t));
  for (int64_t i = 0; i < deqOffsetSize; i++) {
    deqOffsetData[i] = static_cast<int32_t>(round(addOffsetData[i] / antiquantScale - antiquantOffset));
  }
  std::vector<int32_t> deqOffsetHostData(deqOffsetData, deqOffsetData + deqOffsetSize);

  // Get deqScale
  uint64_t deqScaleSize = 0;
  uint64_t *deqScaleData = nullptr;
  ret = aclnnTransQuantParam(mulScaleData, mulScaleSize, addOffsetData, addOffsetSize, &deqScaleData, &deqScaleSize);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParam failed. ERROR: %d\n", ret); return ret);
  std::vector<uint64_t> deqScaleHostData(deqScaleData, deqScaleData + deqScaleSize);

  // create aclTensor
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* addOffset = nullptr;
  aclTensor* mulScale = nullptr;
  aclTensor* diagonalMatrix = nullptr;
  aclTensor* deqOffset = nullptr;
  aclTensor* deqScale = nullptr;
  aclTensor* out = nullptr;
  aclTensor* x1Fp16 = nullptr;
  aclTensor* addOffsetFp16 = nullptr;
  aclTensor* mulScaleFp16 = nullptr;
  aclTensor* outFp16 = nullptr;

  // Create the x1 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the x1Fp16 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Fp16DeviceAddr, aclDataType::ACL_FLOAT16, &x1Fp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the x2 aclTensor.
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the addOffset aclTensor.
  ret = CreateAclTensor(addOffsetHostData, addOffsetShape, &addOffsetDeviceAddr, aclDataType::ACL_FLOAT, &addOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the addOffsetFp16 aclTensor.
  ret = CreateAclTensor(addOffsetHostData, addOffsetShape, &addOffsetFp16DeviceAddr, aclDataType::ACL_FLOAT16, &addOffsetFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mulScale aclTensor.
  ret = CreateAclTensor(mulScaleHostData, mulScaleShape, &mulScaleDeviceAddr, aclDataType::ACL_FLOAT, &mulScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mulScaleFp16 aclTensor.
  ret = CreateAclTensor(mulScaleHostData, mulScaleShape, &mulScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &mulScaleFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the diagonalMatrix aclTensor.
  ret = CreateAclTensor(diagonalMatrixHostData, diagonalMatrixShape, &diagonalMatrixDeviceAddr, aclDataType::ACL_INT8, &diagonalMatrix);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the deqOffset aclTensor.
  ret = CreateAclTensor(deqOffsetHostData, deqOffsetShape, &deqOffsetDeviceAddr, aclDataType::ACL_INT32, &deqOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the deqScale aclTensor.
  ret = CreateAclTensor(deqScaleHostData, deqScaleShape, &deqScaleDeviceAddr, aclDataType::ACL_UINT64, &deqScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the outFp16 aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outFp16DeviceAddr, aclDataType::ACL_FLOAT16, &outFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  // Example of the aclnnWeightQuantBatchMatmul API call
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // aclnn cast fp16
  //x1
  ret = aclnnCastGetWorkspaceSize(x1, aclDataType::ACL_FLOAT16, x1Fp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // addOffset
  ret = aclnnCastGetWorkspaceSize(addOffset, aclDataType::ACL_FLOAT16, addOffsetFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // mulScale
  ret = aclnnCastGetWorkspaceSize(mulScale, aclDataType::ACL_FLOAT16, mulScaleFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Call the first-phase API of aclnnWeightQuantBatchMatmul.
  ret = aclnnWeightQuantBatchMatmulGetWorkspaceSize(x1Fp16, x2, diagonalMatrix, deqOffset, deqScale, addOffsetFp16, mulScaleFp16, nullptr, transposeX1, transposeX2, antiquantScale, antiquantOffset, outFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);


  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnWeightQuantBatchMatmul.
  ret = aclnnWeightQuantBatchMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // fp16 to fp 32
  ret = aclnnCastGetWorkspaceSize(outFp16, aclDataType::ACL_FLOAT, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host memory. The code needs to be modified based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. The code needs to be modified based on the specific API definition.
  aclDestroyTensor(x1);
  aclDestroyTensor(x1Fp16);
  aclDestroyTensor(x2);
  aclDestroyTensor(addOffset);
  aclDestroyTensor(addOffsetFp16);
  aclDestroyTensor(mulScaleFp16);
  aclDestroyTensor(mulScale);
  aclDestroyTensor(out);
  aclDestroyTensor(outFp16);


  // 7. Release device resources. The code needs to be modified based on the specific API definition.
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(addOffsetDeviceAddr);
  aclrtFree(deqScaleDeviceAddr);
  aclrtFree(mulScaleDeviceAddr);
  aclrtFree(diagonalMatrixDeviceAddr);
  aclrtFree(deqOffsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(x1Fp16DeviceAddr);
  aclrtFree(addOffsetFp16DeviceAddr);
  aclrtFree(mulScaleFp16DeviceAddr);
  aclrtFree(outFp16DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  free(diagonalMatrixData);
  free(deqOffsetData);
  free(deqScaleData);
  return 0;
}
```
