# aclnnInplaceQuantScatterV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:34:04.612Z -->

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

A fusion of the [Quantize](https://gitcode.com/cann/ops-nn/blob/master/quant/quantize/docs/aclnnQuantize.md) operator and the [Scatter](https://gitcode.com/cann/ops-nn/blob/master/index/scatter/docs/aclnnInplaceScatterUpdate.md) operator. First, updates is quantized along the quantAxis axis: quantScales scales updates, and quantZeroPoints applies the offset. Then, the values in the quantized updates update the corresponding values in selfRef one by one along the specified axis according to the indices tensor indices. Compared with aclnnInplaceQuantScatter, a roundMode input is added.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnInplaceQuantScatterV2GetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnInplaceQuantScatterV2" to execute the computation.

```cpp
aclnnStatus aclnnInplaceQuantScatterV2GetWorkspaceSize(
  aclTensor       *selfRef,
  const aclTensor *indices,
  const aclTensor *updates,
  const aclTensor *quantScales,
  const aclTensor *quantZeroPoints,
  int64_t          axis,
  int64_t          quantAxis,
  int64_t          reduction,
  const char       *roundMode,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor);
```

```cpp
aclnnStatus aclnnInplaceQuantScatterV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream);
```

## aclnnInplaceQuantScatterV2GetWorkspaceSize

- **Parameters**:
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>selfRef (aclTensor*)</td>
      <td>Input|Output</td>
      <td>Source data tensor.</td>
      <td>Empty tensor is supported.</td>
      <td>INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
      <td>3-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices (aclTensor*)</td>
      <td>Input</td>
      <td>Index tensor.</td>
      <td><ul><li>When the shape of indices is 1-dimensional, the value range of indices is [0, selfRef.shape(axis) - updates.shape(axis)).</li><li>When the shape of indices is 2-dimensional, the value range of the 0th element of each item in indices is [0, selfRef.shape(0)), and the value range of the 1st element of each item in indices is [0, selfRef.shape(axis) - updates.shape(axis)).</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1,2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>updates (aclTensor*)</td>
      <td>Input</td>
      <td>Update data tensor.</td>
      <td><ul><li>The number of dimensions of updates must be the same as that of selfRef. The dimension size of its first dimension must be equal to that of the first dimension of indices, and must not be greater than the dimension size of the first dimension of selfRef.</li><li>The axis size of updates must not be greater than the axis size of selfRef.</li><li>The dimension sizes of the remaining dimensions must be equal to those of the corresponding dimensions of selfRef.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScales (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization scale tensor.</td>
      <td>The number of elements in quantScales must equal the axis size of updates along quantAxis.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantZeroPoints (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization offset tensor.</td>
      <td>The number of elements in quantZeroPoints must be equal to the axis size of updates on quantAxis.</td>
      <td>BFLOAT16, INT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Axis on updates used for updating.</td>
      <td>Value range: [-len(updates.shape) + 1, -1) or [1, len(updates.shape) - 1).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantAxis (int64_t)</td>
      <td>Input</td>
      <td>Axis on updates used for quantization.</td>
      <td>The value can be -1 or len(updates.shape) - 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction (int64_t)</td>
      <td>Input</td>
      <td>Specifies the data operation mode.</td>
      <td>Currently, only the value 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>Input</td>
      <td>Round in the Quantize formula. Specifies the mode of data conversion.</td>
      <td><ul><li>The {"rint", "round", "hybrid"} modes are supported.</li><li>Only {"rint"} is supported when the data type of selfRef is INT8/FLOAT8_E4M3FN/FLOAT8_E5M2.</li><li>{"round", "hybrid"} are supported when the data type is HIFLOAT8.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
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

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
  <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If a parameter that is a mandatory input, output, or mandatory attribute is a null pointer, 161001 is returned.</td>
  </tr>
  <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of selfRef, indices, updates, quantScales, or quantZeroPoints are not within the supported range.</td>
  </tr>
  <tr>
    <td>The data type combinations of selfRef, indices, updates, quantScales, and quantZeroPoints are not within the supported range. For details about the combinations, see Constraints.</td>
  </tr>
  <tr>
    <td>selfRef and updates have different numbers of dimensions.</td>
  </tr>
  <tr>
    <td>axis, quantAxis, or roundMode is not within the supported range.</td>
  </tr>
  <tr>
    <td>Unsupported product model.</td>
  </tr>
  </tbody>
  </table>

## aclnnInplaceQuantScatterV2

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
      <td>Workspace size requested on the device, obtained by calling the first-phase API aclnnInplaceQuantScatterV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
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

- deterministic computation:
  - aclnnInplaceQuantScatterV2 defaults to a deterministic implementation.

- The dimension of indices can only be 1D or 2D. If it is 2D, the size of the second dimension must be 2. Out-of-bounds indices are not supported and are not validated. The selfRef data segments mapped by indices must not overlap; otherwise, the results may vary across multiple executions due to multi-core concurrency.
- The data type input combinations of selfRef, indices, updates, quantScales, and quantZeroPoints include:

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    | selfRef | Indices | Updates | quantScales | quantZeroPoints |
    |---|---|---|---|---|
    |INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8|INT32|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8|INT64|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8|INT64|FLOAT16|FLOAT32|INT32|

  <!-- end id7 -->

## Example

For reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_scatter_v2.h"

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
  // Boilerplate code. Initialize resources.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Construct them as required by the API.
  std::vector<int64_t> selfRefShape = {1, 1, 32};
  std::vector<int64_t> indicesShape = {1};
  std::vector<int64_t> updatesShape = {1, 1, 32};
  std::vector<int64_t> quantScalesShape = {1, 1, 32};
  std::vector<int64_t> quantZeroPointsShape = {1, 1, 32};
  void* selfRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  void* quantScalesDeviceAddr = nullptr;
  void* quantZeroPointsDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  aclTensor* quantScales = nullptr;
  aclTensor* quantZeroPoints = nullptr;
  std::vector<int8_t> selfRefHostData(32, 0);
  std::vector<int32_t> indicesHostData{0};
  std::vector<uint16_t> updatesHostData(32, 15360);  // FP16 1.0 = 0x3C00 = 15360
  std::vector<float> quantScalesHostData(32, 0.5);
  std::vector<int32_t> quantZeroPointsHostData(32, 0);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;

  // Create the selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_INT8, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT16, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the quantScales aclTensor.
  ret = CreateAclTensor(quantScalesHostData, quantScalesShape, &quantScalesDeviceAddr, aclDataType::ACL_FLOAT,
                        &quantScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the quantZeroPoints aclTensor.
  ret = CreateAclTensor(quantZeroPointsHostData, quantZeroPointsShape, &quantZeroPointsDeviceAddr,
                        aclDataType::ACL_INT32, &quantZeroPoints);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  const char* roundMode = "rint";
  // Call the first-phase API of aclnnInplaceQuantScatterV2.
  ret = aclnnInplaceQuantScatterV2GetWorkspaceSize(selfRef, indices, updates, quantScales, quantZeroPoints, axis,
                                                 quantAxis, reduction, roundMode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnInplaceQuantScatterV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceQuantScatterV2.
  ret = aclnnInplaceQuantScatterV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceQuantScatterV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result on the device side to the host side. Change it based on the API definition of the specific API.
  auto size = GetShapeSize(selfRefShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  aclDestroyTensor(quantScales);
  aclDestroyTensor(quantZeroPoints);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
  aclrtFree(quantScalesDeviceAddr);
  aclrtFree(quantZeroPointsDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
