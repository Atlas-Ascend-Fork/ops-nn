# aclnnMatmulEmuSplitWeight

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:20:54.033Z -->

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

- Description: Simulates FP32-precision matrix multiplication by fusing dual BF16 GEMM paths. The FP32 weight is split offline into the high BF16 weight and the low residual BF16 weight. During inference, two BF16 GEMMs are performed and linearly combined. Activations remain BF16 throughout, and both matrix multiplications run on the Cube.

- Formula:

  $$
  Y = X \times W_{high} + scale \times (X \times W_{low})
  $$

  where $X$ is the BF16 activation matrix, $W_{high}$ is the high BF16 weight obtained by truncating the FP32 weight to BF16, $W_{low}$ is the low residual BF16 weight obtained by dividing the FP32 weight residual by scale and truncating it to BF16, and $scale$ is the scaling factor (typically 1/256 = 0.00390625).

- Offline weight splitting formula:

  $$
  W_{high} = \text{BF16}(W_{fp32})
  $$

  $$
  W_{low} = \text{BF16}\left(\frac{W_{fp32} - W_{high}.float()}{scale}\right)
  $$

## Function Prototype

Each operator is a two-phase API. Call "aclnnMatmulEmuSplitWeightGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnMatmulEmuSplitWeight" to perform the computation.

```cpp
aclnnStatus aclnnMatmulEmuSplitWeightGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *wHigh,
  const aclTensor *wLow,
  const aclTensor *y,
  float            wLowScale,
  int8_t           yDtype,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMatmulEmuSplitWeight(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnMatmulEmuSplitWeightGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
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
        <th>Dimensions</th>
        <th>Non-contiguous</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Indicates the activation matrix, which is $X$ in the corresponding formula.</td>
        <td><li>The data type must be BF16.</li></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>wHigh</td>
        <td>Input</td>
        <td>Indicates the high-order weight, which is $W_{high}$ in the corresponding formula.</td>
        <td><li>The data type must be BF16, same as x.</li></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>wLow</td>
        <td>Input</td>
        <td>Represents the low residual weight, corresponding to $W_{low}$ in the formula.</td>
        <td><li>The data type must be BF16, same as x.</li><li>The shape must be exactly the same as wHigh.</li></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>wLowScale</td>
        <td>Input</td>
        <td>Scaling factor, which is scale in the corresponding formula.</td>
        <td><li>Only 1/256 (0.00390625) is supported.</li></td>
        <td>float</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yDtype</td>
        <td>Input</td>
        <td>Data type identifier of the output.</td>
        <td><li>Only 0 (FP32) is supported.</li></td>
        <td>int8_t</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Indicates the output matrix of the computation, which corresponds to $Y$ in the corresponding formula.</td>
        <td><li>Only FP32 output (yDtype=0) is supported.</li><li>The shape is [M, N].</li></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Indicates the size of the workspace that the user needs to allocate on the Device.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>x, wHigh, wLow, or y is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>The data types of x, wHigh, and wLow are not BFLOAT16.</td>
      </tr>
      <tr>
        <td>The data type of y is not FLOAT32.</td>
      </tr>
      <tr>
        <td>yDtype is not 0.</td>
      </tr>
      <tr>
        <td>wLowScale is NaN or Inf, or wLowScale is not equal to 1/256 (0.00390625).</td>
      </tr>
      <tr>
        <td>The data format of the input is not ND.</td>
      </tr>
      <tr>
        <td>The input dimension is not 2, or the K dimension does not match (the K of x is inconsistent with the K of wHigh/wLow), or the shape of wLow is inconsistent with that of wHigh, or the shape of y does not match [M, N], or M, K, or N is out of the range (0, INT32_MAX].</td>
      </tr>
  </tbody></table>

## aclnnMatmulEmuSplitWeight

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnMatmulEmuSplitWeightGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
      </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnMatmulEmuSplitWeight defaults to a deterministic implementation.

  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: aclnnMatmulEmuSplitWeight defaults to a deterministic implementation.
  <!-- end id8 -->

- **Shape constraints**:
  - M, K, and N are in the range (0, INT32_MAX].
- **Transpose**: The aclnn API itself does not accept a transpose attribute. If transposition is required, the caller should perform permute on the tensor before passing it in. The operator supports non-contiguous tensors, so no additional contiguous operation is required.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_emu_split_weight.h"

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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Initialize device/stream.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs.
  // x: BF16 [128, 256], wHigh/wLow: BF16 [256, 128], y: FP32 [128, 128]
  std::vector<int64_t> xShape = {128, 256};
  std::vector<int64_t> wShape = {256, 128};
  std::vector<int64_t> yShape = {128, 128};
  void* xDeviceAddr = nullptr;
  void* wHighDeviceAddr = nullptr;
  void* wLowDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* wHigh = nullptr;
  aclTensor* wLow = nullptr;
  aclTensor* y = nullptr;

  // Construct BF16 data (float is used as a placeholder here; pass BF16 data in actual use).
  std::vector<float> xHostData(128 * 256, 1.0f);
  std::vector<float> wHighHostData(256 * 128, 1.0f);
  std::vector<float> wLowHostData(256 * 128, 0.0f);
  std::vector<float> yHostData(128 * 128, 0.0f);

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(wHighHostData, wShape, &wHighDeviceAddr, aclDataType::ACL_BF16, &wHigh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(wLowHostData, wShape, &wLowDeviceAddr, aclDataType::ACL_BF16, &wLow);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call aclnnMatmulEmuSplitWeight.
  // wLowScale: scale factor, only 1/256 (0.00390625) is supported.
  // yDtype: output data type, only 0 (FP32) is supported.
  float wLowScale = 0.00390625f;
  int8_t yDtype = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnMatmulEmuSplitWeightGetWorkspaceSize(x, wHigh, wLow, y, wLowScale, yDtype, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulEmuSplitWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMatmulEmuSplitWeight(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulEmuSplitWeight failed. ERROR: %d\n", ret); return ret);

  // 4. Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value.
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < std::min(size, 16L); i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release resources.
  aclDestroyTensor(x);
  aclDestroyTensor(wHigh);
  aclDestroyTensor(wLow);
  aclDestroyTensor(y);
  aclrtFree(xDeviceAddr);
  aclrtFree(wHighDeviceAddr);
  aclrtFree(wLowDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
