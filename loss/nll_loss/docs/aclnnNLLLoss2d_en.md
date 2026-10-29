# aclnnNLLLoss2d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:33:17.194Z -->

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

- Description: Computes the negative log-likelihood loss.

- Formula:

  When `reduction` is `none`:

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = - w_{y_n} x_{n,y_n}, \quad
  w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignoreIndex}\},
  $$

  where $x$ is self, $y$ is target, $w$ is weight, and $N$ is the batch size. If `reduction` is not `'none'`, then:

  $$
  \ell(x, y) = \begin{cases}
      \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
      \text{if reduction} = \text{`mean`}\\
      \sum_{n=1}^N l_n,  &
      \text{if reduction} = \text{`sum`}
  \end{cases}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnNLLLoss2dGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation flow of the operator, and then call the "aclnnNLLLoss2d" API to execute the computation.

```cpp
aclnnStatus aclnnNLLLoss2dGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weight,
    int64_t          reduction,
    int64_t          ignoreIndex,
    aclTensor       *out,
    aclTensor       *totalWeightOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnNLLLoss2d(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnNLLLoss2dGetWorkspaceSize

- **Parameters**

   <table style="undefined;table-layout: fixed; width: 1446px"><colgroup>
    <col style="width: 158px">
    <col style="width: 120px">
    <col style="width: 290px">
    <col style="width: 320px">
    <col style="width: 218px">
    <col style="width: 115px">
    <col style="width: 120px">
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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Tensor to be computed.</td>
        <td>Input x in the formula. The shape is 4D. The 2nd dimension is C, indicating the number of classes.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target (aclTensor*)</td>
        <td>Input</td>
        <td>Indicates the ground-truth labels.</td>
        <td><ul><li>y in the formula, with a 3-dimensional shape</li><li>The 1st dimension of target and the 1st dimension of self are equal, the 2nd dimension of target and the 3rd dimension of self are equal, and the 3rd dimension of target and the 4th dimension of self are equal</li><li>The value range of each element is [0, C - 1]</li></ul></td>
        <td>INT64, UINT8, INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight (aclTensor*)</td>
        <td>Input</td>
        <td>Represents the scaling weight of each class.</td>
        <td>w in the formula, with a shape of (C,).</td>
        <td>The data type must remain consistent with self.</td>
        <td>ND</td>
        <td>(C,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction (int64_t)</td>
        <td>Input</td>
        <td>Specifies the reduction to apply to the output.</td>
        <td><ul>Supports 0(none)|1(mean)|2(sum).<li>'none' indicates that no reduction will be applied</li><li>'mean' indicates that the sum of the output will be divided by the number of elements in the output</li><li>'sum' indicates that the output will be summed</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex (int64_t)</td>
        <td>Input</td>
         <td>Specifies a target value that is ignored and does not contribute to the input gradient.
        </td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>out in the formula.</td>
        <td>When reduction is 0 (none), the shape is the same as the shape of target; otherwise, the shape is (1,).</td>
        <td>The data type remains consistent with that of self.</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>totalWeightOut (aclTensor*)</td>
        <td>Output</td>
        <td>totalWeightOut in the formula.</td>
        <td>The output value is valid when reduction is non-zero (non-none), and the shape is (1,).</td>
        <td>The data type remains consistent with self.</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the size of the workspace that needs to be applied on the side of Device.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 300px">
      <col style="width: 134px">
      <col style="width: 716px">
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
      <td>The input self, target, weight, out, or totalWeightOut is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type or data format of self, target, weight, out, or totalWeightOut is not within the supported range.</td>
      </tr>
      <tr>
      <td>The data types of self, weight, out, and totalWeightOut are inconsistent.</td>
      </tr>
       <tr>
      <td>The shape and format of self, target, weight, out, and totalWeightOut are incorrect.</td>
      </tr>
      <tr>
      <td> The value of reduction is not within the range of 0-2.</td>
      </tr>
      </tbody>
      </table>

## aclnnNLLLoss2d

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
      <col style="width: 184px">
      <col style="width: 134px">
      <col style="width: 833px">
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
      <td>Memory address of the workspace applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnNLLLoss2dGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
    </tbody>
     </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation

  <!-- npu="A3,910b,910,310p,310b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: aclnnNLLLoss2d defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnNLLLoss2d defaults to a deterministic implementation.
  <!-- end id8 -->

## Example

The sample code is as follows and is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nll_loss2d.h"

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
  // Call aclrtMalloc to apply for memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data on the host side to the memory on the device side.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {1, 2, 3, 2};
  std::vector<int64_t> targetShape = {1, 3, 2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> outShape = {1, 3, 2};
  std::vector<int64_t> totalWeightOutShape = {1};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* totalWeightOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* out = nullptr;
  aclTensor* totalWeightOut = nullptr;
  std::vector<float> selfHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<int32_t> targetHostData = {1, 0, 1, 1, 2, 1};
  std::vector<float> weightHostData = {1.1, 1.2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};
  std::vector<float> totalWeightOutHostData = {0};
  int64_t reduction = 0;
  int64_t ignoreIndex = -100;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the totalWeightOut aclTensor.
  ret = CreateAclTensor(totalWeightOutHostData, totalWeightOutShape, &totalWeightOutDeviceAddr, aclDataType::ACL_FLOAT,
                        &totalWeightOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNLLLoss2d.
  ret = aclnnNLLLoss2dGetWorkspaceSize(self, target, weight, reduction, ignoreIndex, out, totalWeightOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnNLLLoss2d.
  ret = aclnnNLLLoss2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2d failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify it according to the actual API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar, modify as required by the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(out);
  aclDestroyTensor(totalWeightOut);

  // 7. Release device resources, modify as required by the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(totalWeightOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
