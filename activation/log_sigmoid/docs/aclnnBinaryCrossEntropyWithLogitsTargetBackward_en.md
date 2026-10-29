# aclnnBinaryCrossEntropyWithLogitsTargetBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T12:03:44.104Z -->

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

Computes logits from input self, and performs backward propagation of [BCELoss](../../../loss/sigmoid_cross_entropy_with_logits_v2/docs/aclnnBinaryCrossEntropyWithLogits.md) with respect to target using the obtained values and the label value target.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size, and then call the "aclnnBinaryCrossEntropyWithLogitsTargetBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(
  const aclTensor *gradOutput,
  const aclTensor *self,
  const aclTensor *target,
  const aclTensor *weightOptional,
  const aclTensor *posWeightOptional,
  int64_t          reduction,
  aclTensor       *gradTarget,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 350px">
  <col style="width: 220px">
  <col style="width: 115px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Gradient value from the previous step of backward propagation.</td>
      <td class="tg-0pky">The shape must be able to be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to the shape of self.</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Computation result of the preceding layer in forward propagation.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Label value.</td>
      <td class="tg-0pky">The shape must be the same as the shape of self.</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">Same as self</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOptional (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight of binary cross entropy.</td>
      <td class="tg-0pky">The shape must be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to the shape of self. If weightOptional is empty, a tensor filled with ones is created based on the shape of self.</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">posWeightOptional (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight of the positive class.</td>
      <td class="tg-0pky">The shape can be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to the shape of self. When posWeightOptional is empty, an all-1 tensor is created with the shape of self.</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Specifies the reduce operation applied to the backward gradient computation result of binary cross entropy.</td>
      <td class="tg-0pky">Only 0, 1, and 2 are supported. 0 indicates no operation; 1 indicates the mean of the result; 2 indicates the sum of the result.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">gradTarget (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Stores the gradient computation result.</td>
      <td class="tg-0lax">The inferred dtype of the input Tensor. It must be convertible to the dtype of gradTarget.</td>
      <td class="tg-0lax">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0lax">Consistent with self</td>
      <td class="tg-0lax">Consistent with self</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be allocated on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the op executor, which contains the computation process of the operator.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table class="tg" style="undefined;table-layout: fixed; width: 950px"><colgroup>
  <col style="width: 276px">
  <col style="width: 132px">
  <col style="width: 542px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Value</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">gradOutput, self, target, or gradTarget is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="7">161002</td>
      <td class="tg-0pky">The data types or data formats of gradOutput, self, target, and gradTarget are not in the supported range.</td>
    </tr>
    <tr>
      <td class="tg-0pky">If weightOptional and posWeightOptional are not null pointers, their data types or data formats are not in the supported range.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shapes of self, target, and gradTarget are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">If weightOptional and posWeightOptional are not null pointers, their shapes cannot be broadcast to the shape of self.</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradOutput cannot be broadcast to the shape of self.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The value of reduction is not 0, 1, or 2.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The dimensions of gradOutput, self, target, and gradTarget are greater than 8.</td>
    </tr>
  </tbody></table>

## aclnnBinaryCrossEntropyWithLogitsTargetBackward

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 200px">
    <col style="width: 162px">
    <col style="width: 882px">
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
        <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computing process.</td>
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
  - aclnnBinaryCrossEntropyWithLogitsTargetBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy_with_logits_target_backward.h"

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
  // Boilerplate, resource initialization.
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
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
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> weightShape = {2, 2};
  std::vector<int64_t> posWeightShape = {2, 2};
  std::vector<int64_t> gradTargetShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* posWeightDeviceAddr = nullptr;
  void* gradTargetDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* gradTarget = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* posWeight = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {0.1, 0.1, 0.1, 0.1};
  std::vector<float> weightHostData = {0, 1, 2, 3};
  std::vector<float> posWeightHostData = {0, 1, 2, 3};
  std::vector<float> gradTargetHostData = {0, 0, 0, 0};
  int64_t reduction = 0;

  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the posWeight aclTensor.
  ret = CreateAclTensor(posWeightHostData, posWeightShape, &posWeightDeviceAddr, aclDataType::ACL_FLOAT, &posWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradTarget aclTensor.
  ret = CreateAclTensor(gradTargetHostData, gradTargetShape, &gradTargetDeviceAddr, aclDataType::ACL_FLOAT, &gradTarget);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Example of calling the aclnnBinaryCrossEntropyWithLogitsTargetBackward API
  // 3. Call the CANN operator library API. Replace with the specific API name.
  // Call the first-phase API of aclnnBinaryCrossEntropyWithLogitsTargetBackward.
  ret = aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(gradOutput, self, target, weight, posWeight,
      reduction, gradTarget, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize failed. ERROR: %d\n",
                                          ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBinaryCrossEntropyWithLogitsTargetBackward.
  ret = aclnnBinaryCrossEntropyWithLogitsTargetBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsTargetBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result in the Device-side memory to the Host side. Modify based on the specific API definition.
  auto size = GetShapeSize(gradTargetShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradTargetDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify based on the specific API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(posWeight);
  aclDestroyTensor(gradTarget);

  // 7. Release device resources. Modify based on the specific API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(posWeightDeviceAddr);
  aclrtFree(gradTargetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
