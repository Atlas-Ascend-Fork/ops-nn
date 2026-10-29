# aclnnSoftmaxCrossEntropyWithLogits

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T14:24:28.013Z -->

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

- Description: Computes the Cross-Entropy loss of softmax and cross entropy, and provides the backward gradient with respect to the input logits.
- Formula:

  $$
  \text{loss} = - \sum_i y_i \cdot \log(\text{softmax}(x_i))
  $$

$$
\text{backprop} = \text{softmax}(x_i) - y_i
$$

Where $x_i$ corresponds to the input features, and $y_i$ corresponds to the input labels.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation flow of the operator, and then call the "aclnnSoftmaxCrossEntropyWithLogits" API to execute the computation.

```Cpp
aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(
  const aclTensor* features,
  aclTensor*       labels,
  aclTensor*       loss,
  aclTensor*       backprop,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSoftmaxCrossEntropyWithLogits(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
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
      <td>features (aclTensor*)</td>
      <td>Input</td>
      <td>Output logits of the model, x_i in the formula.</td>
      <td>Supports empty Tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>labels (aclTensor*)</td>
      <td>Input</td>
      <td>Input labels, y_i in the formula.</td>
      <td><ul><li>The shape and data type are the same as those of features.</li><li>Supports empty tensors.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>loss (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the cross-entropy loss of softmax and cross entropy, that is, loss in the formula.</td>
      <td><ul><li>The data type is the same as that of features.</li><li>Supports empty tensors.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>backprop (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the backward gradient of input logits, that is, backprop in the formula.</td>
      <td><ul><li>The data type is the same as that of features.</li><li>Supports empty tensors.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
       <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
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

   <!-- npu="910,310p" id7 -->
   - <term>Atlas training products</term>, <term>Atlas inference products</term>: The supported data types are FLOAT and FLOAT16.
   <!-- end id7 -->

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
      <td>Any of the input features, labels, loss, or backprop is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The data types of features and labels are not within the supported range.</td>
    </tr>
  </tbody></table>

## aclnnSoftmaxCrossEntropyWithLogits

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnSoftmaxCrossEntropyWithLogits defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_softmax_cross_entropy_with_logits.h"

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
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // boilerplate for resource initialization
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

  // Calculate the strides of the contiguous tensor.
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
  // 1. (boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction must be customized based on the API.
  std::vector<int64_t> featuresShape = {16, 4};
  std::vector<int64_t> labelsShape = {16, 4};

  std::vector<float> featuresHostData(16 * 4, 1);
  std::vector<float> labelsHostData(16 * 4, 2);

  void* featuresDeviceAddr = nullptr;
  void* labelsDeviceAddr = nullptr;

  aclTensor* features = nullptr;
  aclTensor* labels = nullptr;

  ret = CreateAclTensor(featuresHostData, featuresShape, &featuresDeviceAddr, aclDataType::ACL_FLOAT, &features);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(labelsHostData, labelsShape, &labelsDeviceAddr, aclDataType::ACL_FLOAT, &labels);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> lossShape = {16};
  std::vector<float> lossHostData(16, 8);
  aclTensor* loss = nullptr;
  void* lossDeviceAddr = nullptr;

  std::vector<int64_t> backpropShape = {16,4};
  std::vector<float> backpropHostData(16 * 4, 9);
  aclTensor* backprop = nullptr;
  void* backpropDeviceAddr = nullptr;

  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(backpropHostData, backpropShape, &backpropDeviceAddr, aclDataType::ACL_FLOAT, &backprop);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnSoftmaxCrossEntropyWithLogits.
  ret = aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(features, labels, loss, backprop,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnSoftmaxCrossEntropyWithLogits.
  ret = aclnnSoftmaxCrossEntropyWithLogits(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmaxCrossEntropyWithLogits failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result on the device memory to the host, and modify it based on the specific API definition.
  PrintOutResult(lossShape, &lossDeviceAddr);

  // 6. Release aclTensor and aclScalar, and modify it based on the specific API definition.
  aclDestroyTensor(features);
  aclDestroyTensor(labels);
  aclDestroyTensor(loss);
  aclDestroyTensor(backprop);

  // 7. Release device resources, and modify it based on the specific API definition.
  aclrtFree(featuresDeviceAddr);
  aclrtFree(labelsDeviceAddr);
  aclrtFree(lossDeviceAddr);
  aclrtFree(backpropDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
