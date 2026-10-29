# aclnnRepeatInterleaveGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:38:26.067Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description: backward propagation of the repeatInterleave operator, which performs ReduceSum on the axis dimension of the yGrad tensor according to repeats.

- Example:
    Assume tensor yGrad is ([[a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>, d<sub>1</sub>, e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub>, c<sub>2</sub>, d<sub>2</sub>, e<sub>2</sub>, f<sub>2</sub>]]), repeats is ([1, 2, 2, 1]), and axis is 1. Then the resulting tensor is tensor([[a<sub>1</sub>, b<sub>1</sub> + c<sub>1</sub>, d<sub>1</sub> + e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub> + c<sub>2</sub>, d<sub>2</sub> + e<sub>2</sub>, f<sub>2</sub>]]). This performs ReduceSum on the axis dimension of tensor yGrad according to repeats.

    Assume that the yGrad tensor is ([[a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>, d<sub>1</sub>, e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub>, c<sub>2</sub>, d<sub>2</sub>, e<sub>2</sub>, f<sub>2</sub>]]), repeats is ([2]), and axis is 1.
    The final tensor is tensor([[a<sub>1</sub> + b<sub>1</sub>, c<sub>1</sub> + d<sub>1</sub>, e<sub>1</sub> + f<sub>1</sub>], [a<sub>2</sub> + b<sub>2</sub>, c<sub>2</sub> + d<sub>2</sub>, e<sub>2</sub> + f<sub>2</sub>]]). Perform ReduceSum on every two elements along the axis dimension of the yGrad tensor according to the value of repeats.
    Note: This use case is equivalent to repeats being (2).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnRepeatInterleaveGradGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnRepeatInterleaveGrad" API to perform the computation.

 ```cpp
  aclnnStatus aclnnRepeatInterleaveGradGetWorkspaceSize(
    const aclTensor *yGrad,
    const aclTensor *repeats,
    int64_t          axis,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
  ```

  ```cpp
  aclnnStatus aclnnRepeatInterleaveGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
  ```

## aclnnRepeatInterleaveGradGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1714px"><colgroup>
  <col style="width: 138px">
  <col style="width: 120px">
  <col style="width: 304px">
  <col style="width: 424px">
  <col style="width: 291px">
  <col style="width: 132px">
  <col style="width: 160px">
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
      <td>yGrad</td>
      <td>Input</td>
      <td>Input tensor to be reduced by ReduceSum in Function.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>repeats</td>
      <td>Input</td>
      <td>Number of repetitions.</td>
      <td>repeats can only be a 0D / 1D tensor. If repeats is a 1D tensor whose size is 1, broadcasting is supported. If repeats is a 1D tensor whose size is greater than 1, the sum of its elements must equal the size of the axis dimension of yGrad. Empty tensors are not supported.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td></td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>The dimension along which ReduceSum is performed.</td>
      <td>The value range of axis is [-n, n), where n is the number of dimensions of yGrad.</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output tensor completed by ReduceSum in Function.</td>
      <td>For shape constraints, see Constraints.</td>
      <td>Same as yGrad</td>
      <td>ND</td>
      <td>-</td>
      <td></td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size required on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The data type of yGrad supports FLOAT16, BFLOAT16, and FLOAT.

  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1039px"><colgroup>
  <col style="width: 292px">
  <col style="width: 138px">
  <col style="width: 609px">
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
      <td>The passed-in yGrad, repeats, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of yGrad and repeats are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of yGrad and out are different.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>repeats is an empty tensor.</td>
    </tr>
  </tbody>
  </table>

## aclnnRepeatInterleaveGrad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1045px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 794px">
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
      <td>Size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnRepeatInterleaveGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnRepeatInterleaveGrad defaults to a deterministic implementation.

The following requirements must be met during computation:

- If repeats is a 0D tensor or a 1D tensor with size 1, the element value of repeats must be a divisor of the dimension of yGrad on axis.
    If it is a 1D tensor with a size greater than 1, the sum of the elements in repeats must equal the dimension of yGrad along axis.
    The values in the repeats tensor must be natural numbers.
- The shape size of out satisfies the shape size after ReduceSum along the axis of yGrad.
    For example, if the shape of yGrad is [64], repeats is [2], and axis is 0, the shape of out can be [32], [2, 16], [2, 4, 4], and so on, as long as the size of the shape of out is 32.

## Example

The following sample code is for reference only. For details about how to compile and run it, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat_interleave_grad.h"

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
  // Boilerplate code for resource initialization.
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

  // Calculate the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. You need to customize the construction based on the API.
  std::vector<int64_t> yGradShape = {4, 3};
  std::vector<int64_t> repeatsShape = {2};
  std::vector<int64_t> outShape = {2, 3};
  void* yGradDeviceAddr = nullptr;
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* yGrad = nullptr;
  aclTensor* repeats = nullptr;
  aclTensor* out = nullptr;
  int64_t axis = 0;
  std::vector<float> yGradHostData = {3, 4, 5, 3, 4, 5, -3, -4, -5, -3, -4, -5};
  std::vector<int64_t> repeatsHostData = {2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};

  // Create the yGrad aclTensor.
  ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT, &yGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the repeats aclTensor.
  ret = CreateAclTensor(repeatsHostData, repeatsShape, &repeatsDeviceAddr, aclDataType::ACL_INT64, &repeats);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API and replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRepeatInterleaveGrad.
  ret = aclnnRepeatInterleaveGradGetWorkspaceSize(yGrad, repeats, axis, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRepeatInterleaveGrad.
  ret = aclnnRepeatInterleaveGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to the host. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(yGrad);
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(yGradDeviceAddr);
  aclrtFree(repeatsDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
