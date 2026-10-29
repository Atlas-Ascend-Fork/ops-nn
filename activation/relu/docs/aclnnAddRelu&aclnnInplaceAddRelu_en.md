# aclnnAddRelu&aclnnInplaceAddRelu

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T13:07:07.924Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Performs activation on the result after completing the addition computation.
- Formula:

  $$
  out_i = self_i+alpha \times other_i
  $$

  $$
  relu(self) = \begin{cases} self, & self\gt 0 \\ 0, & self\le 0 \end{cases}
  $$

## Function Prototype

- aclnnAddRelu and aclnnInplaceAddRelu implement the same functionality. The differences are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnAddRelu: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceAddRelu: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnAddReluGetWorkspaceSize" or "aclnnInplaceAddReluGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call "aclnnAddRelu" or "aclnnInplaceAddRelu" to execute the computation.

  ```Cpp
  aclnnStatus aclnnAddReluGetWorkspaceSize(
    const aclTensor   *self,
    const aclTensor   *other,
    aclScalar         *alpha,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
  ```

  ```Cpp
  aclnnStatus aclnnAddRelu(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddReluGetWorkspaceSize(
    aclTensor         *selfRef,
    const aclTensor   *other,
    aclScalar         *alpha,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddRelu(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
  ```

## aclnnAddReluGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
    <col style="width: 249px">
    <col style="width: 121px">
    <col style="width: 164px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 148px">
    <col style="width: 135px">
    <col style="width: 146px">
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
      <td>Input self in the formula, indicating the target tensor to be converted.</td>
      <td><ul><li>shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</li><li>The data type must satisfy the data type derivation rule with other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>Input other in the formula.</td>
      <td><ul><li>The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li><li>The data type must meet the data type derivation rule with self (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>alpha (aclScalar*)</td>
      <td>Input</td>
      <td>alpha in the formula.</td>
      <td>The data type must be convertible to the derived data type of self and other.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>out in the formula.</td>
      <td>The data type must be convertible from the data types derived for self and other, and the shape must be the shape after self and other are broadcast.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="910" id7 -->
    - <term>Atlas training products</term>: The data types of <code>self</code>, <code>other</code>, <code>alpha</code>, and <code>out</code> do not support BFLOAT16.
    <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
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
        <td>Null pointer passed for self, other, alpha, or out.</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>The data types of self and other are not within the supported range.</td>
      </tr>
      <tr>
        <td>The data types of self and other cannot be derived.</td>
      </tr>
      <tr>
        <td>The derived data type cannot be converted to the type of the specified output out.</td>
      </tr>
      <tr>
        <td>The shapes of self and other cannot be broadcast.</td>
      </tr>
      <tr>
        <td>alpha cannot be converted to the derived data type of self and other.</td>
      </tr>
    </tbody>
    </table>

## aclnnAddRelu

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Size of the workspace allocated on the Device side, obtained by the first-phase API aclnnAddReluGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddReluGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
    <col style="width: 249px">
    <col style="width: 121px">
    <col style="width: 164px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 148px">
    <col style="width: 135px">
    <col style="width: 146px">
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
      <td>self and out in the formula, indicating the target tensor to be converted.</td>
      <td>The data type must satisfy the data type derivation rule with other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and must be a data type that can be converted after derivation.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>Input other in the formula.</td>
      <td><ul><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef.</li><li>The data type must conform to the data type derivation rule with selfRef (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>alpha (aclScalar*)</td>
      <td>Input</td>
      <td>alpha in the formula.</td>
      <td>The data type must be convertible to the deduced data type of selfRef and other.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="910" id8 -->
    - <term>Atlas training products</term>: The data type of parameters `selfRef`, `other`, and `alpha` does not support BFLOAT16.
    <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
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
        <td>Null pointer passed for selfRef, other, or alpha.</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>The data types of selfRef and other are out of the supported range.</td>
      </tr>
      <tr>
        <td>Data type derivation cannot be performed on selfRef and other.</td>
      </tr>
      <tr>
        <td>The derived data type cannot be converted to the type of selfRef.</td>
      </tr>
      <tr>
        <td>The shapes of selfRef and other cannot be broadcast.</td>
      </tr>
      <tr>
        <td>alpha cannot be converted to the data type derived from selfRef and other.</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceAddRelu

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnInplaceAddReluGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
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

- Deterministic computation:
  - aclnnAddRelu&aclnnInplaceAddRelu default to a deterministic implementation.

- For scenarios where the data type of selfRef is INT8 and the data type of other is INT32:
    The cast operator has a precision issue when converting INT32 to INT8 (for details, see [aclnnCast](https://gitcode.com/cann/ops-math/blob/master/math/cast/docs/aclnnCast.md)). In this use case, the output precision cannot be guaranteed.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_relu.h"

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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. The construction must be customized based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  float alphaValue = 1.2f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnAddRelu API call example.
  // 3. Call the CANN operator library API.
  // Call the aclnnAddRelu first-phase API.
  ret = aclnnAddReluGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddReluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnAddRelu second-phase API.
  ret = aclnnAddRelu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRelu failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }


  // Release the workspace allocated by aclnnAddRelu to avoid memory leaks caused by pointer overwriting during subsequent in-place API allocation.
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    workspaceAddr = nullptr;
  }

  // Example of calling the aclnnInplaceAddRelu API
  // 3. Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceAddRelu\n");
  // Call the first-phase API of aclnnInplaceAddRelu.
  ret = aclnnInplaceAddReluGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddReluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceAddRelu.
  ret = aclnnInplaceAddRelu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddRelu failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize to wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify this part based on the specific API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }


  // 6. Release aclTensor and aclScalar. Modify this part based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. Release Device resources. Modify this part based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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
