# aclnnForeachSubListInplace

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T03:05:23.604Z -->

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

- Description: Updates the input tensor lists in place by performing element-wise subtraction on the two input tensor lists, with the subtraction coefficient adjusted by the alpha parameter, and writes the result back to the first tensor list in place.

- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}]\\
  x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  $$

  $$
  x1_i = x1_i - alpha \times x2_i (i=0,1,...n-1)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnForeachSubListInplaceGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnForeachSubListInplace" API to perform the computation.

```Cpp
aclnnStatus aclnnForeachSubListInplaceGetWorkspaceSize(
  aclTensorList           *x1Ref,
  const aclTensorList     *x2,
  const aclTensor         *alpha,
  uint64_t                *workspaceSize,
  aclOpExecutor          **executor)
```

```Cpp
aclnnStatus aclnnForeachSubListInplace(
  void               *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor      *executor,
  aclrtStream         stream)
```

## aclnnForeachSubListInplaceGetWorkspaceSize

- **Parameters**

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
      <td>x1Ref (aclTensorList*)</td>
      <td>Input/Output</td>
      <td>Indicates the input and output tensor list of the minuend in the subtraction operation, corresponding to `x1` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>All tensors in this parameter must have the same data type.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensorList*)</td>
      <td>Input</td>
      <td>Indicates the tensor list of the subtrahend in the subtraction operation, corresponding to `x2` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data types of all tensors in this parameter must be the same.</li><li>The data type, data format, and shape are the same as those of input `x1Ref`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the coefficient of the subtrahend in the subtraction operation, corresponding to `alpha` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>Contains one element.</li><li>The data type has a specific mapping with the data type of input `x1Ref`:<ul><li>When the data type of `x1Ref` is FLOAT32, FLOAT16, or INT32, the data type must be the same as that of `x1Ref`.</li><li>When the data type of `x1Ref` is BFLOAT16, the data type must be FLOAT32.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, INT32</td>
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
      <td>The passed-in x1Ref, x2, or alpha is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of x1Ref, x2, or alpha is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of x1Ref and x2 are inconsistent.</td></tr>
    <tr>
      <td>x1Ref or x2 contains a null tensor pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="5">561002</td>
      <td>The shapes of x1Ref or x2 do not meet the constraints.</td>
    </tr>
    <tr>
      <td>The tensors in x1Ref or x2 have inconsistent data types.</td></tr>
    <tr>
      <td>The dimensions of tensors in x1Ref or x2 exceed 8.</td></tr>
    <tr>
      <td>The number of elements in alpha is not 1.</td>
    </tr>
    <tr>
      <td>The numbers of tensors in x1Ref and x2 are inconsistent.</td>
    </tr>
  </tbody>
  </table>

## aclnnForeachSubListInplace

- **Parameters**

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
      <td>Size of the workspace requested on the device, obtained by the first-phase API aclnnForeachSubListInplaceGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnForeachSubListInplace defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_sub_list_inplace.h"

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
  // Boilerplate: resource initialization.
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
  // 1. (boilerplate) Device/stream initialization. Refer to the acl API documentation.
  // Specify deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> otherShape1 = {2, 3};
  std::vector<int64_t> otherShape2 = {1, 3};
  std::vector<int64_t> alphaShape = {1};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* other1DeviceAddr = nullptr;
  void* other2DeviceAddr = nullptr;
  void* alphaDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* other1 = nullptr;
  aclTensor* other2 = nullptr;
  aclTensor* alpha = nullptr;
  std::vector<float> input1HostData = {3, 5, 7, 4, 5, 9};
  std::vector<float> input2HostData = {5, 4, 1};
  std::vector<float> other1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> other2HostData = {7, 8, 9};
  std::vector<float> alphaValueHostData = {1.2f};
  // Create the input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create other1 aclTensor.
  ret = CreateAclTensor(other1HostData, otherShape1, &other1DeviceAddr, aclDataType::ACL_FLOAT, &other1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create other2 aclTensor.
  ret = CreateAclTensor(other2HostData, otherShape2, &other2DeviceAddr, aclDataType::ACL_FLOAT, &other2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create alpha aclTensor.
  ret = CreateAclTensor(alphaValueHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tempInput1{input1, input2};
  aclTensorList* tensorListInput1 = aclCreateTensorList(tempInput1.data(), tempInput1.size());
  std::vector<aclTensor*> tempInput2{other1, other2};
  aclTensorList* tensorListInput2 = aclCreateTensorList(tempInput2.data(), tempInput2.size());

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnForeachSubListInplace.
  ret = aclnnForeachSubListInplaceGetWorkspaceSize(tensorListInput1, tensorListInput2, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachSubListInplaceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnForeachSubListInplace.
  ret = aclnnForeachSubListInplace(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachSubListInplace failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values. Copy the results from device memory to host memory. Modify the code according to the specific API definition.
  auto size = GetShapeSize(selfShape1);
  std::vector<float> self1Data(size, 0);
  ret = aclrtMemcpy(self1Data.data(), self1Data.size() * sizeof(self1Data[0]), input1DeviceAddr,
                    size * sizeof(self1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out1 result[%ld] is: %f\n", i, self1Data[i]);
  }

  size = GetShapeSize(selfShape2);
  std::vector<float> self2Data(size, 0);
  ret = aclrtMemcpy(self2Data.data(), self2Data.size() * sizeof(self2Data[0]), input2DeviceAddr,
                    size * sizeof(self2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out2 result[%ld] is: %f\n", i, self2Data[i]);
  }

  // 6. Release aclTensor. Modify the code according to the specific API definition.
  aclDestroyTensorList(tensorListInput1);
  aclDestroyTensorList(tensorListInput2);
  aclDestroyTensor(alpha);

  // 7. Release device resources. Modify the code according to the specific API definition.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(other1DeviceAddr);
  aclrtFree(other2DeviceAddr);
  aclrtFree(alphaDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
