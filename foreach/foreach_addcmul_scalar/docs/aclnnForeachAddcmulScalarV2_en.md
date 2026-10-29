# aclnnForeachAddcmulScalarV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T21:44:57.439Z -->

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

- Description: Performs element-wise multiplication on tensor list x2 and tensor list x3, multiplies the product by the scalar, and then performs element-wise addition between the obtained result and tensor list x1. Compared with [aclnnForeachAddcmulScalar](aclnnForeachAddcmulScalar.md), this API changes the structure type of the input parameter scalar from aclTensor to aclScalar. Choose an appropriate API based on actual requirements.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_{i} + {\rm scalar} \times x2_{i} \times x3_{i} (i=0,1,...n-1)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnForeachAddcmulScalarV2GetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the computation process, and then call the "aclnnForeachAddcmulScalarV2" API to perform the computation.

```Cpp
aclnnStatus aclnnForeachAddcmulScalarV2GetWorkspaceSize(
  const aclTensorList *x1,
  const aclTensorList *x2,
  const aclTensorList *x3,
  const aclScalar     *scalar,
  aclTensorList       *out,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachAddcmulScalarV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachAddcmulScalarV2GetWorkspaceSize

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
      <td>x1 (aclTensorList*)</td>
      <td>Input</td>
      <td>First input tensor list of the addition in the mixed operation, corresponding to `x1` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data types of all tensors in this parameter must be consistent.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x2 (aclTensorList*)</td>
      <td>Input</td>
      <td>Second input tensor list of the multiplication in the mixed operation, corresponding to `x2` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>All tensors in this parameter must have consistent data types.</li><li>The data type, data format, and shape must be consistent with input parameter `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x3 (aclTensorList*)</td>
      <td>Input</td>
      <td>The third input tensor list of the multiplication in the mixed computation. Corresponds to `x3` in the formula.</td>
     <td><ul><li>Does not support empty tensors.</li><li>All tensors in this parameter must have consistent data types.</li><li>The data type, data format, and shape must be consistent with input parameter `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scalar (aclScalar*)</td>
      <td>Input</td>
      <td>Specifies the first input scalar of the multiplication in the blended operation. It corresponds to `scalar` in the corresponding formula.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, INT32, DOUBLE, INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensorList*)</td>
      <td>Output</td>
      <td>Output tensor list of the mixed operation. It is `y` in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensor.</li><li>All tensors in this parameter have consistent data types.</li><li>The data type and data format are consistent with those of input parameter `x1`, and the shape size is greater than or equal to the shape size of input parameter `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    The data type of `scalar` has a corresponding relationship with input parameter `x1`:
    - When the data type of `x1` is FLOAT32 or BFLOAT16, the supported data types are FLOAT32 and DOUBLE.
    - When the data type of `x1` is FLOAT16, the supported data types are FLOAT16 and DOUBLE.
    - When the data type of `x1` is INT32, the supported data types are INT32 and INT64.
  <!-- end id7 -->

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The data type of `scalar` has a specific correspondence with the data type of input parameter `x1`:
      - When the data type of `x1` is BFLOAT16 or FLOAT32, the supported data types are FLOAT32 and DOUBLE.
      - When the data type of `x1` is FLOAT16, the supported data types are FLOAT16, FLOAT32, and DOUBLE.
      - When the data type of `x1` is INT32, the supported data types are INT32 and INT64.
    - The maximum number of tensors supported by the input `x1`, `x2`, `x3`, and the output `y` is 50 each.
  <!-- end id8 -->

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
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed x1, x2, x3, scalar, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of x1, x2, x3, scalar, and out are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of x1, x2, x3, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of x1, x2, x3, and out do not meet the constraints.</td>
    </tr>
    <tr>
      <td>The element data types of the tensors in x1, x2, x3, or out are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimensions of the tensors in x1, x2, x3, or out exceed 8.</td>
    </tr>
  </tbody></table>

## aclnnForeachAddcmulScalarV2

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnForeachAddcmulScalarV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, including the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnForeachAddcmulScalarV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_addcmul_scalar_v2.h"

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

int Init(int32_t deviceId, aclrtStream *stream)
{
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of contiguous tensors.
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

  // 2. Construct inputs and outputs. Customize the construction as required by the API.
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> otherShape1 = {2, 3};
  std::vector<int64_t> otherShape2 = {1, 3};
  std::vector<int64_t> anotherShape1 = {2, 3};
  std::vector<int64_t> anotherShape2 = {1, 3};
  std::vector<int64_t> outShape1 = {2, 3};
  std::vector<int64_t> outShape2 = {1, 3};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* other1DeviceAddr = nullptr;
  void* other2DeviceAddr = nullptr;
  void* another1DeviceAddr = nullptr;
  void* another2DeviceAddr = nullptr;
  void* out1DeviceAddr = nullptr;
  void* out2DeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* other1 = nullptr;
  aclTensor* other2 = nullptr;
  aclTensor* another1 = nullptr;
  aclTensor* another2 = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out1 = nullptr;
  aclTensor* out2 = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9};
  std::vector<float> other1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> other2HostData = {7, 8, 9};
  std::vector<float> another1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> another2HostData = {7, 8, 9};
  std::vector<float> out1HostData(6, 0);
  std::vector<float> out2HostData(3, 0);
  float alphaValue = 1.2f;
  // Create input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create other1 aclTensor.
  ret = CreateAclTensor(other1HostData, otherShape1, &other1DeviceAddr, aclDataType::ACL_FLOAT, &other1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create other2 aclTensor.
  ret = CreateAclTensor(other2HostData, otherShape2, &other2DeviceAddr, aclDataType::ACL_FLOAT, &other2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create another1 aclTensor.
  ret = CreateAclTensor(another1HostData, anotherShape1, &another1DeviceAddr, aclDataType::ACL_FLOAT, &another1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create another2 aclTensor.
  ret = CreateAclTensor(another2HostData, anotherShape2, &another2DeviceAddr, aclDataType::ACL_FLOAT, &another2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return -1);
  // Create out1 aclTensor.
  ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out2 aclTensor.
  ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tempInput1{input1, input2};
  aclTensorList* tensorListInput1 = aclCreateTensorList(tempInput1.data(), tempInput1.size());
  std::vector<aclTensor*> tempInput2{other1, other2};
  aclTensorList* tensorListInput2 = aclCreateTensorList(tempInput2.data(), tempInput2.size());
  std::vector<aclTensor*> tempAnother{another1, another2};
  aclTensorList* tensorListAnother = aclCreateTensorList(tempAnother.data(), tempAnother.size());
  std::vector<aclTensor*> tempOutput{out1, out2};
  aclTensorList* tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnForeachAddcmulScalarV2.
  ret = aclnnForeachAddcmulScalarV2GetWorkspaceSize(tensorListInput1, tensorListInput2, tensorListAnother, alpha, tensorListOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachAddcmulScalarV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnForeachAddcmulScalarV2.
  ret = aclnnForeachAddcmulScalarV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachAddcmulScalarV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from device memory to host memory, and modify it based on the specific API definition.
  auto size = GetShapeSize(outShape1);
  std::vector<float> out1Data(size, 0);
  ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), out1DeviceAddr,
                    size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
  }

  size = GetShapeSize(outShape2);
  std::vector<float> out2Data(size, 0);
  ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), out2DeviceAddr,
                    size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
  }

  // 6. Release aclTensor and aclScalar, and modify it based on the specific API definition.
  aclDestroyTensorList(tensorListInput1);
  aclDestroyTensorList(tensorListInput2);
  aclDestroyTensorList(tensorListAnother);
  aclDestroyTensorList(tensorListOutput);
  aclDestroyScalar(alpha);

  // 7. Release device resources, and modify it based on the specific API definition.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(other1DeviceAddr);
  aclrtFree(other2DeviceAddr);
  aclrtFree(another1DeviceAddr);
  aclrtFree(another2DeviceAddr);
  aclrtFree(out1DeviceAddr);
  aclrtFree(out2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
