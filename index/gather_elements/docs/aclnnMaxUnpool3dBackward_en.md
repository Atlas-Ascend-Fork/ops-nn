# aclnnMaxUnpool3dBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:16:41.123Z -->

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

- Description: backward propagation of [aclnnMaxUnpool3d](../../scatter_elements/docs/aclnnMaxUnpool3d.md), the inverse operation of MaxPool3d. Elements of gradOutput are filled into out based on indices.
- Formula:
  - When the input is 4-dimensional, the dimensions represent (N, D, H, W):
  $$
  out[N][i] = gradOutput[N][indices[N][i]]
  $$

  - When the input is 5-dimensional, the dimensions represent (N, C, D, H, W):
  $$
  out[N][C][i] = gradOutput[N][C][indices[N][C][i]]
  $$
  where out, gradOutput, and indices are obtained by merging the last two axes into one axis through reshape, and i ∈ [0, D*H*W).

## Function Prototype

  Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMaxUnpool3dBackwardGetWorkspaceSize" API to obtain the workspace size required for calculation and the executor that includes the calculation process of the operator, and then call the "aclnnMaxUnpool3dBackward" API to execute the calculation.

```Cpp
aclnnStatus aclnnMaxUnpool3dBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     self,
  const aclTensor*     indices,
  const aclIntArray*   outputSize,
  const aclIntArray*   stride,
  const aclIntArray*   padding,
  aclTensor*           out,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnMaxUnpool3dBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnMaxUnpool3dBackwardGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
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
      <td class="tg-0pky">gradOutput (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">gradOutput in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type needs to be consistent with self and out.</li>
          <li>The meanings of each dimension are respectively (N, outputSize[0], outputSize[1], outputSize[2]) or (N, C, outputSize[0], outputSize[1], outputSize[2]).</li>
          <li>When the dimension is 4, dimensions represent N, D, H, and W in sequence. When the dimension is 5, dimensions represent N, C, D, H, and W in sequence.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND, NCDHW</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">        <ul>
          <li>The data type needs to be consistent with gradOutput and out.</li>
          <li>The meanings of each dimension are respectively (N, D, H, W) or (N, C, D, H, W).</li>
          <li>The dimension needs to be consistent with gradOutput, and the shape needs to be consistent with indices and out.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND, NCDHW</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">indices in the formula, indicating the index positions of the elements in input gradOutput in the output result.</td>
      <td class="tg-0pky">The shape needs to be consistent with self.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">ND, NCDHW</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">outputSize (aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the spatial size of the output result on the D, H, and W dimensions.</td>
      <td class="tg-0pky">The size is 3.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">stride (aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the stride size of the max pooling window on the D, H, and W dimensions.</td>
      <td class="tg-0pky">The size is 3.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">padding (aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Represents the padding values of the max pooling window on the D, H, and W dimensions.</td>
      <td class="tg-0pky">size is 3.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">out in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type needs to be consistent with gradOutput and self.</li>
          <li>The shape needs to be consistent with self and indices.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND, NCDHW</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">×</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be applied on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor that includes the calculation process of the operator.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
  <col style="width: 286px">
  <col style="width: 123px">
  <col style="width: 738px">
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
      <td>gradOutput, self, indices, outputSize, stride, padding, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="14">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="14">161002</td>
      <td>out is a non-contiguous tensor.</td>
    </tr>
    <tr>
      <td>The data type of gradOutput, self, indices, or out is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, self, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of self is not 4 or 5.</td>
    </tr>
    <tr>
      <td>The dimensions of self, indices, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of self, indices, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The size of self on each dimension except N is not greater than 0.</td>
    </tr>
    <tr>
      <td>The size of outputSize, stride, or padding is not equal to 3.</td>
    </tr>
    <tr>
      <td>The element value of outputSize or stride is not greater than 0.</td>
    </tr>
    <tr>
      <td>The product of the three elements of outputSize is smaller than the product of the sizes of self on the D, H, and W dimensions.</td>
    </tr>
    <tr>
      <td>The size of gradOutput on the D, H, and W dimensions is not equal to the three element values in outputSize.</td>
    </tr>
    <tr>
      <td>gradOutput and self have different dimensions.</td>
    </tr>
    <tr>
      <td>When self is 4D, the size of gradOutput on the N dimension is not consistent with that of self.</td>
    </tr>
    <tr>
      <td>When self is 5D, the size of gradOutput on the C or N dimension is not consistent with that of self.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxUnpool3dBackward

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>Memory address of the workspace applied on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the device side, obtained by the first-phase API aclnnMaxUnpool3dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that includes the calculation flow of the operator.</td>
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

> **Warning**: Each element of **indices** must satisfy `0 <= indices[i] < outputSize[0] * outputSize[1] * outputSize[2]`. The upper bound is the product of the last three dimensions of **gradOutput**. This interface does not check the data value range of **indices**. The caller must ensure that all indices are valid; out-of-range values (including negative values) result in undefined behavior.

- Deterministic computation:
  - aclnnMaxUnpool3dBackward defaults to a deterministic implementation.

## Example

The sample code below is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool3d_backward.h"

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
  // Call aclrtMemcpy to copy data from the host side to the device side memory.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {1, 1, 4, 4};
  std::vector<int64_t> indicesShape = {1, 1, 4, 4};
  std::vector<int64_t> gradShape = {1, 1, 4, 4};
  std::vector<int64_t> outShape = {1, 1, 4, 4};
  void* gradDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> gradHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData = {0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0};
  std::vector<int64_t> indicesHostData = {0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 13};
  // Create the grad aclTensor.
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize1 = {1, 4, 4};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize1.data(), arraySize1.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  std::vector<int64_t> arraySize2 = {1, 2, 3};
  const aclIntArray *stride = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  const aclIntArray *padding = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMaxUnpool3dBackward.
  ret = aclnnMaxUnpool3dBackwardGetWorkspaceSize(grad, self, indices, outputSize, stride, padding, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory using the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMaxUnpool3dBackward.
  ret = aclnnMaxUnpool3dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from the device side to the host side. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr, size * sizeof(outData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(grad);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. Release device resources.
  aclrtFree(gradDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
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
