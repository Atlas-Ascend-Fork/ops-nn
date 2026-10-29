# aclnnMaxUnpool3d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:45:43.421Z -->

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

- Description: the inverse operation of [aclnnMaxPool](../../../pooling/max_pool_v3/docs/aclnnMaxPool.md) in 3D. The D, H, and W axis sizes of outRef are determined by outputSize. The elements of self are filled into outRef based on the indices, and other positions are set to 0.

- Formula:

  - When the input is 4-dimensional, with dimensions N, D, H, and W (where N (Batch) represents the batch size, D (Depth) represents the feature map depth, H (Height) represents the feature map height, and W (Width) represents the feature map width):

    $$
    outRef[N][indices[N][i]] = self[N][i]
    $$

  - When the input is 5-dimensional, with dimensions N, C, D, H, and W (where C (Channels) represents the feature map channels):

    $$
    outRef[N][C][indices[N][C][i]] = self[N][C][i]
    $$

    Here, outRef, indices, and self are obtained by merging the last two axes into one axis through reshape, where i ∈ [0, D*H*W).

## Function Prototype

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMaxUnpool3dGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnMaxUnpool3d" API to perform the computation.

```Cpp
aclnnStatus aclnnMaxUnpool3dGetWorkspaceSize(
  const aclTensor*     self,
  const aclTensor*     indices,
  const aclIntArray*   outputSize,
  const aclIntArray*   stride,
  const aclIntArray*   padding,
  aclTensor*           outRef,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnMaxUnpool3d(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnMaxUnpool3dGetWorkspaceSize

- **Parameters:**

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
      <th class="tg-0pky">Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">self in the formula, indicating the target tensor to be converted.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type is the same as that of outRef.</li>
          <li>The shape is the same as that of indices.</li>
          <li>When the dimension is 4, each dimension indicates N, D, H, W in turn. When the dimension is 5, each dimension indicates N, C, D, H, W in turn.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">indices in the formula, indicating the output positions of the elements in input self.</td>
      <td class="tg-0pky">        <ul>
          <li>The shape is consistent with self.</li>
          <li>When the dimension is 4, each dimension indicates N, D, H, W in turn. When the dimension is 5, each dimension indicates N, C, D, H, W in turn.</li>
        </ul>
      </td>
      <td class="tg-0pky">INT64, INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">outputSize (aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Represents the spatial size of the output on the D, H, and W dimensions.</td>
      <td class="tg-0pky">The size is 3. The product of the three elements must be greater than or equal to the product of the sizes of self on the D, H, and W dimensions.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">stride (aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the stride size of the max pooling window on the D, H, and W dimensions.</td>
      <td class="tg-0pky">Reserved parameter. It is not involved in computation in the current version. An aclIntArray of size 3, with values greater than 0, applied on the host side, must be passed in.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">padding (aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Represents the padding values of the max pooling window on the D, H, and W dimensions.</td>
      <td class="tg-0pky">Reserved parameter, not involved in computation in the current version. A Host-side aclIntArray whose size is 3 must be passed in.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">outRef (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">outRef in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type is the same as that of self.</li>
          <li>When the dimension is 4, each dimension represents N, D, H, and W in sequence. When the dimension is 5, each dimension represents N, C, D, H, and W in sequence.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size that needs to be applied on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor containing the operator calculation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Return value**

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
      <td>self, indices, outputSize, stride, padding, or outRef is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The data types of self and indices are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and outRef are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of self is not 4D or 5D.</td>
    </tr>
    <tr>
      <td>The shapes of self and indices are inconsistent.</td>
    </tr>
    <tr>
      <td>The size of outputSize is not 3.</td>
    </tr>
    <tr>
      <td>The product of the three elements of outputSize is less than the product of the sizes of self on the D, H, and W dimensions.</td>
    </tr>
    <tr>
      <td>The size of stride is not 3.</td>
    </tr>
    <tr>
      <td>The size of padding is not 3.</td>
    </tr>
    <tr>
      <td>The size of self on the C, D, H, or W dimension is not greater than 0.</td>
    </tr>
    <tr>
      <td>The value of an element in stride is not greater than 0.</td>
    </tr>
    <tr>
      <td>The size of outRef on the N and C dimensions is not exactly the same as that of self.</td>
    </tr>
    <tr>
      <td>The size of outRef on the D, H, and W dimensions is not equal to the three element values in outputSize.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxUnpool3d

- **Parameters:**

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnMaxUnpool3dGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnMaxUnpool3d defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool3d.h"

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
  // Call aclrtMalloc to apply for memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction as required by the API.
  std::vector<int64_t> selfShape = {1, 1, 2, 2};
  std::vector<int64_t> outShape = {1, 1, 4, 4};
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> outHostData = {0, 0, 0, 0.0, 0, 0, 0, 0,
                                    0, 0, 0, 0.0, 0, 0, 0, 0};
  std::vector<int64_t> indicesHostData = {3, 8, 11, 13};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, selfShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
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
  // Call the aclnnMaxUnpool3d first-phase API.
  ret = aclnnMaxUnpool3dGetWorkspaceSize(self, indices, outputSize, stride, padding, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory using the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnMaxUnpool3d second-phase API.
  ret = aclnnMaxUnpool3d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3d failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. Release device resources.
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
