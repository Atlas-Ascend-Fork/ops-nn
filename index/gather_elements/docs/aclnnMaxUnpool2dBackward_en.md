# aclnnMaxUnpool2dBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:15:12.615Z -->

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

- Description: backward propagation of the MaxPool2d inverse operation [aclnnMaxUnpool2d](../../scatter_elements/docs/aclnnMaxUnpool2d.md). Fills elements of gradOutput into out based on the indices.

- Formula:

  $$
  out[n][c][i] = gradOutput[n][c][indices[n][c][i]]
  $$

  where out, gradOutput, and indices are obtained by reshaping the last two axes into one axis, H and W represent the last two axes respectively, and $i ∈ [0, H*W)$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMaxUnpool2dBackwardGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnMaxUnpool2dBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnMaxUnpool2dBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     self,
  const aclTensor*     indices,
  const aclIntArray*   outputSize,
  aclTensor*           out,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnMaxUnpool2dBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnMaxUnpool2dBackwardGetWorkspaceSize

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
      <td class="tg-0pky">gradOutput (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">gradOutput in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type must be convertible to the data type of out (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">conversion relationship</a>).</li>
          <li>The data type must be consistent with that of self.</li>
          <li>The shape must be (N, C, outputSize[0], outputSize[1]) or (C, outputSize[0], outputSize[1]).</li>
          <li>The number of dimensions must be consistent with that of self.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">        <ul>
          <li>The data type must be consistent with gradOutput.</li>
          <li>The shape must be (N, C, H, W) or (C, H, W), and must be consistent with indices.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">indices in the formula.</td>
      <td class="tg-0pky">The shape must be consistent with self.</td>
      <td class="tg-0pky">INT64, INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">outputSize(aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">The number of elements must be 2, and the element values must be consistent with the last two dimensions of the shape of gradOutput.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">out in the formula.</td>
      <td class="tg-0pky">The shape must be consistent with self.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be requested on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor containing the operator computation flow.</td>
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
      <td>The input gradOutput, self, indices, outputSize, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>gradOutput, self, indices, or out is not within the supported data type range.</td>
    </tr>
    <tr>
      <td>gradOutput cannot be converted to the data type of out.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput and self are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of self is not 3 or 4.</td>
    </tr>
    <tr>
      <td>The dimension of gradOutput is inconsistent with that of self.</td>
    </tr>
    <tr>
      <td>When self is 3-dimensional, the sizes of self and gradOutput in the C dimension are inconsistent.</td>
    </tr>
    <tr>
      <td>If self is 4-dimensional, the sizes of self and gradOutput are inconsistent on the N and C dimensions.</td>
    </tr>
    <tr>
      <td>The shapes of self, indices, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of elements in outputSize is not 2.</td>
    </tr>
    <tr>
      <td>The sizes of the last two dimensions of gradOutput are inconsistent with the elements of outputSize.</td>
    </tr>
    <tr>
      <td>self and indices are not empty tensors, but gradOutput is an empty tensor.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxUnpool2dBackward

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the device, obtained by the first-phase API aclnnLogAddExpGetWorkspaceSize.</td>
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
  - aclnnMaxUnpool2dBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool2d_backward.h"

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
  // Boilerplate. Resource initialization.
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
  // Call aclrtMemcpy to copy data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate strides of the contiguous tensor.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
  int64_t N = 1;
  int64_t C = 3;
  int64_t H = 2;
  int64_t W = 2;
  std::vector<int64_t> outputSizeData = {3, 1};
  std::vector<int64_t> gradOutputShape = {N, C, outputSizeData[0], outputSizeData[1]};
  std::vector<int64_t> selfShape = {N, C, H, W};
  std::vector<int64_t> indicesShape = {N, C, H, W};
  std::vector<int64_t> outShape = {N, C, H, W};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclIntArray* outputSize = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> selfHostData(12, 1);
  std::vector<int32_t> indicesHostData = {0, 1, 2, 1, 2, 0, 1, 1, 1, 0, 0, 0};
  std::vector<float> outHostData(12, 0);
  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the outputSize aclIntArray.
  outputSize = aclCreateIntArray(outputSizeData.data(), 2);
  CHECK_RET(outputSize != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnMaxUnpool2dBackward first-phase API.
  ret = aclnnMaxUnpool2dBackwardGetWorkspaceSize(gradOutput, self, indices, outputSize, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnMaxUnpool2dBackward second-phase API.
  ret = aclnnMaxUnpool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Change it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Change it according to the specific API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyTensor(out);

  // 7. Release the device resources. Change it according to the specific API definition.
  aclrtFree(gradOutputDeviceAddr);
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
