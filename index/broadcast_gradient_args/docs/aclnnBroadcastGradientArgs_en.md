# aclnnBroadcastGradientArgs

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:10:11.981Z -->

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

- Description: BroadcastGradientArgs is a TensorFlow operator used to compute the broadcast dimension indices required for gradient propagation. During backward propagation, based on the original shapes of the two tensors in forward propagation, it automatically identifies the dimensions extended by the broadcasting mechanism and outputs the dimensions along which gradients need to be reduced, so that the gradients can be restored from the broadcast shape to the shape of each original tensor.
- Computation rules:
  1. If the two inputs have the same shape length and all dimension values are equal one by one (that is, the two inputs are fully elementwise consistent), both y1 and y2 are empty, indicating that no reduction is required.
  2. If the two inputs have different shape lengths, pad the shorter one with 1s on the left until its length is consistent with the longer one. After padding, the extra leading dimensions of the shorter one are regarded as dimensions with value 1 for that tensor and must be added to the corresponding reduction list.
  3. Except for use case 1, compare x1[i] with x2[i] dimension by dimension after alignment. If the value of either side is 1, the dimension is added to the reduction list corresponding to the side whose value is 1 (when x1[i]==1, index i is added to y1; when x2[i]==1, index i is added to y2).
- Example:

  ```text
  Example 1 (the two inputs have the same length, and some dimensions have value 1):
    The shape of tensor a is [2, 1, 4, 1, 6]
    The shape of tensor b is [2, 3, 1, 5, 1]
    x1_data: [2, 1, 4, 1, 6]   # x1_shape=[5]
    x2_data: [2, 3, 1, 5, 1]   # x2_shape=[5]

    Dimension-by-dimension comparison:
      dim0: x1=2, x2=2   → Equal, no output
      dim1: x1=1, x2=3   → x1[i]==1, y1 collects index 1
      dim2: x1=4, x2=1   → x2[i]==1, y2 collects index 2
      dim3: x1=1, x2=5   → x1[i]==1, y1 collects index 3
      dim4: x1=6, x2=1   → x2[i]==1, y2 collects index 4

    y1_data: [1, 3]            # y1_shape=[2]
    y2_data: [2, 4]            # y2_shape=[2]

  Example 2 (the two inputs have different lengths; left-pad with 1 to align):
    The shape of the original tensor a is [4, 1, 6]
    The shape of the original tensor b is [2, 3, 1, 5, 1]
    x1_data: [4, 1, 6]         # x1_shape=[3], shorter
    x2_data: [2, 3, 1, 5, 1]   # x2_shape=[5], longer

    Alignment: left-pad x1 with 1 to length 5, obtaining [1, 1, 4, 1, 6]
          x1(after padding): [1, 1, 4, 1, 6]
          x2:         [2, 3, 1, 5, 1]

    Dimension-by-dimension comparison (using the padded x1):
      dim0: x1=1(padded), x2=2 → x1[i]==1, y1 collects index 0
      dim1: x1=1(padded), x2=3 → x1[i]==1, y1 collects index 1
      dim2: x1=4,     x2=1 → x2[i]==1, y2 collects index 2
      dim3: x1=1,     x2=5 → x1[i]==1, y1 collects index 3
      dim4: x1=6,     x2=1 → x2[i]==1, y2 collects index 4

    y1_data: [0, 1, 3]         # y1_shape=[3]
    y2_data: [2, 4]            # y2_shape=[2]

  Example 3 (the two inputs are completely consistent, no reduction required):
    The shape of the original tensor a is [2, 1, 4, 1, 6]
    The shape of the original tensor b is [2, 1, 4, 1, 6]
    x1_data: [2, 1, 4, 1, 6]   # x1_shape=[5]
    x2_data: [2, 1, 4, 1, 6]   # x2_shape=[5]

    The two inputs have the same length and all dimension values are equal one by one. This belongs to use case 1, so no reduction is required.

    y1_data: []                # y1_shape=[0]
    y2_data: []                # y2_shape=[0]
  ```

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnBroadcastGradientArgsGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnBroadcastGradientArgs" API to execute the computation.

```Cpp
aclnnStatus aclnnBroadcastGradientArgsGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  aclTensor       *y1,
  aclTensor       *y2,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBroadcastGradientArgs(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBroadcastGradientArgsGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 150px">
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
      <td>x1</td>
      <td>Input</td>
      <td>Shape of the original tensor a.</td>
      <td>Must be 1-dimensional.<br>The data type must be consistent with x2.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Shape of the original tensor b.</td>
      <td>Must be 1D.<br>Data type must be consistent with x1.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Indicates the indices to be broadcast in the tensor shape corresponding to x1.</td>
      <td>Data type must be consistent with x1.<br>The output size must be allocated based on max(x1_len, x2_len). The actual output may be smaller than this value. After execution is complete, obtain the actual shape through aclGetViewShape.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Indicates the indices to be broadcast in the tensor shape corresponding to x2.</td>
      <td>The data type must be consistent with x1.<br>The output size must be allocated based on max(x1_len, x2_len). The actual output may be smaller than this value. After execution is complete, obtain the actual shape through aclGetViewShape.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be requested on the Device.</td>
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
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1071px"><colgroup>
  <col style="width: 265px">
  <col style="width: 149px">
  <col style="width: 657px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x1, x2, y1, or y2 is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of x1 or x2 is not in the supported range (only INT32 and INT64 are supported).</td>
    </tr>
    <tr>
      <td>The data types of x1 and x2 are inconsistent, or the data type of y1/y2 is inconsistent with that of x1/x2.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, y1, or y2 is not 1-dimensional.</td>
    </tr>
    <tr>
      <td>The capacity of y1 or y2 is less than max(x1 length, x2 length), which may cause an out-of-bounds kernel write.</td>
    </tr>
    <tr>
      <td>The current device is not Ascend 950PR&950DT products. The operator is not supported on this architecture.</td>
    </tr>
  </tbody>
  </table>

## aclnnBroadcastGradientArgs

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1042px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 791px">
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
      <td>Pointer to the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the device, obtained by the first-phase API aclnnBroadcastGradientArgsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBroadcastGradientArgs defaults to a deterministic implementation.

- Supported only on <term>Ascend 950PR&950DT products</term>; not supported on other architectures.
- Inputs x1 and x2 must satisfy the broadcast rule: the corresponding dimensions must be equal, or at least one of them must be 1.
- y1 and y2 are dynamic shape outputs. The output memory must be pre-allocated based on max(x1 length, x2 length). The number of actual output elements may be less than this value. After execution is complete, call aclGetViewShape to obtain the actual output shape.
- Non-contiguous tensors are not supported for y1 and y2.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_broadcast_gradient_args.h"

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
  // Boilerplate: resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            aclrtResetDevice(deviceId); aclFinalize(); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream, see the acl API documentation
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output.
  //    x1 = {2, 1, 4, 1, 6}  (shape of the original tensor a)
  //    x2 = {2, 3, 1, 5, 1}  (shape of the original tensor b)
  //    Output is allocated with the maximum possible size, max(5, 5) = 5.
  std::vector<int32_t> x1HostData = {2, 1, 4, 1, 6};
  std::vector<int32_t> x2HostData = {2, 3, 1, 5, 1};
  std::vector<int64_t> y1Shape = {5};
  std::vector<int64_t> y2Shape = {5};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* y1DeviceAddr = nullptr;
  void* y2DeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* y1 = nullptr;
  aclTensor* y2 = nullptr;
  std::vector<int32_t> y1HostData(5, 0);
  std::vector<int32_t> y2HostData(5, 0);
  ret = CreateAclTensor(x1HostData, {5}, &x1DeviceAddr, aclDataType::ACL_INT32, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(x2HostData, {5}, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y1HostData, y1Shape, &y1DeviceAddr, aclDataType::ACL_INT32, &y1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y2HostData, y2Shape, &y2DeviceAddr, aclDataType::ACL_INT32, &y2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnBroadcastGradientArgs.
  ret = aclnnBroadcastGradientArgsGetWorkspaceSize(x1, x2, y1, y2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBroadcastGradientArgsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBroadcastGradientArgs.
  ret = aclnnBroadcastGradientArgs(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBroadcastGradientArgs failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the actual output shape (a key step for dynamic shape) and copy the result from the device side to the host side.
  int64_t* y1ViewDims = nullptr;
  uint64_t y1ViewDimsNum = 0;
  ret = aclGetViewShape(y1, &y1ViewDims, &y1ViewDimsNum);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclGetViewShape y1 failed. ERROR: %d\n", ret); return ret);
  int64_t y1Size = 1;
  for (uint64_t i = 0; i < y1ViewDimsNum; i++) {
    y1Size *= y1ViewDims[i];
  }
  if (y1Size > 0) {
    std::vector<int32_t> y1ResultData(y1Size, 0);
    ret = aclrtMemcpy(y1ResultData.data(), y1ResultData.size() * sizeof(int32_t), y1DeviceAddr,
                      y1Size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1 result failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < y1Size; i++) {
      LOG_PRINT("y1[%ld] is: %d\n", i, y1ResultData[i]);
    }
  } else {
    LOG_PRINT("y1 is empty (no broadcast axis)\n");
  }
  delete[] y1ViewDims;

  int64_t* y2ViewDims = nullptr;
  uint64_t y2ViewDimsNum = 0;
  ret = aclGetViewShape(y2, &y2ViewDims, &y2ViewDimsNum);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclGetViewShape y2 failed. ERROR: %d\n", ret); return ret);
  int64_t y2Size = 1;
  for (uint64_t i = 0; i < y2ViewDimsNum; i++) {
    y2Size *= y2ViewDims[i];
  }
  if (y2Size > 0) {
    std::vector<int32_t> y2ResultData(y2Size, 0);
    ret = aclrtMemcpy(y2ResultData.data(), y2ResultData.size() * sizeof(int32_t), y2DeviceAddr,
                      y2Size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2 result failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < y2Size; i++) {
      LOG_PRINT("y2[%ld] is: %d\n", i, y2ResultData[i]);
    }
  } else {
    LOG_PRINT("y2 is empty (no broadcast axis)\n");
  }
  delete[] y2ViewDims;

  // 6. Release the aclTensor.
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(y1);
  aclDestroyTensor(y2);

  // 7. Release device resources.
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(y1DeviceAddr);
  aclrtFree(y2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
