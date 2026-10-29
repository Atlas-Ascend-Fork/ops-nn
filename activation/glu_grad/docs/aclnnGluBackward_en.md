# aclnnGluBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T10:44:15.625Z -->

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

- Description: Implements the backward propagation of [aclnnGlu](../../glu/docs/aclnnGlu.md).
- Formula:

  $$
  \frac{\partial GLU(a,b)}{\partial(a,b)}=cat(\sigma(b),\sigma(b) \otimes a \otimes (1-\sigma(b)))
  $$

- Mathematical expression:

  Assume that the output GLUGrad consists of two parts: out=[a_grad, b_grad]. Then:
  sig_b = sigmoid(b)
  **a_grad**=grad_out*sig_b
  **b_grad**=a_grad*(a-a*sig_b)
  Here: grad_out is gradOut, a is the first half of the input tensor after it is split equally along the specified dim, and b is the second half.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGluBackwardGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the computation flow, and then call the "aclnnGluBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnGluBackwardGetWorkspaceSize(
  const aclTensor *gradOut,
  const aclTensor *self,
  int64_t          dim,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGluBackward(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnGluBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1410px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 260px">
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
      <td>gradOut (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient update coefficient, which is grad_out in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type must be the same as the data type of self, and the shape is $(*_1,M,*_2)$, where $*$ indicates the corresponding dimensions of self and $M = N /2$.</li></ul></td>
      <td>DOUBLE, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input parameter for GluBackward computation.</td>
      <td>The tensor dimension must be greater than 0, and the shape must be divisible by 2 along the dimension specified by input parameter dim. The shape is represented as $(*_1,N,*_2)$, where $*$ denotes any number of additional dimensions and $N$ denotes the size of the dimension specified by dim.</td>
      <td>DOUBLE, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>Indicates the dimension along which the input self is split.</td>
      <td>Value range: [-self.dim, self.dim-1].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output parameter.</td>
      <td>The data type and shape must be the same as those of self.</td>
      <td>DOUBLE, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
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
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The supported data types are DOUBLE, FLOAT, and FLOAT16.
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
      <td>gradOut, self, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of gradOut, self, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The input dim exceeds the available dimension range [-self.dim, self.dim-1] of self.</td>
    </tr>
    <tr>
      <td>The dimension of self specified by the input dim is not divisible by 2.</td>
    </tr>
      <tr>
      <td>The shape of out is not equal to the shape of self.</td>
    </tr>
      <tr>
      <td>The data types of gradOut and out are inconsistent with that of self.</td>
    </tr>
      <tr>
      <td>The shape of gradOut does not meet (*1,M,*2), where M = N /2 and N is the value of self on the dimension specified by dim.</td>
    </tr>
      <tr>
      <td>The dimensions of gradOut, self, or out are greater than 8.</td>
    </tr>
      <tr>
      <td>The dimension of self is 0.</td>
    </tr>
  </tbody></table>

## aclnnGluBackward

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the device, obtained by the first-phase API aclnnGluBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation flow.</td>
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

## Constraints

- Deterministic computation:
  - aclnnGluBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_glu_backward.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization
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

  // Call aclrtMemcpy to copy host data to the device memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs. Construct them as required based on the API.
  std::vector<int64_t> gradOutShape = {2,4,3};
  std::vector<int64_t> selfShape = {2,4,6};
  std::vector<int64_t> outShape = {2,4,6};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradOutHostData = {
    1,  1,  1,
    1,  1,  1,
    1,  1,  1,
    1,  1,  1,
    1,  1,  1,
    1,  1,  1,
    1,  1,  1,
    1,  1,  1
  };

  std::vector<float> selfHostData = {
    0.2948,  1.6331,  2.3158, -0.6872,  0.3036,  0.1575,
    0.2992,  1.0893, -0.1126,  0.1910, -1.3675,  0.5587,
    0.4928,  1.4385,  0.6834, -0.6529,  1.0361, -0.6160,
    1.2554, -2.0038,  0.5361, -1.4009, -0.7497, -0.8814,
    0.4113,  0.7549, -1.2869, -1.4354,  0.6939,  0.2192,
    0.3932,  1.8506, -0.7737,  3.6379, -0.9404, -1.1261,
    -1.6927,  0.8456,  0.6500,  0.2738,  0.5115,  0.3356,
    0.5763,  0.2667, -0.6570, -0.4159,  1.5258,  0.0843
  };
  std::vector<float> outHostData = {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
  };

  // Create the gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t dim = -1;

  // 3. Call the CANN operator library API. Replace it with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGluBackward.
  ret = aclnnGluBackwardGetWorkspaceSize(gradOut, self, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGluBackward.
  ret = aclnnGluBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGluBackward failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value, and copy the result on device memory to the host. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it according to the specific API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(gradOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
