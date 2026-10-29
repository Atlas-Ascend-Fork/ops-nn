# aclnnGroupNormSilu

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:06:43.912Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Computes the Group Normalization result groupnormOut, mean meanOut, and reciprocal of standard deviation rstdOut of the input self, and applies the SiLU operation to groupnormOut to obtain the final output out.
- Formula:
  - **GroupNorm:**

    Let $E[x] = \bar{x}$ denote the mean of $x$, $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ denote the variance of $x$, then

    $$
    \left\{
    \begin{array} {rcl}
    groupnormOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Silu:**

    $$
    out = \frac{groupnormOut}{1+e^{-groupnormOut}}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGroupNormSiluGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the computation process, and then call "aclnnGroupNormSilu" to perform the computation.

```c++
aclnnStatus aclnnGroupNormSiluGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* gamma,
    const aclTensor* beta,
    int64_t          group,
    double           eps,
    aclTensor*       out,
    aclTensor*       meanOut,
    aclTensor*       rstdOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);
```

```c++
aclnnStatus aclnnGroupNormSilu(
    void *         workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupNormSiluGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 187px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
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
        <th>Non-contiguous tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>x in the formula.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>2-8, where the first dimension is N and the second dimension is C.</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gamma (aclTensor*)</td>
        <td>Input</td>
        <td>γ in the formula.</td>
        <td>The data type must be the same as that of self or FLOAT, and the number of elements must be the same as the size of the second dimension of input self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>beta (aclTensor*)</td>
        <td>Input</td>
        <td>β in the formula.</td>
        <td>The data type must be the same as that of self or FLOAT, and the number of elements must be the same as the size of the second dimension of input self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group (int64_t)</td>
        <td>Input</td>
        <td>Indicates that the second dimension of input self is divided into group groups.</td>
        <td>The first dimension of self must be divisible by group.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>eps (double)</td>
        <td>Input</td>
        <td>eps in the formula.</td>
        <td>eps must be greater than 0.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>out in the formula.</td>
        <td>The data type is the same as self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>Same as self</td>
        <td>√</td>
    </tr>
    <tr>
        <td>meanOut (aclTensor*)</td>
        <td>Output</td>
        <td>meanOut in the formula.</td>
        <td>Data type is consistent with self, and N in the shape is the size of the first dimension of self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>rstdOut (aclTensor*)</td>
        <td>Output</td>
        <td>rstdOut in the formula.</td>
        <td>Data type is consistent with self, and N in the shape is the size of the first dimension of self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>
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
    </tbody></table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: Data type BFLOAT16 is not supported.
  <!-- end id7 -->

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: gamma or beta can be empty. When gamma or beta is not empty, meanOut and rstdOut must have the same data type as gamma or beta. When both gamma and beta are empty, meanOut and rstdOut must have the same data type as self.
  <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
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
      <td>161001 is returned if a required input, output, or attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of the input and output are out of the supported range.</td>
    </tr>
    <tr>
      <td>The input and output parameters do not meet the constraints described in Parameters.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSilu

- **Parameters**

  <table>
  <thead>
    <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnGroupNormSiluGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td> Op executor, containing the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td> Specifies the Stream for task execution.</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnGroupNormSilu defaults to a deterministic implementation.

- Input shape constraints:
    1. The size of the second dimension of self must be divisible by group.
    2. The shape of meanOut and rstdOut must be (N, group), where N is the size of the first dimension of self.
- Input attribute restriction: eps > 0.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <cinttypes>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_silu.h"

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
  // Boilerplate for AscendCL initialization
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
  // Call aclrtMalloc to allocate memory on the device
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the list of ACL APIs.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {2, 3, 4};
  std::vector<int64_t> gammaShape = {3};
  std::vector<int64_t> betaShape = {3};
  std::vector<int64_t> outShape = {2, 3, 4};
  std::vector<int64_t> meanOutShape = {2, 1};
  std::vector<int64_t> rstdOutShape = {2, 1};
  void* selfDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* out = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;
  std::vector<float> selfHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> gammaHostData = {2.0, 2, 2};
  std::vector<float> betaHostData = {2.0, 2, 2};
  std::vector<float> outHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> meanOutHostData = {2.0, 2};
  std::vector<float> rstdOutHostData = {2.0, 2};

  int64_t group = 1;
  double eps = 0.00001;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the beta aclTensor.
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the meanOut aclTensor.
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the rstdOut aclTensor.
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnGroupNormSilu first-phase API.
  ret = aclnnGroupNormSiluGetWorkspaceSize(self, gamma, beta, group, eps, out, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnGroupNormSilu second-phase API.
  ret = aclnnGroupNormSilu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSilu failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Wait for the task execution to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values and copy the results from the device side to the host side. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outResultData(size, 0);
  ret = aclrtMemcpy(outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outResultData[%" PRId64 "] is: %f\n", i, outResultData[i]);
  }

  size = GetShapeSize(meanOutShape);
  std::vector<float> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(), meanResultData.size() * sizeof(meanResultData[0]), meanOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResultData[%" PRId64 "] is: %f\n", i, meanResultData[i]);
  }

  size = GetShapeSize(rstdOutShape);
  std::vector<float> rstdResultData(size, 0);
  ret = aclrtMemcpy(rstdResultData.data(), rstdResultData.size() * sizeof(rstdResultData[0]), rstdOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("rstdResultData[%" PRId64 "] is: %f\n", i, rstdResultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(out);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. Release device resources, which need to be modified based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  aclrtFree(rstdOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
