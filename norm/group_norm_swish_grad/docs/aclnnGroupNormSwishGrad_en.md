# aclnnGroupNormSwishGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:18:19.319Z -->

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

- Description: Backward operation of [aclnnGroupNormSwish](../../group_norm_swish/docs/aclnnGroupNormSwish.md).
- Formula:

$$
dYTemp_i = \hat{x_i} \cdot gamma + beta
$$

$$
dSwishTemp_i = swishScale \cdot dYTemp_i - \frac{swishScale \cdot dYTemp_i}{\exp(-swishScale \cdot dYTemp_i) + 1} + 1
$$

$$
dYNew_i = \frac{dSwishTemp_i}{\exp(-swishScale \cdot dYTemp_i) + 1} \cdot dy
$$

$$
dBeta = \sum_{i=1}^n dYNew_i
$$

$$
dGamma = \sum_{i=1}^n (dYNew_i \cdot \hat{x_i})
$$

$$
dx = rstd \cdot (dYNew * gamma - \hat{x} * (\sum_{i=1}^n gamma_i * dGamma) - (\sum_{i=1}^n gamma_i * dBeta))
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGroupNormSwishGradGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnGroupNormSwishGrad" to perform the computation.

```c++
aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *mean,
    const aclTensor *rstd,
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *beta,
    int64_t          numGroups,
    char            *dataFormatOptional,
    double           swishScale,
    bool             dgammaIsRequire,
    bool             dbetaIsRequire,
    const aclTensor *dxOut,
    const aclTensor *dgammaOut,
    const aclTensor *dbetaOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnGroupNormSwishGrad(
    void *         workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupNormSwishGradGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
      <col style="width: 220px">
      <col style="width: 120px">
      <col style="width: 287px">
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
          <th>Non-contiguous Tensor</th>
      </tr></thead>
      <tbody>
      <tr>
          <td>dy (aclTensor*)</td>
          <td>Input</td>
          <td>Gradient for backward computation, <code>dy</code> in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>Dimensions range from 2D to 8D, where the first dimension is N and the second dimension is C.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>√</td>
      </tr>
      <tr>
          <td>mean (aclTensor*)</td>
          <td>Input</td>
          <td>Second output of the forward computation, indicating the mean of each group after input is grouped, <code>mean</code> in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>The data type is the same as that of gamma, and N is the same as dimension 0 of dy.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>rstd (aclTensor*)</td>
          <td>Input</td>
          <td>The third output of forward computation, indicating the reciprocal of the standard deviation of each group after input is grouped, corresponding to <code>rstd</code> in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>The data type is the same as that of gamma, and N is the same as dimension 0 of dy.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>x (aclTensor*)</td>
          <td>Input</td>
          <td>Forward input, <code>x</code> in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>The data type and shape are the same as those of dy.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>√</td>
      </tr>
      <tr>
          <td>gamma (aclTensor*)</td>
          <td>Input</td>
          <td>Scaling coefficient for each channel, which is <code>gamma</code> in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>The data type and dimensions must be the same as those of dy, and the number of elements must be equal to C.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
      </tr>
      <tr>
          <td>beta (aclTensor*)</td>
          <td>Input</td>
          <td>Offset coefficient of each channel, which is <code>beta</code> in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>The data type and dimensions are the same as those of dy, and the number of elements must be equal to C.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
      </tr>
      <tr>
          <td>numGroups (int64_t)</td>
          <td>Input</td>
          <td>The C dimension of the input x is divided into numGroups groups.</td>
          <td>group must be greater than 0. C must be divisible by group, and the ratio must not exceed 4000.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dataFormatOptional (char*)</td>
          <td>Input</td>
          <td>Data format.</td>
          <td>Recommended value: NCHW.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>swishScale (double)</td>
          <td>Input</td>
          <td>Computation coefficient.</td>
          <td>Recommended value: 1.0.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dgammaIsRequire (bool)</td>
          <td>Input</td>
          <td>Whether to output dgamma.</td>
          <td>Recommended value: TRUE.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dbetaIsRequire (bool)</td>
          <td>Input</td>
          <td>Whether dbeta needs to be output.</td>
          <td>Recommended value: TRUE.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dxOut (aclTensor*)</td>
          <td>Output</td>
          <td><code>dx</code> in the formula.</td>
          <td>Same data type and shape as x.</td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>x</td>
      </tr>
      <tr>
          <td>dgammaOut (aclTensor*)</td>
          <td>Output</td>
          <td><code>dGamma</code> in the formula.</td>
          <td>Same data type and shape as gamma.</td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>x</td>
      </tr>
      <tr>
          <td>dbetaOut (aclTensor*)</td>
          <td>Output</td>
          <td>Output gradient of beta, which is <code>dBeta</code> in the formula.</td>
          <td>The data type and shape are the same as those of gamma.</td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>x</td>
      </tr>
      <tr>
          <td>workspaceSize (uint64_t*)</td>
          <td>Output</td>
          <td>Returns the size of the workspace to be applied for on the Device.</td>
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
      <td>dy, mean, rstd, x, gamma, beta, dxOut, dgammaOut, or dbetaOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of dy is not supported.</td>
    </tr>
    <tr>
      <td>The data types of mean, rstd, x, gamma, and beta are not the same as that of dy.</td>
    </tr>
    <tr>
      <td>The data type of dxOut is not the same as that of dy.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSwishGrad

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
          <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnGroupNormSwishGradGetWorkspaceSize.</td>
      </tr>
      <tr>
          <td>executor</td>
          <td>Input</td>
          <td> op executor, containing the operator computation flow.</td>
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

- Deterministic Computation
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: aclnnGroupNormSwishGrad defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnGroupNormSwishGrad defaults to a deterministic implementation.
  <!-- end id8 -->

- Input shape constraints:
    1. numGroups is greater than 0.
    2. C is divisible by group.
    3. The number of elements of dy equals $N * C * HxW$.
    4. The number of elements of mean equals $N * group$.
    5. The number of elements of rstd equals $N * group$.
    6. The number of elements of x equals $N * C * HxW$.
    7. The number of elements in gamma equals C.
    8. The number of elements in beta equals C.
    9. The ratio of C to group does not exceed 4000.

## Example

The following sample code is for reference only. For details about how to compile and run, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_swish_grad.h"

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
  // Call aclrtMalloc to apply for memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> dyShape = {2, 3, 4};
  std::vector<int64_t> meanShape = {2, 1};
  std::vector<int64_t> rstdShape = {2, 1};
  std::vector<int64_t> xShape = {2, 3, 4};
  std::vector<int64_t> gammaShape = {3};
  std::vector<int64_t> betaShape = {3};
  std::vector<int64_t> dxOutShape = {2, 3, 4};
  std::vector<int64_t> dgammaOutShape = {3, 1};
  std::vector<int64_t> dbetaOutShape = {3, 1};
  void* dyDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* rstdDeviceAddr = nullptr;
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* dxOutDeviceAddr = nullptr;
  void* dgammaOutDeviceAddr = nullptr;
  void* dbetaOutDeviceAddr = nullptr;
  aclTensor* dy = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* rstd = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* dxOut = nullptr;
  aclTensor* dgammaOut = nullptr;
  aclTensor* dbetaOut = nullptr;
  std::vector<float> dyHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> meanHostData = {2.0, 2};
  std::vector<float> rstdHostData = {2.0, 2};
  std::vector<float> xHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                  13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> gammaHostData = {2.0, 2, 2};
  std::vector<float> betaHostData = {2.0, 2, 2};
  std::vector<float> dxOutHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> dgammaOutHostData = {2.0, 2, 2};
  std::vector<float> dbetaOutHostData = {2.0, 2, 2};
  int64_t numGroups = 1;
  char* dataFormatOptional = nullptr;
  double swishScale = 1.0;
  bool dgammaIsRequire = true;
  bool dbetaIsRequire = true;
  // Create the dy aclTensor.
  ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mean aclTensor.
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the rstd aclTensor.
  ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the beta aclTensor.
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the dxOut aclTensor.
  ret = CreateAclTensor(dxOutHostData, dxOutShape, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the dgammaOut aclTensor.
  ret = CreateAclTensor(dgammaOutHostData, dgammaOutShape, &dgammaOutDeviceAddr, aclDataType::ACL_FLOAT, &dgammaOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the dbetaOut aclTensor.
  ret = CreateAclTensor(dbetaOutHostData, dbetaOutShape, &dbetaOutDeviceAddr, aclDataType::ACL_FLOAT, &dbetaOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGroupNormSwishGrad
  ret = aclnnGroupNormSwishGradGetWorkspaceSize(dy, mean, rstd, x, gamma, beta, numGroups, dataFormatOptional, swishScale, dgammaIsRequire, dbetaIsRequire, dxOut, dgammaOut, dbetaOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGroupNormSwishGrad
  ret = aclnnGroupNormSwishGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGrad failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronize and wait for the task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value, and copy the result on the device side to the host side. Modify it based on the specific API definition.
  auto size = GetShapeSize(dxOutShape);
  ret = aclrtMemcpy(dxOutHostData.data(), dxOutHostData.size() * sizeof(dxOutHostData[0]), dxOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dxOutHostData[%ld] is: %f\n", i, dxOutHostData[i]);
  }

  size = GetShapeSize(dgammaOutShape);
  ret = aclrtMemcpy(dgammaOutHostData.data(), dgammaOutHostData.size() * sizeof(dgammaOutHostData[0]), dgammaOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dgammaOutHostData[%ld] is: %f\n", i, dgammaOutHostData[i]);
  }

  size = GetShapeSize(dbetaOutShape);
  ret = aclrtMemcpy(dbetaOutHostData.data(), dbetaOutHostData.size() * sizeof(dbetaOutHostData[0]), dbetaOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dbetaOutHostData[%ld] is: %f\n", i, dbetaOutHostData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(dy);
  aclDestroyTensor(mean);
  aclDestroyTensor(rstd);
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(dxOut);
  aclDestroyTensor(dgammaOut);
  aclDestroyTensor(dbetaOut);

  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(dyDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(rstdDeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(dxOutDeviceAddr);
  aclrtFree(dgammaOutDeviceAddr);
  aclrtFree(dbetaOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
