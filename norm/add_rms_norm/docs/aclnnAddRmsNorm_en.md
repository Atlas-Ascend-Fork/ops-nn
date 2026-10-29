# aclnnAddRmsNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:37:49.308Z -->

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

- Description: The RMSNorm operator is a commonly used normalization operation in large models. Compared with the LayerNorm operator, it removes the part of subtracting the mean. The AddRmsNorm operator fuses the Add operator before RMSNorm to reduce data movement.
- Formula:

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnAddRmsNormGetWorkspaceSize` first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call `aclnnAddRmsNorm` to perform the computation.

```Cpp
aclnnStatus aclnnAddRmsNormGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  double           epsilon,
  aclTensor       *yOut,
  aclTensor       *rstdOut,
  aclTensor       *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormGetWorkspaceSize

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
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the first input for the Add computation. Corresponds to `x1` in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the second input for the Add computation. Corresponds to `x2` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>shape and data type must keep consistent with those of `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scaling factor (weight) for RmsNorm. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Data type must keep consistent with that of `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability. Corresponds to `epsilon` in the formula.</td>
      <td><ul><li>The value of epsilon must be greater than or equal to zero.</li><li>The recommended value is 1e-6.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>The final output. Corresponds to `RmsNorm(x)` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>shape and data type keep consistent with those of input `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the reciprocal of the standard deviation after normalization. Corresponds to the reciprocal of `Rms(x)` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The number of dimensions must keep consistent with `x1`. Dimensions that do not need normalization keep consistent with the corresponding dimensions of `x1`. Dimensions that need normalization (the trailing dimensions with the same number of dimensions as `gamma`) are all 1. Example of the relationship among rstdOut shape, x1 shape, and gamma shape: if x1 shape is (2,3,4,8) and gamma shape is (8), rstdOut shape is (2,3,4,1); if x1 shape is (2,3,4,8) and gamma shape is (4,8), rstdOut shape is (2,3,1,1).</li><li>If nullptr is passed in, the final output of this parameter is invalid.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of the Add computation. Corresponds to `x` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>shape and data type must keep consistent with those of input `x1`.</li><li>If nullptr is passed for `xOut`, the final output of this parameter is invalid.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
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

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>:
    - The data type of parameters `x1`, `x2`, `gamma`, `yOut`, and `xOut` does not support BFLOAT16.
    - The parameter `rstdOut` is invalid in the current product usage scenario.
  <!-- end id7 -->

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
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>x1, x2, gamma, or yOut is a null pointer.</td>
    </tr>
    <tr>
      <td>The preset value of rstdOut is not nullptr, but the preset value of xOut is nullptr.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>The input and output parameters do not meet the constraints described in Parameters.</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNorm

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnAddRmsNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return code. (For details, see [aclnn return code](../../../docs/en/context/aclnn_return_code.md).)

## Constraints

- Boundary value scenarios
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.

- The supported combinations of inputs x1, x2, gamma, yOut, rstdOut, and xOut are as follows:

  <!-- npu="950,A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

    | x1 | x2 | Gamma | yOut | rstdOut | xOut |
    | --------| --------| --------| --------| --------| :------ |
    | FLOAT32 | FLOAT32 | shape must be consistent with the last few dimensions of `x1`, where the last few dimensions are the dimensions of `x1` to be normalized; FLOAT32 | FLOAT32 | Required | Required, FLOAT32 |
    | FLOAT16 | FLOAT16 | shape must be consistent with the last few dimensions of `x1`, where the last few dimensions are the dimensions of `x1` to be normalized; FLOAT16 | FLOAT16 | Required | Required, FLOAT16 |
    | BFLOAT16 | BFLOAT16 | shape must be consistent with the last few dimensions of `x1`, where the last few dimensions are the dimensions of `x1` to be normalized; BFLOAT16 | BFLOAT16 | Required | Required, BFLOAT16 |
    | FLOAT16 | FLOAT16 | shape is [1, the last dimension of x1]; FLOAT16 | FLOAT16 | Null pointer. | null pointer, FLOAT16 |
    | BFLOAT16 | BFLOAT16 | shape is [1, the last dimension of x1]; BFLOAT16 | BFLOAT16 | Null pointer. | null pointer, BFLOAT16 |
    | FLOAT16 | FLOAT16 | shape is [1, the last dimension of x1]; FLOAT16 | FLOAT16 | Null pointer. | Required, FLOAT16 |
    | BFLOAT16 | BFLOAT16 | shape is [1, the last dimension of x1]; BFLOAT16 | BFLOAT16 | Null pointer. | Required, BFLOAT16 |
  <!-- end id8 -->

  <!-- npu="310p" id9 -->
  - <term>Atlas inference products</term>:

    | x1 | x2 | Gamma | yOut | rstdOut | xOut |
    | --------| --------| --------| --------| --------| :------ |
    | FLOAT32 | FLOAT32 | shape must be consistent with the last few dimensions of `x1`, where the last few dimensions are the dimensions of `x1` to be normalized; FLOAT32 | FLOAT32 | Required | Required, FLOAT32 |
    | FLOAT16 | FLOAT16 | shape must be consistent with the last few dimensions of `x1`, where the last few dimensions are the dimensions of `x1` to be normalized; FLOAT16 | FLOAT16 | Required | Required, FLOAT16 |
    | FLOAT16 | FLOAT16 | shape is [1, the last dimension of x1]; FLOAT16 | FLOAT16 | Null pointer. | null pointer, FLOAT16 |
    | FLOAT16 | FLOAT16 | shape is [1, the last dimension of x1]; FLOAT16 | FLOAT16 | Null pointer. | Required, FLOAT16 |
  <!-- end id9 -->

- Deterministic computation:
  - aclnnAddRmsNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate for resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId); aclFinalize(); return ret);

    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> xShape = {2, 16};
    std::vector<int64_t> gammaShape = {16};
    std::vector<int64_t> yShape = {2, 16};
    std::vector<int64_t> rstdShape = {2, 1};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* y = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* x = nullptr;
    std::vector<float> x1HostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                     0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> x2HostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                     0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> gammaHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> yHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> rstdHostData = {1, 2};
    std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    float epsilon = 1e-6;

    // Create the x1 aclTensor.
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x2 aclTensor.
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAddRmsNorm.
    ret = aclnnAddRmsNormGetWorkspaceSize(x1, x2, gamma, epsilon, y, rstd, x, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddRmsNorm.
    ret = aclnnAddRmsNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNorm failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, copy the result from the device memory to the host side, and modify it based on the specific API definition.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor, and modify it based on the specific API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(y);
    aclDestroyTensor(rstd);
    aclDestroyTensor(x);

    // 7. Release device resources, and modify it based on the specific API definition.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
