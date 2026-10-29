# aclnnDeepNormGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:03:11.235Z -->

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

- Description: Backward propagation of [aclnnDeepNorm](../../deep_norm/docs/aclnnDeepNorm.md). Computes the gradients of tensor x, tensor gx, and tensor gamma, and the sum of tensor dy.

- Formula:

  $$
  dgx_i = tmpone_i * rstd + dvar * tmptwo_i + dmean
  $$

  $$
  dx_i = alpha * {dgx}_i
  $$

  $$
  dbeta = \sum_{i=1}^{N} dy_i
  $$

  $$
  dgamma =  \sum_{i=1}^{N} dy_i * rstd * {tmptwo}_i
  $$

  Where:

  $$
  oneDiv=-1/SizeOf(gamma)
  $$

  $$
  tmpone_i = dy_i * gamma
  $$

  $$
  tmptwo_i = alpha * x_i + {gx}_i - mean
  $$

  $$
  dvar = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * {tmptwo}_i * {rstd}^3
  $$

  $$
  dmean = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * rstd
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnDeepNormGradGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call the "aclnnDeepNormGrad" API to perform the computation.

```Cpp
aclnnStatus aclnnDeepNormGradGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x,
  const aclTensor *gx,
  const aclTensor *gamma,
  const aclTensor *mean,
  const aclTensor *rstd,
  double           alpha,
  const aclTensor *dxOut,
  const aclTensor *dgxOut,
  const aclTensor *dbetaOut,
  const aclTensor *dgammaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDeepNormGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDeepNormGradGetWorkspaceSize

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
      <td>dy (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the main grad input, corresponding to `dy` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The data type and shape are consistent with those of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor of the forward fusion operator, corresponding to `x` in the formula.</td>
      <td>Supports null Tensor.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gx (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor of the forward fused operator, corresponding to `gx` in the corresponding formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>The data type and shape keep consistent with those of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scaling parameter indicating forward propagation, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The data type must keep consistent with the data type of `x`.</li><li>The shape dimensions are the same as the last few dimensions of input `x`. The last few dimensions indicate the dimensions to normalize.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the mean of the sum of forward inputs x and gx, corresponding to `mean` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape dimensions are the same as the first several dimensions of input `x`. These dimensions do not require norm.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the rstd of the sum of forward inputs x and gx, corresponding to `rstd` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape must keep consistent with the shape of `mean`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha (double)</td>
      <td>Input</td>
      <td>Indicates the weight parameter used to adjust the weight of input data, corresponding to `alpha` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates computation of the output gradient, used to update the gradient of input data x. Corresponds to `dx` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>Keeps consistent with `x` in data type and shape.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dgxOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the computation output gradient, used to update the gradient of input data gx. Corresponds to `dgx` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The data type and shape are consistent with those of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dbetaOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output gradient of the computation, used to update the gradient of the bias parameter. Corresponds to `dbeta` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape is consistent with `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dgammaOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the computation of the output gradient, used to update the gradient of the scaling parameter. Corresponds to `dgamma` in the formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>The shape keeps consistent with `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that the user needs to side-request on the Device.</td>
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

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: The data type of parameters `dy`, `x`, `gx`, `gamma`, `dxOut`, and `dgxOut` does not support BFLOAT16.
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
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If a passed-in parameter is a mandatory input, output, or mandatory attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the input or output is out of the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of the input and output do not match or are out of the supported dimension range.</td>
    </tr>
    <tr>
      <td>The data types of the input and output do not meet the constraints described in Parameters.</td>
    </tr>
  </tbody></table>

## aclnnDeepNormGrad

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
      <td>Workspace memory address side-requested on Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size side-requested on Device, obtained by the first-phase API aclnnDeepNormGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Unsupported type description:

  DOUBLE: The instruction does not support DOUBLE.
- Boundary value scenario description:
  * When the input is Inf, the output is Inf.
  * When the input is NaN, the output is NaN.
- Deterministic computation:
  - aclnnDeepNormGrad defaults to a non-deterministic implementation. Deterministic computation cannot be enabled through aclrtCtxSetSysParamOpt.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_deep_norm_grad.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate, resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); aclrtFree(*deviceAddr); *deviceAddr = nullptr;
              return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    CHECK_RET(*tensor != nullptr,
              LOG_PRINT("aclCreateTensor failed.\n"); aclrtFree(*deviceAddr); *deviceAddr = nullptr;
              return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
    float alpha = 0.3;
    std::vector<int64_t> dyShape = {3, 1, 4};
    std::vector<int64_t> xShape = {3, 1, 4};
    std::vector<int64_t> gxShape = {3, 1, 4};
    std::vector<int64_t> gammaShape = {4};
    std::vector<int64_t> meanShape = {3, 1, 1};
    std::vector<int64_t> rstdShape = {3, 1, 1};
    std::vector<int64_t> outputpdxShape = {3, 1, 4};
    std::vector<int64_t> outputpdgxShape = {3, 1, 4};
    std::vector<int64_t> outputpdbetaShape = {4};
    std::vector<int64_t> outputpdgammaShape = {4};
    void* dyDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* gxDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* outputpdxDeviceAddr = nullptr;
    void* outputpdgxDeviceAddr = nullptr;
    void* outputpdbetaDeviceAddr = nullptr;
    void* outputpdgammaDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclTensor* gx = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* outputpdx = nullptr;
    aclTensor* outputpdgx = nullptr;
    aclTensor* outputpdbeta = nullptr;
    aclTensor* outputpdgamma = nullptr;

    std::vector<float> dyHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> gxHostData = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8};
    std::vector<float> gammaHostData = {0, 1, 2, 3};
    std::vector<float> meanHostData = {0, 1, 2};
    std::vector<float> rstdHostData = {0, 1, 2};
    std::vector<float> outputpdxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> outputpdgxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> outputpdbetaHostData = {0, 1, 2, 3};
    std::vector<float> outputpdgammaHostData = {0, 1, 2, 3};

    // Create the self aclTensor.
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gxHostData, gxShape, &gxDeviceAddr, aclDataType::ACL_FLOAT, &gx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputpdxHostData, outputpdxShape, &outputpdxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgxHostData, outputpdgxShape, &outputpdgxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdbetaHostData, outputpdbetaShape, &outputpdbetaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdbeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgammaHostData, outputpdgammaShape, &outputpdgammaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnDeepNormGrad API call example.
    // 3. Call the CANN operator library API. Change it to the specific API name.
    // Call the first-phase API of aclnnDeepNormGrad.
    LOG_PRINT("\nUse aclnnDeepNormGrad Port.");
    ret = aclnnDeepNormGradGetWorkspaceSize(
        dy, x, gx, gamma, mean, rstd, alpha, outputpdx, outputpdgx, outputpdbeta, outputpdgamma, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNormGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnDeepNormGrad.
    ret = aclnnDeepNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNormGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, copy the result from device-side memory to the host side, and modify it based on the specific API definition.
    auto outputpdxsize = GetShapeSize(outputpdxShape);
    std::vector<float> resultDataPdx(outputpdxsize, 0);
    ret = aclrtMemcpy(
        resultDataPdx.data(), resultDataPdx.size() * sizeof(resultDataPdx[0]), outputpdxDeviceAddr,
        outputpdxsize * sizeof(resultDataPdx[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdx output");
    for (int64_t i = 0; i < outputpdxsize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdx[i]);
    }
    auto outputpdgxsize = GetShapeSize(outputpdgxShape);
    std::vector<float> resultDataPdgx(outputpdgxsize, 0);
    ret = aclrtMemcpy(
        resultDataPdgx.data(), resultDataPdgx.size() * sizeof(resultDataPdgx[0]), outputpdgxDeviceAddr,
        outputpdgxsize * sizeof(resultDataPdgx[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdgx output");
    for (int64_t i = 0; i < outputpdgxsize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdgx[i]);
    }
    auto outputpdbetasize = GetShapeSize(outputpdbetaShape);
    std::vector<float> resultDataPdBeta(outputpdbetasize, 0);
    ret = aclrtMemcpy(
        resultDataPdBeta.data(), resultDataPdBeta.size() * sizeof(resultDataPdBeta[0]), outputpdbetaDeviceAddr,
        outputpdbetasize * sizeof(resultDataPdBeta[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdbeta output");
    for (int64_t i = 0; i < outputpdbetasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdBeta[i]);
    }
    auto outputpdgammasize = GetShapeSize(outputpdgammaShape);
    std::vector<float> resultDataPdGamma(outputpdgammasize, 0);
    ret = aclrtMemcpy(
        resultDataPdGamma.data(), resultDataPdGamma.size() * sizeof(resultDataPdGamma[0]), outputpdgammaDeviceAddr,
        outputpdgammasize * sizeof(resultDataPdGamma[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdgamma output");
    for (int64_t i = 0; i < outputpdgammasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdGamma[i]);
    }

    // 6. Release aclTensor and aclScalar, and modify it based on the specific API definition.
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyTensor(gx);
    aclDestroyTensor(gamma);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(outputpdx);
    aclDestroyTensor(outputpdgx);
    aclDestroyTensor(outputpdbeta);
    aclDestroyTensor(outputpdgamma);

    // 7. Release device resources, and modify it based on the specific API definition.
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(gxDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(outputpdxDeviceAddr);
    aclrtFree(outputpdgxDeviceAddr);
    aclrtFree(outputpdbetaDeviceAddr);
    aclrtFree(outputpdgammaDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
