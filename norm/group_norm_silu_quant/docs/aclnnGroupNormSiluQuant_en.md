# aclnnGroupNormSiluQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:08:53.518Z -->

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

- Description: Computes the group normalization of the input self, outputs the mean meanOut, the reciprocal of the standard deviation rstdOut, and the quantized result out of the SiLU output.
- Formula:
  - **GroupNorm:**

    Let $E[x] = \bar{x}$ denote the mean of $x$, and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ denote the variance of $x$, then

    $$
    \left\{
    \begin{array} {rcl}
    groupNormOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Silu:**

    $$
    siluOut = \frac{groupNormOut}{1+e^{-groupNormOut}}
    $$

  - **Quant:**

    $$
    out = round(siluOut / quantScale)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGroupNormSiluQuantGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnGroupNormSiluQuant" API to perform the computation.

```c++
aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* gammaOptional,
    const aclTensor* betaOptional,
    const aclTensor* quantScale,
    int64_t          group,
    double           eps,
    bool             activateSilu,
    aclTensor*       out,
    aclTensor*       meanOut,
    aclTensor*       rstdOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);
```

```c++
aclnnStatus aclnnGroupNormSiluQuant(
    void *         workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupNormSiluQuantGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 146px">
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
        <td>self</td>
        <td>Input</td>
        <td>x in the formula.</td>
        <td>Empty tensors are supported: N (dim 0) and C (dim 1) must be greater than 0, and the remaining dimensions can be 0. In this case, out is empty, meanOut is filled with 0, and rstdOut is filled with NAN.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2-8, where dim 0 is N and dim 1 is C</td>
        <td>√</td>
    </tr>
    <tr>
        <td>gammaOptional</td>
        <td>Optional input</td>
        <td>γ in the formula.</td>
        <td>Default value of elements is 1 when empty. The data type remains consistent with self, and the number of elements must be the same as that of the 1st dimension of input self.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>betaOptional</td>
        <td>Optional input</td>
        <td>β in the formula.</td>
        <td>Default value of elements is 1 when empty. The data type remains consistent with self, and the number of elements must be the same as that of the 1st dimension of input self.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>quantScale</td>
        <td>Input</td>
        <td>quantScale in the formula.</td>
        <td>The number of elements must be 1 or the same as the first dimension of the input self.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Input</td>
        <td>Indicates that the first dimension of input self is divided into group groups.</td>
        <td>The first dimension of self must be divisible by group.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>eps</td>
        <td>Input</td>
        <td>eps in the formula.</td>
        <td>eps must be greater than 0.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activateSilu</td>
        <td>Input</td>
        <td>Whether to enable silu computation.</td>
        <td>Currently, only enabling is supported.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out</td>
        <td>Output</td>
        <td>Quantized result, corresponding to out in the formula.</td>
        <td>-</td>
        <td>INT8</td>
        <td>ND</td>
        <td>Consistent with self</td>
        <td>√</td>
    </tr>
    <tr>
        <td>meanOut</td>
        <td>Output</td>
        <td>The meanOut in the formula.</td>
        <td>The data type remains consistent with self, and N in shape remains consistent with dimension 0 of self.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>rstdOut</td>
        <td>Output</td>
        <td>rstdOut in the formula.</td>
        <td>The data type is consistent with self, and N in the shape remains consistent with the 0th dimension of self.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
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
      <td>If a parameter that is a required input, output, or required attribute is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of the inputs and outputs are not within the supported range.</td>
    </tr>
    <tr>
      <td>The input and output parameters do not meet the constraints described in Parameters.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSiluQuant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
    <col style="width: 173px">
    <col style="width: 112px">
    <col style="width: 668px">
    </colgroup>
        <thead>
            <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
        </thead>
        <tbody>
            <tr><td>workspace</td><td>Input</td><td>Memory address of the workspace allocated on the Device.</td></tr>
            <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnGroupNormSiluQuantGetWorkspaceSize.</td></tr>
            <tr><td>executor</td><td>Input</td><td> op executor, which contains the operator computation flow. </td></tr>
            <tr><td>stream</td><td>Input</td><td> Stream where the task is executed. </td></tr>
        </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupNormSiluQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_silu_quant.h"

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
  // Boilerplate, AscendCL initialization
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
  // Call aclrtMalloc to allocate device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory
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
  // 1. (boilerplate) device/stream initialization, see the acl external API list for details.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs, which must be custom-built based on the API.
  std::vector<int64_t> selfShape = {1, 2, 4, 4};    // Main input.
  std::vector<int64_t> gammaShape = {2};      // gamma parameter.
  std::vector<int64_t> betaShape = {2};       // beta parameter
  std::vector<int64_t> quantScaleShape = {1}; // quantization scale factor
  std::vector<int64_t> outShape = {1, 2, 4, 4};     // quantized output
  std::vector<int64_t> meanOutShape = {1, 2};  // mean output [N, G]
  std::vector<int64_t> rstdOutShape = {1, 2};   // reciprocal standard deviation output [N, G]

  void* selfDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* quantScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* quantScale = nullptr;
  aclTensor* out = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;
  // Use uint16_t to store the bit patterns of BF16/FP16.
  std::vector<uint16_t> selfHostData = {
  0x3C00, 0x4000, 0x4200, 0x4400, // FP16 bit patterns of 1.0, 2.0, 3.0, 4.0
  0x4500, 0x4600, 0x4700, 0x4800, // FP16 bit patterns of 5.0, 6.0, 7.0, 8.0
  0x3C00, 0x4000, 0x4200, 0x4400, // FP16 bit patterns of 1.0, 2.0, 3.0, 4.0
  0x4500, 0x4600, 0x4700, 0x4800, // FP16 bit patterns of 5.0, 6.0, 7.0, 8.0
  0x3C00, 0x4000, 0x4200, 0x4400, // FP16 bit patterns of 1.0, 2.0, 3.0, 4.0
  0x4500, 0x4600, 0x4700, 0x4800, // FP16 bit patterns of 5.0, 6.0, 7.0, 8.0
  0x3C00, 0x4000, 0x4200, 0x4400, // FP16 bit patterns of 1.0, 2.0, 3.0, 4.0
  0x4500, 0x4600, 0x4700, 0x4800 // FP16 bit patterns of 5.0, 6.0, 7.0, 8.0
  };
  std::vector<uint16_t> gammaHostData = {0x3C00, 0x3C00}; // FP16 bit patterns of 1.0, 1.0
  std::vector<uint16_t> betaHostData = {0x0000, 0x0000}; // FP16 bit patterns of 0.0, 0.0
  std::vector<float> quantScaleHostData = {1.0};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Initialized to 0, actually filled by the operator.
  // Statistics outputs also use half-precision format.
  std::vector<uint16_t> meanOutHostData = {0x0000, 0x0000}; // Initialized to 0.
  std::vector<uint16_t> rstdOutHostData = {0x0000, 0x0000}; // Initialized to 0.

  int64_t group = 2;
  double eps = 0.00001;
  bool activateSilu = true;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the beta aclTensor.
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the quantScale aclTensor.
  ret = CreateAclTensor(quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the meanOut aclTensor.
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT16, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the rstdOut aclTensor.
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT16, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnGroupNormSiluQuant first-phase API.
  ret = aclnnGroupNormSiluQuantGetWorkspaceSize(self, gamma, beta, quantScale, group, eps, activateSilu, out, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API aclnnGroupNormSiluQuant.
  ret = aclnnGroupNormSiluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluQuant failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values and copy the results from device memory to host memory. Modify as required by the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> outResultData(size, 0);
  ret = aclrtMemcpy(outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(int8_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outResultData[%ld] is: %d\n", i, outResultData[i]);
  }

  // Receive the meanOut result.
  size = GetShapeSize(meanOutShape);
  std::vector<uint16_t> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(),
                    meanResultData.size() * sizeof(uint16_t), // 2 bytes
                    meanOutDeviceAddr,
                    size * sizeof(uint16_t), // 2 bytes
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy meanOut from device to host failed. ERROR: %d\n", ret); return ret);

  // Receive the rstdOut result.
  size = GetShapeSize(rstdOutShape);
  std::vector<uint16_t> rstdResultData(size, 0);
  ret = aclrtMemcpy(rstdResultData.data(),
                    rstdResultData.size() * sizeof(uint16_t), // 2 bytes
                    rstdOutDeviceAddr,
                    size * sizeof(uint16_t), // 2 bytes
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rstdOut from device to host failed. ERROR: %d\n", ret); return ret);

  // Print the results (converted to float for readability).
  for (int64_t i = 0; i < meanResultData.size(); i++) {
    // Simple conversion: convert the FP16 bit pattern to float.
    __fp16 fp16_val = *reinterpret_cast<__fp16*>(&meanResultData[i]);
    float fp32_val = static_cast<float>(fp16_val);
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, fp32_val);
  }

  for (int64_t i = 0; i < rstdResultData.size(); i++) {
    __fp16 fp16_val = *reinterpret_cast<__fp16*>(&rstdResultData[i]);
    float fp32_val = static_cast<float>(fp16_val);
    LOG_PRINT("rstdResultData[%ld] is: %f\n", i, fp32_val);
  }

  // 6. Release aclTensor and aclScalar, which needs to be modified according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(quantScale);
  aclDestroyTensor(out);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. Release device resources, which needs to be modified according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(quantScaleDeviceAddr);
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
