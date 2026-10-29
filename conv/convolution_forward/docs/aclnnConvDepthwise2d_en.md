# aclnnConvDepthwise2d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T17:17:18.111Z -->

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

- Description: DepthwiseConv2D is a 2D depthwise convolution operation. In this operation, each input channel is convolved with a separate kernel (called a depthwise kernel).

- Formula:

  Assuming the shape of input self is $(N, C_{\text{in}}, H, W)$ and the shape of output out is $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$, the output of each kernel is represented as follows:

  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \text{weight}(C_{\text{out}_j}, C_{\text{in}_j}) \star \text{self}(N_i, C_{\text{in}_j})
  $$

  Where $\star$ represents the convolution operation, $N$ represents the batch size, $C$ represents the number of channels, and $W$ and $H$ represent the width and height, respectively.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call aclnnConvDepthwise2dGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, call aclnnConvDepthwise2d to perform the computation.

```cpp
aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(
    const aclTensor       *self,
    const aclTensor       *weight,
    const aclIntArray     *kernelSize,
    const aclTensor       *bias,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    aclTensor             *out,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvDepthwise2d(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnConvDepthwise2dGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:220px">Parameter</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:280px">Description</th>
  <th style="width:350px">Instruction</th>
  <th style="width:200px">Data Type</th>
  <th style="width:120px">Data Format</th>
  <th style="width:130px">Dimension (shape)</th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>self (aclTensor*)</td>
  <td>Input</td>
  <td>self in the formula, representing the convolution input.</td>
  <td><ul><li>Supports empty tensors.</li><li>The data type must satisfy the data type deduction rule with the data type of weight (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a>).</li><li>The shape is (N,C<sub>in</sub>,H<sub>in</sub>,W<sub>in</sub>).</li><li>N≥0, C≥1, H≥0, W≥0.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight (aclTensor*)</td>
  <td>Input</td>
  <td>weight in the formula, representing the convolution weight.</td>
  <td><ul><li>Supports empty tensors.</li><li>The data type must comply with the data type deduction rules for the data type of self (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a>).</li><li>The shape is (C<sub>out</sub>,C<sub>in</sub>/groups,K<sub>H</sub>,K<sub>W</sub>).</li><li>The value of the first dimension of weight must equal an integer multiple of the number of channels of self, and the value of the second dimension can only be 1.</li><li>All dimensions must be greater than or equal to 1, and the H and W dimensions must be smaller than the H and W dimensions of self.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>kernelSize (aclIntArray*)</td>
  <td>Input</td>
  <td>Kernel size.</td>
  <td><ul><li>Tuple of (INT64, INT64).</li><li>The values are those of the H and W dimensions of weight.</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>bias (aclTensor*)</td>
  <td>Input</td>
  <td>bias in the formula, which indicates the convolution bias.</td>
  <td><ul><li>The shape is (C<sub>out</sub>).</li><li>1D, and the value must equal the first dimension of weight.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>stride (aclIntArray*)</td>
  <td>Input</td>
  <td>Scan stride of the convolution.</td>
  <td><ul><li>The array length must equal the number of dimensions of self minus 2.</li><li>strideH and strideW ∈ [1,63].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding (aclIntArray*)</td>
  <td>Input</td>
  <td>The padding for self.</td>
  <td><ul><li>The array length must equal the self dimension minus 2.</li><li>paddingH and paddingW ∈ [0,255].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation (aclIntArray*)</td>
  <td>Input</td>
  <td>The spacing between elements in the convolution kernel.</td>
  <td><ul><li>Array length must equal the dimension of self minus 2.</li><li>dilationH and dilationW∈[1,255].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>out (aclTensor*)</td>
  <td>Output</td>
  <td>out in the formula, representing the convolution output.</td>
  <td><ul><li>Supports empty tensor.</li><li>The shape is (N,C<sub>out</sub>,H<sub>out</sub>,W<sub>out</sub>).</li><li>The number of channels equals the first dimension of weight. H≥0, W≥0, and other dimensions ≥1.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>cubeMathType(int8_t)</td>
  <td>Input</td>
  <td>Determines the computation logic to be used by the Cube unit.</td>
  <td><ul><li> 0 (KEEP_DTYPE): Computes with the input data type retained.</li></ul><ul><li> 1 (ALLOW_FP32_DOWN_PRECISION): Allows reduced-precision computation of FLOAT to improve performance.</li></ul><ul><li> 2 (USE_FP16): Computes with FLOAT16 precision.</li></ul><ul><li> 3 (USE_HF32): Computes with HFLOAT32 (mixed precision).</li></ul></td>
  <td>INT8</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*)</td>
  <td>Output</td>
  <td>Returns the workspace size that needs to be applied by the Device side.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor (aclOpExecutor**)</td>
  <td>Output</td>
  <td>Returns the op executor, which includes the computation process of the operator.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **Return Value**

  `aclnnStatus`: return status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Codes</a>.

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
   <thead>
  <tr>
  <td>Return Value</td>
  <td>Error Code</td>
  <td>description</td>
  </tr> </thead>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The pointer-type input parameter is a null pointer.</td>
  </tr>
  <tr>
  <td rowspan="6" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="6" align="left">161002</td>
  <td align="left">The data types and data formats of self, weight, bias, and out are not supported.</td>
  </tr>
  <tr><td align="left">The data types of self, weight, and out are inconsistent.</td></tr>
  <tr><td align="left">The input shapes of stride, padding, and dilation are incorrect.</td></tr>
  <tr><td align="left">The number of channels of weight and self does not meet the requirement.</td></tr>
  <tr><td align="left">The shape of out does not match the infer_shape result.</td></tr>
  <tr><td align="left">self, weight, bias, or out is an unsupported empty tensor input or output.</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">API internal verification error, usually caused by input data or attribute specifications that are out of the supported range.</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">API call to npu runtime fails, for example, soc_version is not supported.</td>
  </tr>
  </table>

## aclnnConvDepthwise2d

- **Parameters**

  <table>
  <tr>
  <th style="width:120px">Parameter</th>
  <th style="width:80px">Input/Output</th>
  <th>Description</th>
  </tr>
  <tr>
  <td>workspace</td>
  <td>Input</td>
  <td>Workspace memory address applied by the Device.</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Input</td>
  <td>Size of the workspace applied by the Device, obtained by the first-phase API aclnnConvDepthwise2dGetWorkspaceSize.</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>Input</td>
  <td>op executor, which contains the operator computation flow.</td>
  </tr>
  <tr>
  <td>stream</td>
  <td>Input</td>
  <td>Stream for task execution.</td>
  </tr>
  </table>

- **Return Value**

  `aclnnStatus`: return status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Codes</a>.

## Constraints

- Deterministic computation
  - aclnnConvDepthwise2d defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width:150px">
  <col style="width:500px">
  <col style="width:500px">
  </colgroup>
  <thead>
  <tr>
    <th>Constraint Type</th>

    <th><term>Atlas A2 products</term>, <term>Atlas A3 products</term></th>


    <th><term>Ascend 950PR&950DT products</term></th>

  </tr>
  </thead>
  <tbody>
  <tr>
    <th scope="row">self, weight</th>
    <td>        <ul>
          <li>HIFLOAT8 is not supported for the data types of self and weight.</li>
          <li>The number of channels of self must be less than or equal to 65535.</li>
        </ul>
      </td>
    <td> - </td>
  </tr>
  <tr>
    <th scope="row">bias</th>
    <td>      HIFLOAT8 and FLOAT8_E4M3FN are not supported for the data type of bias. The data type of bias must be the same as those of input and weight.
    </td>
    <td>      When the data type of self is HIFLOAT8, the data type of bias is eventually converted to FLOAT for computation.
    </td>
  </tr>
  <tr>
    <th scope="row">cubeMathType</th>
    <td>        <ul>
          <li>When cubeMathType is 1 (ALLOW_FP32_DOWN_PRECISION) and the input is FLOAT, the input is allowed to be converted to HFLOAT32 for computation.</li>
          <li>When cubeMathType is 2 (USE_FP16), this option is not supported when the input is BFLOAT16.</li>
          <li>When cubeMathType is 3 (USE_HF32) and the input is FLOAT, the input is converted to HFLOAT32 for computation.</li>
        </ul>
      </td>
    <td>        <ul>
          <li>When cubeMathType is 1 (ALLOW_FP32_DOWN_PRECISION) and the input is FLOAT, the input is allowed to be converted to HFLOAT32 for computation.</li>
          <li>When cubeMathType is 2 (USE_FP16), this option is not supported when the input is BFLOAT16.</li>
          <li>When cubeMathType is 3 (USE_HF32) and the input is FLOAT, the input is converted to HFLOAT32 for computation.</li>
        </ul>
      </td>
  </tr>
  <tr>
    <th scope="row">kernelSize constraint</th>
    <td colspan="2">      The value of kernelSize is the size of the H and W dimensions of weight.
    </td>
  </tr>
  <tr>
    <th scope="row">Other constraints</th>
    <td colspan="2">      For self, weight, and bias, each dimension size of each tensor must not exceed 1000000.
    </td>
  </tr>
  </tbody>
</table>

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                    \
    if (!(cond)) {                        \
      Finalize(deviceId, stream);         \
      return_expr;                        \
    }                                     \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i: shape) {
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
  // Call aclrtMalloc to apply for device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvDepthwise2dTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction according to the API.
  std::vector<int64_t> shapeSelf = {2, 2, 2, 2};
  std::vector<int64_t> shapeWeight = {2, 1, 1, 1};
  std::vector<int64_t> shapeBias = {2};
  std::vector<int64_t> shapeResult = {2, 2, 2, 2};

  void* deviceDataSelf = nullptr;
  void* deviceDataWeight = nullptr;
  void* deviceDataBias = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* result = nullptr;
  aclIntArray* kernelSize = nullptr;
  aclIntArray* stride = nullptr;
  aclIntArray* padding = nullptr;
  aclIntArray* dilation = nullptr;

  std::vector<float> selfData(GetShapeSize(shapeSelf), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> outData(GetShapeSize(shapeResult), 1);
  std::vector<int64_t> kernelSizeData = {1, 1};
  std::vector<int64_t> strideData = {1, 1};
  std::vector<int64_t> paddingData = {0, 0};
  std::vector<int64_t> dilationData = {1, 1};

  // Create self aclTensor.
  ret = CreateAclTensor(selfData, shapeSelf, &deviceDataSelf, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataSelfPtr(deviceDataSelf, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataWeight, aclDataType::ACL_FLOAT, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataWeightPtr(deviceDataWeight, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create bias aclTensor.
  ret = CreateAclTensor(biasData, shapeBias, &deviceDataBias, aclDataType::ACL_FLOAT, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBiasPtr(deviceDataBias, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create out aclTensor.
  ret = CreateAclTensor(outData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> kernelSizePtr(kernelSize, aclDestroyIntArray);
  CHECK_FREE_RET(kernelSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  stride = aclCreateIntArray(strideData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridePtr(stride, aclDestroyIntArray);
  CHECK_FREE_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  padding = aclCreateIntArray(paddingData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> paddingPtr(padding, aclDestroyIntArray);
  CHECK_FREE_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  dilation = aclCreateIntArray(dilationData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationPtr(dilation, aclDestroyIntArray);
  CHECK_FREE_RET(dilation != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Modify it to the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConvDepthwise2d.
  ret = aclnnConvDepthwise2dGetWorkspaceSize(self, weight, kernelSize, bias, stride, padding, dilation, result, 1,
                                             &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnConvDepthwise2d.
  ret = aclnnConvDepthwise2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2d failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from device memory to host memory, and modify it according to the specific API definition.
  auto size = GetShapeSize(shapeResult);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (boilerplate) Initialize device/stream, referring to the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvDepthwise2dTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2dTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
