# aclnnConvTbc

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T17:38:13.517Z -->

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

- Description: Implements one-dimensional convolution with input and output dimensions of **T** (time or spatial dimension), **B** (batch), and **C** (channel).
- Formula:
  Assume that the input (self) shape is ($H_{\text{in}},N,C_{\text{in}}$) and the output (output) shape is ($H_{\text{out}},N,C_{\text{out}}$). The output is represented as:

  $$
  output_{N_i,C_{out j}} = bias(C_{out j}) + \sum_{k = 0}^{C_{in} - 1} weight(k, C_{out j}) \cdot self(N_i, k)
  $$

  Where $N$ represents the batch size, $C$ represents the number of channels, and $H$ represents the time or spatial dimension.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call aclnnConvTbcGetWorkspaceSize first to obtain the workspace size required for the computation and the executor containing the computation process of the operator, and then call aclnnConvTbc to perform the computation.

```cpp
aclnnStatus aclnnConvTbcGetWorkspaceSize(
    const aclTensor       *self,
    const aclTensor       *weight,
    const aclTensor       *bias,
    const int64_t          pad,
    aclTensor             *output,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvTbc(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnConvTbcGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:220px">Parameter</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:300px">Description</th>
  <th style="width:400px">Instruction</th>
  <th style="width:212px">Data Type</th>
  <th style="width:100px">Data Format</th>
  <th style="width:100px">Dimension (shape)</th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>self (aclTensor*)</td>
  <td>Input</td>
  <td>self in the formula, which denotes the convolution input.</td>
  <td><ul>
  <li>Supports empty tensor.</li>
  <li>The data type must satisfy the data type deduction rule with the data type of weight (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a>).</li>
  <li>The shape is (H<sub>in</sub>,N,C<sub>in</sub>).</li>
  <li>H≥0, N≥0, C<sub>in</sub>≥1.</li>
  </ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>ND, NCL</td>
  <td>3</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight (aclTensor*)</td>
  <td>Input</td>
  <td>weight in the formula, which denotes the convolution weight.</td>
  <td><ul>
  <li>Supports empty tensors.</li>
  <li>The data type must satisfy the data type deduction rules with the data type of self. (See <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a>.)</li>
  <li>The shape is (K,C<sub>in</sub>,C<sub>out</sub>).</li>
  <li>All dimensions ≥1.</li>
  </ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>ND, NCL</td>
  <td>3</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias (aclTensor*)</td>
  <td>Input</td>
  <td>bias in the formula, denoting the convolution bias.</td>
  <td><ul>
  <li>The data type must be the same as those of self and weight.</li>
  <li>One-dimensional and equal to the first dimension of weight. Passing a null pointer is not allowed.</li>
  </ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>pad (int64_t)</td>
  <td>Input</td>
  <td>Denotes the number of paddings on the left and right of the H dimension.</td>
  <td>The size must be within the range of [0,255].</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>output (aclTensor*)</td>
  <td>Output</td>
  <td>Denotes convolution output.</td>
  <td><ul>
  <li>Supports empty tensor.</li>
  <li>Data type is the same as that of self.</li>
  <li>The shape is (H<sub>out</sub>,N,C<sub>out</sub>).</li>
  <li>The number of channels equals the first dimension of weight, and other dimensions are ≥0.</li>
  </ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>ND, NCL</td>
  <td>3</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>cubeMathType (int8_t)</td>
  <td>Input</td>
  <td>Determines which computation logic the Cube unit uses for operations.</td>
  <td>  <ul><li> 0 (KEEP_DTYPE): Retains the input data type for computation.</li></ul>
  <ul><li> 1 (ALLOW_FP32_DOWN_PRECISION): Allows FLOAT computation with reduced precision to improve performance.</li></ul>
  <ul><li> 2 (USE_FP16): Performs computation with FLOAT16 precision.</li></ul>
  <ul><li> 3 (USE_HF32): Performs computation with HFLOAT32 (mixed precision).</li></ul>
  </td>
  <td>INT8</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*)</td>
  <td>Output</td>
  <td>Returns the size of the workspace applied on the Device side.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor (aclOpExecutor**)</td>
  <td>Output</td>
  <td>Returns the op executor, which contains the operator computation flow.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
  <td>Description</td>
  </tr></thead>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The pointer-type input parameter is a null pointer.</td>
  </tr>
  <tr>
  <td rowspan="9" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="9" align="left">161002</td>
  <td align="left">The data types and formats of self, weight, bias, and output are not within the supported range.</td>
  </tr>
  <tr><td align="left">The data types of self, weight, and output are inconsistent.</td></tr>
  <tr><td align="left">The shape of output does not match the infershape result.</td></tr>
  <tr><td align="left">The shape of output contains a value less than 0.</td></tr>
  <tr><td align="left">The dim of self is not 3.</td></tr>
  <tr><td align="left">The dim of weight is not 3.</td></tr>
  <tr><td align="left">The dim of bias is not 1.</td></tr>
  <tr><td align="left">The second dimension of self is not equal to that of weight.</td></tr>
  <tr><td align="left">The value of bias is not equal to the first dimension of weight.</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">API internal verification error, usually caused by input data or attribute specifications not within the supported range.</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">An exception occurs when the API calls the NPU runtime interface, for example, unsupported SocVersion.</td>
  </tr>
  </table>

## aclnnConvTbc

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
  <td>Workspace memory address applied on the side of the Device.</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Input</td>
  <td>Workspace size applied on the side of the Device, obtained by the first-phase API aclnnConvTbcGetWorkspaceSize.</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>Input</td>
  <td>op executor containing the operator computation flow.</td>
  </tr>
  <tr>
  <td>stream</td>
  <td>Input</td>
  <td>Stream for executing tasks.</td>
  </tr>
  </table>

- **Return Value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation
  - aclnnConvTbc defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
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
          <li>The data types of self and weight do not support HIFLOAT8. N, C, and L dimensions greater than or equal to 0 are supported.</li>
          <li>weight supports N and C dimensions greater than or equal to 0.</li>
        </ul>
     </td>
     <td>        <ul>
           <li>self and weight support N and C dimensions greater than or equal to 0, and support an L dimension greater than or equal to 0 (the scenario where the dimension is equal to 0 is supported only when the derived L dimension of output is also equal to 0).</li>
           <li>weight supports an N dimension greater than or equal to 0 (the scenario where the dimension is equal to 0 is supported only when the N dimension of bias and the C dimension of output are also equal to 0). The supported C dimension size is the same as that of self, and the L dimension size should be within the range of [1,255].</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>          The data format is not strictly validated. Formats such as ND and NCL can be passed in, but the shape and dimension sizes must still meet the constraints.
     </td>
     <td>          When the data type of self is HIFLOAT8, the data type of bias is finally converted to FLOAT for computation.
     </td>
   </tr>
   <tr>
     <th scope="row">output</th>
     <td>        output supports N, C, and L dimensions greater than or equal to 0 (the scenario where the dimension is equal to 0 is supported only when the N, C, or L dimension of self is equal to 0).
     </td>
     <td>        <ul>
          <li>Supports an N dimension greater than or equal to 0 and a C dimension greater than or equal to 0 (the scenario where the dimension is equal to 0 is supported only when the N dimension of weight is equal to 0).</li>
          <li>Supports an L dimension greater than or equal to 0 (the scenario where the dimension is equal to 0 is supported only when the L dimension of self is equal to 0).</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType</th>
     <td>        <ul>
          <li>If the value is 0 (KEEP_DTYPE), this option is temporarily not supported when the input is FLOAT.</li>
          <li>If the value is 1 (ALLOW_FP32_DOWN_PRECISION), when the input is FLOAT, it is allowed to be converted to HFLOAT32 for computation.</li>
          <li>If the value is 2 (USE_FP16), this option is not supported when the input is BFLOAT16.</li>
          <li>If the value is 3 (USE_HF32), when the input is FLOAT, it is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
     <td>        <ul>
          <li>If the value is 1 (ALLOW_FP32_DOWN_PRECISION), when the input is FLOAT, it is allowed to be converted to HFLOAT32 for computation.</li>
          <li>If the value is 2 (USE_FP16), this option is not supported when the input is BFLOAT16.</li>
          <li>If the value is 3 (USE_HF32), when the input is FLOAT, it is converted to HFLOAT32 for computation.</li>
        </ul>
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
                    aclDataType dataType, aclTensor** tensor, aclFormat dataFormat) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvTbcTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> shapeInput = {2, 2, 3};
  std::vector<int64_t> shapeWeight = {3, 3, 4};
  std::vector<int64_t> shapeBias = {4};
  std::vector<int64_t> shapeResult = {2, 2, 4};

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataC = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* result = nullptr;
  std::vector<float> inputData(GetShapeSize(shapeInput), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> outputData(GetShapeSize(shapeResult), 1);

  // Create the input aclTensor.
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, aclDataType::ACL_FLOAT, &input, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create the weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, aclDataType::ACL_FLOAT, &weight, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create the bias aclTensor.
  ret = CreateAclTensor(biasData, shapeBias, &deviceDataC, aclDataType::ACL_FLOAT, &bias, aclFormat::ACL_FORMAT_ND);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataCPtr(deviceDataC, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create the result aclTensor.
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnConvTbc first-phase API.
  ret = aclnnConvTbcGetWorkspaceSize(input, weight, bias, 1, result, 1, &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the aclnnConvTbc second-phase API.
  ret = aclnnConvTbc(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbc failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device memory to the host side, and modify it as required by the specific API definition.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvTbcTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
