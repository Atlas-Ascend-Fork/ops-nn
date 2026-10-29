# aclnnQuantConvolution

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T17:46:16.680Z -->

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

- Description: Performs 2D and 3D convolution computation with per-channel quantization. The convolution process is the same as that of the aclnnConvolution API.

- Formula:
  Assume the input shape is $(N, C_{\text{in}}, D, H, W)$, the weight shape is $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$, the scale shape is $(C_{\text{out}})$, the bias shape is $C_{\text{out}}$, and the output shape is $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$, where $N$ indicates the batch size, $C$ is the number of channels, $D$, $H$, and $W$ are the depth, height and width of a sample, respectively, and $K_d$, $K_h$, and $K_w$ are the depth, height and width of the convolution kernel, respectively. The output is then expressed as:

  $$
  \text{output}(N_i, C_{\text{out}_j}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) = \left[\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)\right] \times \text{scale}(C_{\text{out}_j}) + \text{bias}(C_{\text{out}_j})
  $$

  where $\star$ indicates convolution, which depends on the dimensions of the convolution input and the convolution type (dilated convolution (`dilations` > 1), grouped convolution (`groups` > 1)). $N$ represents the batch size, $C$ represents the number of channels, and $D$, $H$, and $W$ represent depth, height, and width, respectively. The output dimensions are calculated as follows:

  $$
  D_{\text{out}}=[(D + 2 \times padding[0] - dilation[0] \times (K_d - 1) - 1 ) / stride[0]] + 1 \\
  H_{\text{out}}=[(H + 2 \times padding[1] - dilation[1] \times (K_h - 1) - 1 ) / stride[1]] + 1 \\
  W_{\text{out}}=[(W + 2 \times padding[2] - dilation[2] \times (K_w - 1) - 1 ) / stride[2]] + 1
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the aclnnQuantConvolutionGetWorkspaceSize API first to obtain the workspace size required for the computation and the executor containing the operator computation process. Then call the aclnnQuantConvolution API to perform the computation.

```cpp
aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(
    const aclTensor       *input,
    const aclTensor       *weight,
    const aclTensor       *bias,
    const aclTensor       *scale,
    const aclTensor       *offset,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    bool                   transposed,
    const aclIntArray     *outputPadding,
    int64_t                groups,
    int32_t                offsetx,
    const char            *roundMode,
    aclTensor             *output,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnQuantConvolution(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnQuantConvolutionGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:220px">Parameter</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:300px">Description</th>
  <th style="width:400px">Instruction</th>
  <th style="width:212px">Data Type</th>
  <th style="width:120px">Data Format</th>
  <th style="width:145px">Dimension (shape)</th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>input (aclTensor*)</td>
  <td>Input</td>
  <td>input in the formula, which indicates convolution input.</td>
  <td><ul><li>Null tensors are not supported.</li><li>The data type and the data type of weight must satisfy the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a>).</li><li>The dimensions of input, weight, and output must be the same.</li><li>N≥0, C≥1, D≥0, H≥0, W≥0.</li></ul></td>
  <td>INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
  <td>NCHW, NCDHW</td>
  <td>4-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight (aclTensor*)</td>
  <td>Input</td>
  <td>weight in the formula, which indicates convolution weight.</td>
  <td><ul><li>Does not support null tensors.</li><li>The data type and the data type of input must meet the data type inference rules (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a>).</li><li>The C dimension of its shape must be the same as the C dimension of input.</li><li>All dimensions are greater than or equal to 1.</li></ul></td>
  <td>INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
  <td>NCHW, NCDHW</td>
  <td>4-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias (aclTensor*)</td>
  <td>Input</td>
  <td>bias in the formula, which indicates the convolution bias.</td>
  <td>1D and equal to the first dimension of weight.</td>
  <td>BFLOAT16, FLOAT16, FLOAT, INT32</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>scale (aclTensor*)</td>
  <td>Input</td>
  <td>Scale in the formula, indicating quantization parameters.</td>
  <td>1D and equal to the first dimension of weight.</td>
  <td>FLOAT, INT64, UINT64</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>offset (aclTensor*)</td>
  <td>Input</td>
  <td>Reserved quantization parameter.</td>
  <td>Currently not yet supported. Pass in a null pointer nullptr.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>stride (aclIntArray*)</td>
  <td>Input</td>
  <td>Convolution scan stride.</td>
  <td><ul><li>The array length is 2 in 2D scenarios and 3 in 3D scenarios.</li><li>strideH and strideW should be within the range of [1,63].</li><li>In conv3d scenarios, strideD should be within the range of [1,1000000].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding (aclIntArray*)</td>
  <td>Input</td>
  <td>Padding of input.</td>
  <td><ul><li>The value should be ≥ 0.</li><li>paddingH and paddingW should be within the range [0,255].</li><li>In the conv3d scenario, paddingD should be within the range [0,1000000].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation (aclIntArray*)</td>
  <td>Input</td>
  <td>Spacing between the elements in the kernel.</td>
  <td><ul><li>The array length is 2 in the 2D scenario and 3 in the 3D scenario.</li><li>The value should be > 0.</li><li>dilationH and dilationW should be within the range [1,255].</li><li>In the conv3d scenario, dilationD should be within the range [1,1000000].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>transposed (bool)</td>
  <td>Input</td>
  <td>Reserved parameter. Indicates whether this is transposed quantized convolution.</td>
  <td>Currently not yet supported. Pass in false.</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>outputPadding (aclIntArray*)</td>
  <td>Input</td>
  <td>Reserved parameter. Indicates padding on all sides of the output in the transposed convolution case.</td>
  <td>In the case of non-transposed convolution, this attribute configuration is ignored. Currently not yet supported; passing a null pointer nullptr can be used.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups (int64_t)</td>
  <td>Input</td>
  <td>Indicates the number of blocked connections from input channels to output channels.</td>
  <td>Value ≥ 1, and the C dimension of groups*weight = the C dimension of input.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>offsetx (int32_t)</td>
  <td>Input</td>
  <td>Indicates the quantization factor.</td>
  <td>[-128,127] or 0.</td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>roundMode (char*)</td>
  <td>Input</td>
  <td>Indicates the rounding mode.</td>
  <td>rint, round, or nullptr.</td>
  <td>CHAR*</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>output (aclTensor*)</td>
  <td>Output.</td>
  <td>output in the formula, indicates convolution output.</td>
  <td><ul><li>Does not support null tensor output.</li><li>Its shape complies with the convolution inference rule.</li><li>The number of channels equals the first dimension of weight, and other dimensions are ≥ 0.</li></ul></td>
  <td>BFLOAT16, FLOAT16, FLOAT, FLOAT8_E4M3FN, HIFLOAT8</td>
  <td>NCHW, NCDHW</td>
  <td>4-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*)</td>
  <td>Output</td>
  <td>Returns the size of the workspace to be applied on the Device side.</td>
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
  <td>Return value</td>
  <td>Error code</td>
  <td>Description</td>
  </tr></thead>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The pointer-type parameter passed in is a null pointer.</td>
  </tr>
  <tr>
  <td rowspan="10" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="10" align="left">161002</td>
  <td align="left">The data types and data formats of input, weight, bias, scale, offset, and output are not within the supported range.</td>
  </tr>
  <tr><td align="left">The shapes of stride, padding, and dilation are incorrect.</td></tr>
  <tr><td align="left">The groups input is incorrect.</td></tr>
  <tr><td align="left">The shapes of scale and bias are incorrect.</td></tr>
  <tr><td align="left">The output shape does not match the infershape result.</td></tr>
  <tr><td align="left">Tensors with any dimension of zero do not meet the requirement.</td></tr>
  <tr><td align="left">The spatial size of input after padding is smaller than that of weight (expanded by dilation if dilation > 1).</td></tr>
  <tr><td align="left">The number of channels of weight and input does not meet the requirement.</td></tr>
  <tr><td align="left">stride or dilation less than 0 does not meet the requirement.</td></tr>
  <tr><td align="left">The current processor does not support convolution.</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">Internal API verification error, usually caused by input data or attributes out of the supported range.</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">An exception occurs when the API calls the NPU Runtime API, for example, the SocVersion is not supported.</td>
  </tr>
  </table>

## aclnnQuantConvolution

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
  <td>Workspace memory address applied on the Device side.</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Input</td>
  <td>Workspace size applied on the Device side, obtained by calling the first-phase API aclnnQuantConvolutionGetWorkspaceSize.</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>Input</td>
  <td>Op executor that contains the operator computation flow.</td>
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
  - aclnnQuantConvolution defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width:150px">
    <col style="width:700px">
    </colgroup>
   <thead>
    <tr>
     <th>Constraint Type</th>

     <th><term>Atlas A2 products</term>, <term>Atlas A3 products</term></th>

   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">input, weight</th>
     <td>        The data types of input and weight do not support FLOAT8_E4M3FN or HIFLOAT8, and the data format does not support NCHW.
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>          The data type of bias does not support INT32, and it is converted to FLOAT for computation.
     </td>
   </tr>
   <tr>
     <th scope="row">scale</th>
     <td>          The data type of scale does not support INT64 or UINT64.
     </td>
   </tr>
   <tr>
     <th scope="row">padding</th>
     <td>          The array length of padding must be 3.
     </td>
   </tr>
   <tr>
     <th scope="row">groups</th>
     <td>          The value of groups must be 1.
     </td>
   </tr>
   <tr>
     <th scope="row">offsetx</th>
     <td>          offsetx is currently not yet supported. Pass in the value 0.
     </td>
   </tr>
   <tr>
     <th scope="row">roundMode</th>
     <td>          roundMode is currently not yet supported. Passing a null pointer (nullptr) is required.
     </td>
   </tr>
   <tr>
     <th scope="row">output</th>
     <td>          output supports the BFLOAT16 and FLOAT16 data types. Only the NCDHW data format is supported.
     </td>
   </tr>
   <tr>
     <th scope="row">Other Constraints</th>
     <td>        <ul>
          <li>The operator can be called only in the inference scenario.</li>
          <li>Only forward 3D convolution is supported.</li>
          <li>For each group of tensors in input, weight, bias, and scale, the size of each dimension should be less than 1000000.</li>
        </ul>
     </td>
   </tr>
   </tbody>
  </table>

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

Refer to different main functions for different product models.

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_convolution.h"

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
  // Boilerplate, resource initialization.
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
  // Call aclrtMalloc to apply for device-side memory.
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorND(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for device-side memory.
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

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnQuantConvolutionTest(int32_t deviceId, aclrtStream& stream, std::vector<aclDataType> dtypesInfo)
{
  auto ret = Init(deviceId, &stream);
  // Handle the check based on your requirements.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output. Customize the construction based on the API.
  std::vector<int64_t> shapeInput = {2, 2, 32, 32, 32};
  std::vector<int64_t> shapeWeight = {2, 2, 3, 3, 3};
  std::vector<int64_t> shapeScale = {2};
  std::vector<int64_t> shapeBias = {2};
  std::vector<int64_t> shapeResult = {2, 2, 32, 32, 32};
  std::vector<int64_t> convStrides;
  std::vector<int64_t> convPads;
  std::vector<int64_t> convDilations;

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataScale = nullptr;
  void* deviceDataBias = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* scale= nullptr;
  aclTensor* bias= nullptr;
  aclTensor* result = nullptr;
  std::vector<int8_t> inputData(GetShapeSize(shapeInput), 1);
  std::vector<int8_t> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> scaleData(GetShapeSize(shapeScale), 1);
  std::vector<uint16_t> outputData(GetShapeSize(shapeResult), 1);
  convStrides = {1, 1, 1};
  convPads = {1, 1, 1};
  convDilations = {1, 1, 1};
  aclDataType inputDtype = dtypesInfo[0];
  aclDataType weightDtype = dtypesInfo[1];
  aclDataType biasDtype = dtypesInfo[2];
  aclDataType scaleDtype = dtypesInfo[3];
  aclDataType outputDtype = dtypesInfo[4];
  // Create input aclTensor.
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, inputDtype, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, weightDtype, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create scale.
  ret = CreateAclTensorND(scaleData, shapeScale, &deviceDataScale, scaleDtype, &scale);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataScalePtr(deviceDataScale, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create bias.
  ret = CreateAclTensorND(biasData, shapeBias, &deviceDataBias, biasDtype, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBiasPtr(deviceDataBias, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create out aclTensor.
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, outputDtype, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Change it to the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConvolution.
  ret = aclnnQuantConvolutionGetWorkspaceSize(input, weight, bias, scale, nullptr, strides, pads, dilations,
                                              false, nullptr, 1, 0, nullptr, result, &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnConvolution.
  ret = aclnnQuantConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to the host side. Modify this part based on the definition of the specific API.
  auto size = GetShapeSize(shapeResult);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  std::vector<aclDataType> dtypesInfo = {aclDataType::ACL_INT8, aclDataType::ACL_INT8, aclDataType::ACL_FLOAT,
  aclDataType::ACL_FLOAT, aclDataType::ACL_BF16}; // datatypes of input/weight/bias/scale/output, respectively
  auto ret = aclnnQuantConvolutionTest(deviceId, stream, dtypesInfo);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
