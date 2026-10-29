# aclnnQuantConvolutionWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T17:55:20.180Z -->

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

- Description: Performs 3D convolution computation with per-channel quantization. Only the FRACTAL_Z_3D format is supported for weight. The convolution computation process is the same as that of the aclnnConvolution API.

- Formula:

  Assume that the shape of input is $(N, C_{\text{in}}, D, H, W)$, the original shape of weight is $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$, the shape of scale is $(C_{\text{out}})$, the shape of bias is $C_{\text{out}}$, and the shape of output is $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$, where $N$ represents the batch size, $C$ is the number of channels, $D$, $H$, and $W$ are the depth, height, and width of a sample, respectively, and $K_d$, $K_h$, and $K_w$ are the depth, height, and width of the convolution kernel, respectively. The output is expressed as follows:

  $$
  \text{output}(N_i, C_{\text{out}_j}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) = \left[\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)\right] \times \text{scale}(C_{\text{out}_j}) + \text{bias}(C_{\text{out}_j})
  $$

  Where $\star$ represents convolution. The type of convolution (dilated convolution (`dilations` > 1), grouped convolution (`groups` > 1)) depends on the dimensions of the convolution input. $N$ represents the batch size, $C$ represents the number of channels, and $D$, $H$, and $W$ represent depth, height, and width, respectively. The formulas for calculating the corresponding output dimensions are as follows:

  $$
  D_{\text{out}}=[(D + 2 \times padding[0] - dilation[0] \times (K_d - 1) - 1 ) / stride[0]] + 1 \\
  H_{\text{out}}=[(H + 2 \times padding[1] - dilation[1] \times (K_h - 1) - 1 ) / stride[1]] + 1 \\
  W_{\text{out}}=[(W + 2 \times padding[2] - dilation[2] \times (K_w - 1) - 1 ) / stride[2]] + 1
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call aclnnQuantConvolutionWeightNzGetWorkspaceSize first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call aclnnQuantConvolutionWeightNz to perform the computation.

```cpp
aclnnStatus aclnnQuantConvolutionWeightNzGetWorkspaceSize(
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
aclnnStatus aclnnQuantConvolutionWeightNz(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnQuantConvolutionWeightNzGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:170px">Parameter</th>
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
  <td>input in the formula, which represents convolution input.</td>
  <td><ul><li>The dimensions of input, the original shape of weight, and output must be the same.</li><li>Empty tensors are not supported.</li><li>The data type of input and the data type of weight must satisfy the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a>).</li><li>N≥0, C≥1, D≥0, H≥0, W≥0.</li></ul></td>
  <td>INT8</td>
  <td>NCDHW</td>
  <td>5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight (aclTensor*)</td>
  <td>Input</td>
  <td>weight in the formula, which represents convolution weight.</td>
  <td><ul><li>The C dimension of its original shape must be the same as that of input.</li><li>Does not support null tensors.</li><li>The data type and the data type of input must satisfy the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a>).</li><li>FRACTAL_Z_3D is the 3D convolution weight format. The dimensions of the format are (D * C1 * H * W, N1, N0, C0), where D, H, and W are the NCDHW dimensions before transdata. N0=16, C0=32, and N1 and N satisfy the relationship N1=CEIL(N,N0).</li><li>All dimensions are ≥1.</li></ul></td>
  <td>INT8</td>
  <td>Fractal_Z_3D</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias (aclTensor*)</td>
  <td>Input</td>
  <td>bias in the formula, which represents the convolution bias.</td>
  <td>One-dimensional, and equal to the first dimension of the original shape of weight.</td>
  <td>BFLOAT16, FLOAT16, FLOAT</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>scale (aclTensor*)</td>
  <td>Input</td>
  <td>scale in the formula, representing the quantization parameter.</td>
  <td>One dimension, equal to the first dimension of the original shape of weight.</td>
  <td>FLOAT</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>offset (aclTensor*)</td>
  <td>Input</td>
  <td>Reserved quantization parameter.</td>
  <td>Currently not supported; passing a null pointer (nullptr) is sufficient.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>stride (aclIntArray*)</td>
  <td>Input</td>
  <td>Convolution scan stride.</td>
  <td><ul><li>Array length = 3.</li><li>strideH and strideW should be in the range [1,63].</li><li>In conv3d use cases, strideD should be in the range [1,1000000].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding (aclIntArray*)</td>
  <td>Input</td>
  <td>Padding of input.</td>
  <td><ul><li>Value should be ≥0.</li><li>paddingH and paddingW should be in the range [0,255].</li><li>In conv3d use cases, paddingD should be in the range [0,1000000].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation(aclIntArray*)</td>
  <td>Input</td>
  <td>Spacing between elements in the convolution kernel.</td>
  <td><ul><li>Value should be >0.</li><li>dilationH and dilationW should be in the range [1,255].</li><li>In conv3d use cases, dilationD should be in the range [1,1000000].</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>transposed (bool)</td>
  <td>Input</td>
  <td>Reserved parameter. Indicates whether it is a transposed quantized convolution.</td>
  <td>Currently not supported. Pass in false.</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>outputPadding (aclIntArray*)</td>
  <td>Input</td>
  <td>Reserved parameter. Represents the padding on all edges of the output in the case of transposed convolution.</td>
  <td>In the case of non-transposed convolution, this attribute is ignored. Currently not supported. Passing a null pointer nullptr is sufficient.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups (int64_t)</td>
  <td>Input</td>
  <td>Represents the number of block links from the input channel to the output channel.</td>
  <td>Value=1</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>offsetx (int32_t)</td>
  <td>Input</td>
  <td>Represents the quantization factor.</td>
  <td>[-128,127] or 0</td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>roundMode (char*)</td>
  <td>Input</td>
  <td>Represents the rounding mode.</td>
  <td>rint, round, or nullptr</td>
  <td>CHAR*</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>output (aclTensor*)</td>
  <td>Output</td>
  <td>output in the formula, which represents the convolution output.</td>
  <td><ul><li>The shape complies with the convolution deduction rules.</li><li>Does not support null Tensor output.</li><li>The number of channels equals the first dimension of the original shape of weight, and other dimensions are ≥ 0.</li></ul></td>
  <td>BFLOAT16, FLOAT16</td>
  <td>NCDHW</td>
  <td>5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*)</td>
  <td>Output</td>
  <td>Returns the size of the workspace applied by the Device side.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor (aclOpExecutor**)</td>
  <td>Output</td>
  <td>Returns the op executor, containing the operator computation flow.</td>
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
    <td align="left">The passed-in parameter of pointer type is a null pointer.</td>
    </tr>
    <tr>
    <td rowspan="11" align="left">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="11" align="left">161002</td>
    <td align="left">The data types and data formats of input, weight, bias, scale, offset, and output are not in the supported range.</td>
    </tr>
    <tr><td align="left">The input shapes of stride, padding, and dilation are incorrect.</td></tr>
    <tr><td align="left">The input of groups is incorrect.</td></tr>
    <tr><td align="left">The input shapes of scale and bias are incorrect.</td></tr>
    <tr><td align="left">The shape of output does not match the infershape result.</td></tr>
    <tr><td align="left">Tensors with any dimension set to 0 do not meet the requirements.</td></tr>
    <tr><td align="left">The spatial size of input after padding is smaller than that of weight (after dilation expansion, if dilation > 1).</td></tr>
    <tr><td align="left">The number of channels of weight and input does not meet the requirements.</td></tr>
    <tr><td align="left">The shapes of weight before and after conversion do not match the conversion relationship.</td></tr>
    <tr><td align="left">Values of stride and dilation less than 0 do not meet the requirements.</td></tr>
    <tr><td align="left">The current processor does not support convolution.</td></tr>
    <tr>
    <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
    <td align="left">561103</td>
    <td align="left">API internal check error, usually caused by input data or attributes out of the supported range.</td>
    </tr>
    <tr>
    <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
    <td align="left">361001</td>
    <td align="left">An exception occurs when the API calls the NPU runtime API, for example, the SocVersion is not supported.</td>
    </tr>
    </table>

## aclnnQuantConvolutionWeightNz

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
  <td>Workspace memory address applied by the Device side.</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Input</td>
  <td>Workspace size applied by the Device side, obtained by calling the first-phase API aclnnQuantConvolutionWeightNzGetWorkspaceSize.</td>
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
  - aclnnQuantConvolutionWeightNz defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 850px"><colgroup>
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
     <td>        -
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>          -
     </td>
   </tr>
   <tr>
     <th scope="row">scale</th>
     <td>          -
     </td>
   </tr>
   <tr>
     <th scope="row">padding</th>
     <td>          The length of the padding array must be 3.
     </td>
   </tr>
   <tr>
     <th scope="row">groups</th>
     <td>          The groups value must be 1.
     </td>
   </tr>
   <tr>
     <th scope="row">offsetx</th>
     <td>          offsetx is currently not supported. Pass in 0.
     </td>
   </tr>
   <tr>
     <th scope="row">roundMode</th>
     <td>          roundMode is currently not supported. Pass in a null pointer nullptr.
     </td>
   </tr>
   <tr>
     <th scope="row">output</th>
     <td>          -
     </td>
   </tr>
   <tr>
     <th scope="row">Other Constraints</th>
     <td>        <ul>
          <li>The operator can be called only in inference use cases.</li>
          <li>Only forward 3D convolution is supported.</li>
          <li>The size of each dimension of each group of tensors in input, the original shape of weight, bias, and scale should be less than 1000000.</li>
        </ul>
     </td>
   </tr>
   </tbody>
  </table>

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

For different product models, use different main functions.

The sample constructs the weight in FRACTAL_Z_3D format through the view/storage format setting APIs of aclTensor. It depends on the header file "aclnn/opdev/common_types.h", and the C++ standard should be C++14 or later during compilation (for example, add_compile_options(-std=c++17)).

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/common_types.h"
#include "aclnnop/aclnn_npu_format_cast.h"
#include "aclnnop/aclnn_quant_convolution.h"

// The macro with the same name is already defined in aclnn/opdev/common_types.h. Undefine it first to avoid macro redefinition warnings.
#undef CHECK_RET
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
  // Boilerplate. Resource initialization
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
  // Call aclrtMalloc to allocate device-side memory.
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

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                              aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                format, *storageShape, *storageShapeSize, *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorND(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory.
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

int aclnnQuantConvolutionWeightNzTest(int32_t deviceId, aclrtStream& stream, std::vector<aclDataType> dtypesInfo)
{
  auto ret = Init(deviceId, &stream);
  // Handle the check based on your needs.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction based on the API.
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
  void* deviceDataBNz = nullptr;
  void* deviceDataScale = nullptr;
  void* deviceDataBias = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* weightNz = nullptr;
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
  // Create the input aclTensor.
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, inputDtype, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create the weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, weightDtype, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the scale.
  ret = CreateAclTensorND(scaleData, shapeScale, &deviceDataScale, scaleDtype, &scale);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataScalePtr(deviceDataScale, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the bias.
  ret = CreateAclTensorND(biasData, shapeBias, &deviceDataBias, biasDtype, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBiasPtr(deviceDataBias, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create the out aclTensor.
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

  // 3. Call the CANN operator library API, and replace it with the specific API.
  aclOpExecutor* executor;
  void* workspaceAddrNz = nullptr;

  int64_t* dstShape = nullptr;
  uint64_t dstShapeSize = 0;
  int actualFormat;
  // Convert weight to FRACTAL_Z_3D.
  // Calculate the shape and format of the target tensor.
  ret = aclnnNpuFormatCastCalculateSizeAndFormat(weight, aclFormat::ACL_FRACTAL_Z_3D, -1, &dstShape, &dstShapeSize, &actualFormat);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);
  // The current CANN version does not support converting NCDHW data directly to FRACTAL_Z_3D via aclnnNpuFormatCast,
  // rearrange the weight data on the host in the FRACTAL_Z_3D layout (D*C1*H*W, N1, N0, C0), and pad the alignment part with 0s,
  // where C1=CEIL(Cin, C0), N1=CEIL(Cout, N0), N0=16, C0=32.
  std::vector<int64_t> shapeWeightNz(dstShape, dstShape + dstShapeSize);
  std::vector<int8_t> weightNzData(GetShapeSize(shapeWeightNz), 0);
  const int64_t cOut = shapeWeight[0], cIn = shapeWeight[1];
  const int64_t kD = shapeWeight[2], kH = shapeWeight[3], kW = shapeWeight[4];
  const int64_t c1 = shapeWeightNz[0] / (kD * kH * kW);
  const int64_t n1 = shapeWeightNz[1], n0 = shapeWeightNz[2], c0 = shapeWeightNz[3];
  for (int64_t co = 0; co < cOut; co++) {
    for (int64_t ci = 0; ci < cIn; ci++) {
      for (int64_t d = 0; d < kD; d++) {
        for (int64_t h = 0; h < kH; h++) {
          for (int64_t w = 0; w < kW; w++) {
            int64_t dim0 = ((d * c1 + ci / c0) * kH + h) * kW + w;
            int64_t dstIdx = ((dim0 * n1 + co / n0) * n0 + co % n0) * c0 + ci % c0;
            weightNzData[dstIdx] = weightData[(((co * cIn + ci) * kD + d) * kH + h) * kW + w];
          }
        }
      }
    }
  }
  // The viewShape/originalShape of weightNz is 5D NCDHW (same dimensions as input/output), and the storageShape is 4D FRACTAL_Z_3D.
  ret = CreateAclTensorWithFormat(weightNzData, shapeWeight, &dstShape, &dstShapeSize, &deviceDataBNz, weightDtype, &weightNz, aclFormat::ACL_FRACTAL_Z_3D);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);
  // After the tensor is created with FRACTAL_Z_3D via aclCreateTensor, also set the view/original format and original shape to NCDHW so that the operator can correctly parse the N/C/D/H/W axis information of the weight.
  op::Shape weightNcdhwShape = {shapeWeight[0], shapeWeight[1], shapeWeight[2], shapeWeight[3], shapeWeight[4]};
  weightNz->SetViewFormat(op::Format::FORMAT_NCDHW);
  weightNz->SetOriginalFormat(op::Format::FORMAT_NCDHW);
  weightNz->SetOriginalShape(weightNcdhwShape);

  uint64_t workspaceSizeNz = 0;
  // Call the first-phase API of aclnnConvolutionWeightNz.
  ret = aclnnQuantConvolutionWeightNzGetWorkspaceSize(input, weightNz, bias, scale, nullptr, strides, pads, dilations,
                                              false, nullptr, 1, 0, nullptr, result, &workspaceSizeNz, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT(
    "aclnnQuantConvolutionWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.

  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSizeNz > 0) {
    ret = aclrtMalloc(&workspaceAddrNz, workspaceSizeNz, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddrNz);
  }
  // Call the second-phase API aclnnConvolutionWeightNz.
  ret = aclnnQuantConvolutionWeightNz(workspaceAddrNz, workspaceSizeNz, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionWeightNz failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device side to the host side, and modify it based on the specific API definition.
  auto size = GetShapeSize(shapeResult);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  int64_t printSize = size > 10 ? 10 : size;
  for (int64_t i = 0; i < printSize; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  std::vector<aclDataType> dtypesInfo = {aclDataType::ACL_INT8, aclDataType::ACL_INT8, aclDataType::ACL_FLOAT,
    aclDataType::ACL_FLOAT, aclDataType::ACL_BF16}; // Data types of input/weight/bias/scale/output, respectively.
  auto ret = aclnnQuantConvolutionWeightNzTest(deviceId, stream, dtypesInfo);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionWeightNzTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
