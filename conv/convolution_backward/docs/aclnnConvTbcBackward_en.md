# aclnnConvTbcBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T16:58:02.444Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Implements backward propagation for 1D convolution with input and output dimensions of **T** (time or spatial dimension), **B** (batch), and **C** (channel).

- Formula: Assume that the shape of the input $input$ of forward Conv_tbc is $(H_{\text{in}},N,C_{\text{in}})$, and the output gradient $gradOutput$
  has a shape of $(H_{\text{out}},N,C_{\text{out}})$, the convolution kernel $weight$ has a shape of $(K,C_{\text{in}},C_{\text{out}})$, the bias $bias$ has a shape of $(C_{\text{out}})$, and the padding for the input during backward propagation is $pad$. The relationship among these parameters is:

  $$
  H_{out} = {H_{in} + 2 \cdot pad - K} + 1
  $$

  Convolution backward propagation requires computing the gradients with respect to the convolution forward input tensor $x$ (corresponding to the input parameter in the function prototype), the convolution kernel weight tensor $w$
  (corresponding to the weight parameter in the function prototype), and the bias $b$ (corresponding to the bias parameter in the function prototype).

    - For the gradient of $x$, $\frac{\partial L}{\partial x}$ (corresponding to the gradInput parameter in the function prototype):

      $$
      \frac{\partial L}{\partial x_{t,b,c_{in}}} = \sum_{k=0}^{K-1} \sum_{c_{out}=0}^{C_{out}-1} \frac{\partial L}{\partial y_{t-k,b,c_{out}}} \cdot w_{k,c_{in},c_{out}}
      $$

      Where $N$ denotes the batch size, $C$ denotes the number of channels, $H$ denotes the temporal or spatial dimension, $L$
      denotes the loss function, and $\frac{\partial L}{\partial y}$ represents the gradient of the output tensor $y$ with respect to $L$ (corresponding to the self parameter in the function prototype).

    - For the gradient of $w$, $\frac{\partial L}{\partial w}$ (corresponding to the gradWeight parameter in the function prototype):

      $$
      \frac{\partial L}{\partial w_{k,c_{in},c_{out}}} = \sum_{b=0}^{N-1} \sum_{t=0}^{H_{out}-1} x_{t+k,b,c_{in}} \cdot \frac{\partial L}{\partial y_{t,b,c_{out}}}
      $$

    - For the gradient of $b$, $\frac{\partial L}{\partial b}$ (corresponding to the gradBias parameter in the function prototype):

      $$
      \frac{\partial L}{\partial b_{c_{out}}} = \sum_{b=0}^{N-1}\sum_{t=0}^{H_{\text{out}}-1} \frac{\partial L}{\partial y_{t,b,c_{out}}}
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnConvTbcBackwardGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call the "aclnnConvTbcBackward" API to perform the computation.

```cpp
aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *input,
    const aclTensor *weight,
    const aclTensor *bias,
    int64_t          pad,
    int8_t           cubeMathType,
    aclTensor       *gradInput,
    aclTensor       *gradWeight,
    aclTensor       *gradBias,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnConvTbcBackward(
    void                *workspace,
    uint64_t             workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream    stream)
```

## aclnnConvTbcBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width:170px">
    <col style="width:120px">
    <col style="width:300px">
    <col style="width:330px">
    <col style="width:212px">
    <col style="width:100px">
    <col style="width:190px">
    <col style="width:145px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
     <td>self</td>
     <td>Input</td>
     <td>Gradient of the output tensor y with respect to L in the formula. It is the input of the backward convolution.</td>
     <td>       <ul><li>Supports empty tensors.</li><li>The shape is (N,C<sub>out</sub>,H<sub>out</sub>).</li><li>The data type must meet the data type derivation rule together with the data type of weight (see also <a href="../../../docs/zh/context/deduction_relationship.md">deduction relationship</a>).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>Input</td>
     <td>Input</td>
     <td>x in the formula. It is the input of the forward convolution.</td>
     <td>       <ul>
        <li>Supports empty tensors.</li>
        <li>Shape is (N,C<sub>in</sub>,H<sub>in</sub>).</li>
        <li>The data type must satisfy the data type derivation rule with the data type of weight (see also <a href="../../../docs/zh/context/deduction_relationship.md">derivation relationship</a>).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>weight</td>
     <td>Input</td>
     <td>w in the formula, indicating the convolution weight.</td>
     <td>       <ul>
        <li>Supports empty tensors.</li>
        <li>Shape is (C<sub>out</sub>,C<sub>in</sub>,K).</li>
        <li>The data type must satisfy the data type derivation rule with the data types of input and self (see also <a href="../../../docs/zh/context/deduction_relationship.md">derivation relationship</a>).</li>
      </ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>bias</td>
     <td>Input</td>
     <td>b in the formula, indicating the convolution bias.</td>
     <td>       <ul>
        <li>Shape is (C<sub>out</sub>).</li>
        <li>One-dimensional and equal to the first dimension of weight. A null pointer cannot be passed.</li>
        <li>The data type is the same as that of self and weight.</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND</td>
     <td>1</td>
     <td>√</td>
    </tr>
    <tr>
     <td>pad</td>
     <td>Input</td>
     <td>Number of paddings added to the left and right of the H dimension of the input during backward propagation.</td>
     <td>       <ul><li>The value range is [0,2147483647].</li></ul>
     </td>
     <td>INT64</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>cubeMathType</td>
     <td>Input</td>
     <td>Determines which computation logic the Cube unit uses for the computation.</td>
     <td>       The supported enumeration values are as follows:
       <ul>
       <li>0: KEEP_DTYPE, keeping the data type of the input for computation.</li>
       <li>1: ALLOW_FP32_DOWN_PRECISION, allowing the input data to be computed at a lower precision.</li>
       <li>2: USE_FP16, allowing conversion to data type FLOAT16 for computation. When the input data type is FLOAT, it is converted to FLOAT16 for computation.</li>
       <li>3: USE_HF32, allowing conversion to data type HFLOAT32 for computation. When the input is FLOAT16, FLOAT16 is still used for computation.</li>
       </ul>
     </td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradInput</td>
     <td>Output</td>
     <td>The gradient of the input tensor x in the formula with respect to L.</td>
     <td>       <ul>
        <li>Empty tensors are supported.</li>
        <li>The data type is the same as that of input.</li>
        <li>The shape is (N,C<sub>in</sub>,H<sub>in</sub>).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradWeight</td>
     <td>Output</td>
     <td>Gradient of the convolution kernel weight tensor w with respect to L.</td>
     <td>       <ul><li>Empty tensors are supported.</li>
       <li>The data type is the same as that of weight.</li>
       <li>The shape is (C<sub>out</sub>,C<sub>in</sub>,K).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
     <td>ND, NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradBias</td>
     <td>Output</td>
     <td>Gradient of bias b with respect to L.</td>
     <td>      <ul><li>Supports empty tensors.</li>
      <li>Data type is consistent with the type of bias.</li>
      <li>Shape is (C<sub>out</sub>).</li></ul>
    </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>workspaceSize</td>
     <td>Output</td>
     <td>Returns the workspace size to be requested on the device.</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>executor</td>
     <td>Output</td>
     <td>Returns the op executor, which contains the operator computation flow.</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
    <thead>
     <tr>
       <th>Return Value</th>
       <th>Error Code</th>
       <th>Description</th>
     </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The pointer-type input parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data types and data formats of self, input, weight, bias, gradInput, gradWeight, and gradBias are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self, input, weight, and bias are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of gradInput, gradWeight, and gradBias do not match the infershape result.</td>
    </tr>
    <tr>
      <td>The shapes of gradInput, gradWeight, and gradBias contain values less than 0.</td>
    </tr>
    <tr>
      <td>The dim of self, input, or weight is not 3.</td>
    </tr>
    <tr>
      <td>The dim of bias is not 1.</td>
    </tr>
    <tr>
      <td>The value of the third dimension of input is not equal to the value of the second dimension of weight.</td>
    </tr>
    <tr>
      <td>The value of bias is not equal to the third dimension of weight.</td>
    </tr>
    <tr>
      <td>The pad value does not meet the requirements.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>Internal API verification error, usually caused by input data or attribute specifications out of the supported range.</td>
      </tr>
      </tbody>
  </table>

## aclnnConvTbcBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
       <col style="width:100px">
       <col style="width:100px">
       <col style="width:950px">
       </colgroup>
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
       <td>Workspace memory address applied for on the Device.</td>
     </tr>
     <tr>
       <td>workspaceSize</td>
       <td>Input</td>
       <td>Workspace size applied for on the Device, obtained by calling the first-phase API aclnnConvTbcBackwardGetWorkspaceSize.</td>
     </tr>
     <tr>
       <td>executor</td>
       <td>Input</td>
       <td>op executor, which contains the computation flow of the operator.</td>
     </tr>
     <tr>
       <td>stream</td>
       <td>Input</td>
       <td>Specifies the stream for executing tasks.</td>
     </tr>
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation

  <!-- npu="A3,910b,910,310b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas training products</term>: aclnnConvTbcBackward defaults to a non-deterministic implementation. Determinism can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnConvTbcBackward defaults to a deterministic implementation.
  <!-- end id8 -->

<table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
  <col style="width: 150px">
  <col style="width: 440px">
  <col style="width: 410px">
  <col style="width: 400px">
    </colgroup>
   <thead>
    <tr>
     <th>Constraint Type</th>

     <th><term>Ascend 950PR&950DT products</term></th>


     <th><term>Atlas A2 products</term>, <term>Atlas A3 products</term></th>


     <th><term>Atlas training products</term></th>

   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">self constraint</th>
     <td>      <ul>
        <li>Supports the N dimension greater than or equal to 0, and supports the C dimension greater than or equal to 0 (the use case where the value is 0 is supported only when the N dimension of weight is 0).</li>
        <li>Supports the L dimension greater than or equal to 0 (the use case where the value is 0 is supported only when the L dimension of self is 0).</li>
      </ul>
     </td>
     <td colspan="2">        Supports the N, C, and L dimensions greater than or equal to 0 (the use case where the value is 0 is supported only when the N, C, or L dimension of self is 0).
     </td>
   </tr>
   <tr>
     <th scope="row">input constraint</th>
     <td>      input supports the N and C dimensions greater than or equal to 0, and supports the L dimension greater than or equal to 0 (the use case where the value is 0 is supported only when the L dimension of out is also 0).
      </td>
     <td>        The data type of input does not support HIFLOAT8. Supports the N, C, and L dimensions greater than or equal to 0.
     </td>
     <td>        The input data type does not support BFLOAT16 or HIFLOAT8. Dimensions greater than or equal to 0 are supported for N, C, and L.
     </td>
   </tr>
   <tr>
     <th scope="row">weight constraint</th>
     <td>        <ul>
          <li>weight supports dimensions greater than or equal to 0 for N and C, and dimensions greater than or equal to 0 for L (dimensions equal to 0 are supported only when the L dimension of the derived out is also equal to 0).</li>
          <li>weight supports dimensions greater than or equal to 0 for N (dimensions equal to 0 are supported only when the N dimension of bias and the C dimension of out are also equal to 0); the supported C dimension size is consistent with the C dimension of self; the L dimension size should be within the range [1,255].</li>
        </ul>
     </td>
     <td>          The weight data type does not support HIFLOAT8. Dimensions greater than or equal to 0 are supported for N, C, and L.
     </td>
     <td>          The weight data type does not support BFLOAT16 or HIFLOAT8. Dimensions greater than or equal to 0 are supported for N, C, and L.
     </td>
   </tr>
   <tr>
     <th scope="row">dtype constraint</th>
     <td>        HIFLOAT8 and FLOAT8_E4M3FN are supported only in the gradWeight parameter, and are not supported in other parameters.
     </td>
     <td>          HIFLOAT8 and FLOAT8_E4M3FN are not supported.
     </td>
     <td>          BFLOAT16, HIFLOAT8, and FLOAT8_E4M3FN are not supported.
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType description</th>
     <td>        <ul>
        <li>Enumeration value 1: When the input is FLOAT, the processor converts it to HFLOAT32 for computation. No processing is performed for other input data types.</li>
        <li>Enumeration value 2: This option is not supported when the input is BFLOAT16. No processing is performed for other input data types.</li>
        <li>Enumeration value 3: When the input is FLOAT, it is converted to HFLOAT32 for computation. No processing is performed for other input data types.</li>
        </ul>
     </td>
     <td>        <ul><li>Enumeration value 0: When the input is FLOAT, the Cube computation unit is not supported temporarily, and an error is reported if 0 is used.</li>
        <li>Enumeration value 1: When the input is FLOAT, it is converted to HFLOAT32 for computation. No processing is performed for other input data types.</li>
        <li>Enumeration value 2: This option is not supported when the input is BFLOAT16.</li>
        <li>Enumeration value 3: When the input is FLOAT, the Cube computation unit is not supported temporarily, and an error is reported if 3 is used.</li>
        </ul>
     </td>
     <td>        <ul><li>Enumeration value 0: When the input is FLOAT, the Cube computation unit is not supported temporarily, and an error is reported if 0 is used.</li>
        <li>Enumeration value 1: When the input is FLOAT, it is converted to FLOAT16 for computation. No processing is performed for other input data types.</li>
        <li>Enumeration value 2: This option is not supported when the input is BFLOAT16.</li>
        <li>Enumeration value 3: Not supported temporarily.</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">Other constraints</th>
     <td>        <ul>The computation behavior of the gradient in the padding area depends on the input shape. Depending on the operator optimization strategy, the gradient in the padding area may be directly set to 0.</ul>
     </td>
     <td>-</td>
     <td>-</td>
   </tr>
   </tbody>
</table>

## Example

The following sample code is for reference only. For details about how to compile and run it, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_convolution_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Boilerplate: resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    if (shape.size() == 4) {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *deviceAddr);
    } else {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
    }

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnConvTbcBackwardTest(int32_t deviceId, aclrtStream &stream)
{
    // 1. Initialization.
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs as required by the API.
    std::vector<int64_t> selfShape = {5, 1, 2};
    std::vector<int64_t> inputShape = {5, 1, 2};
    std::vector<int64_t> weightShape = {1, 2, 2};
    std::vector<int64_t> biasShape = {2};
    const int64_t pad = 0;
    int8_t cubeMathType = 1;

    std::vector<int64_t> gradInputShape = {5, 1, 2};
    std::vector<int64_t> gradWeightShape = {1, 2, 2};
    std::vector<int64_t> gradBiasShape = {2};

    // Create the self aclTensor.
    std::vector<float> selfData(GetShapeSize(selfShape), 1);
    aclTensor *self = nullptr;
    void *selfdeviceAddr = nullptr;
    ret = CreateAclTensor(selfData, selfShape, &selfdeviceAddr, aclDataType::ACL_FLOAT, &self);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> selfdeviceAddrPtr(selfdeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the input aclTensor.
    std::vector<float> inputData(GetShapeSize(inputShape), 1);
    aclTensor *input = nullptr;
    void *inputdeviceAddr = nullptr;
    ret = CreateAclTensor(inputData, inputShape, &inputdeviceAddr, aclDataType::ACL_FLOAT, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputdeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the weight aclTensor.
    std::vector<float> weightData(GetShapeSize(weightShape), 1);
    aclTensor *weight = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the bias aclTensor.
    std::vector<float> biasData(GetShapeSize(biasShape), 1);
    aclTensor *bias = nullptr;
    void *biasDeviceAddr = nullptr;
    ret = CreateAclTensor(biasData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the gradInput aclTensor.
    std::vector<float> gradInputData(GetShapeSize(inputShape), 1);
    aclTensor *gradInput = nullptr;
    void *gradInputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradInputData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradInputTensorPtr(gradInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradInputDeviceAddrPtr(gradInputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create gradWeight aclTensor.
    std::vector<float> gradWeightData(GetShapeSize(weightShape), 1);
    aclTensor *gradWeight = nullptr;
    void *gradWeightDeviceAddr = nullptr;
    ret = CreateAclTensor(gradWeightData, weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradWeightTensorPtr(gradWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradWeightDeviceAddrPtr(gradWeightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create gradBias aclTensor.
    std::vector<float> gradBiasData(GetShapeSize(gradBiasShape), 1);
    aclTensor *gradBias = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    ret = CreateAclTensor(gradBiasData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradBiasTensorPtr(gradBias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradBiasDeviceAddrPtr(gradBiasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnConvTbcBackward.
    ret = aclnnConvTbcBackwardGetWorkspaceSize(self, input, weight, bias, pad, cubeMathType, gradInput, gradWeight,
                                               gradBias, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // Request device memory based on the workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnConvTbcBackward.
    ret = aclnnConvTbcBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values. Copy the result from the device memory to the host memory. Modify the code based on the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResult(size, 0);
    ret = aclrtMemcpy(gradInputResult.data(), gradInputResult.size() * sizeof(gradInputResult[0]), gradInputDeviceAddr,
                      size * sizeof(gradInputResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResult[%ld] is: %f\n", i, gradInputResult[i]);
    }

    size = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightResult(size, 0);
    ret = aclrtMemcpy(gradWeightResult.data(), gradWeightResult.size() * sizeof(gradWeightResult[0]), gradWeightDeviceAddr,
                      size * sizeof(gradWeightResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradWeightResult[%ld] is: %f\n", i, gradWeightResult[i]);
    }

    size = GetShapeSize(gradBiasShape);
    std::vector<float> gradBiasResult(size, 0);
    ret = aclrtMemcpy(gradBiasResult.data(), gradBiasResult.size() * sizeof(gradBiasResult[0]), gradBiasDeviceAddr,
                      size * sizeof(gradBiasResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBiasResult[%ld] is: %f\n", i, gradBiasResult[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnConvTbcBackwardTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
