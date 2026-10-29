# aclnnConvolution

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T17:26:07.840Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Implements convolution, supporting 1D convolution, 2D convolution, and 3D convolution, as well as transposed convolution, dilated convolution, and grouped convolution.
  When the input parameter is `transposed = True`, transposed convolution or fractionally-strided convolution is used. It can be viewed as the gradient of a normal convolution or as its reverse operation, that is, recovering the input shape from the output shape of the convolution while keeping the connectivity pattern compatible with the convolution. Its parameters are similar to those of a normal convolution, including the number of input channels, the number of output channels, kernel size, stride, padding, output padding, groups, bias, dilation, etc.

- Formula:

  Assume that the shape of the input is $(N, C_{\text{in}}, D, H, W)$, the shape of (weight) is $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$, and the shape of the output is $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$, where $N$ denotes the batch size, $C$ is the number of channels, $D$, $H$, and $W$ are the depth, height, and width of the sample, respectively, and $K_d$, $K_h$, and $K_w$ are the depth, height, and width of the convolution kernel, respectively. Then the output is represented as:

  $$
  \text{output}(N_i, C_{\text{out}_j}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) = \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k) + \text{bias}(C_{\text{out}_j})
  $$

  where $\star$ denotes the convolution computation, which is determined by the dimensions of the convolution input and the convolution type (dilated convolution (`dilations` > 1), grouped convolution (`groups` > 1)). $N$ denotes the batch size, $C$ denotes the number of channels, and $D$, $H$, and $W$ denote the depth, height, and width, respectively. The formulas for calculating the corresponding output dimensions are as follows:

  - When the parameter `transposed = False`:

    $$
    D_{\text{out}}=[(D + 2 \times padding[0] - dilation[0] \times (K_d - 1) - 1 ) / stride[0]] + 1 \\
    H_{\text{out}}=[(H + 2 \times padding[1] - dilation[1] \times (K_h - 1) - 1 ) / stride[1]] + 1 \\
    W_{\text{out}}=[(W + 2 \times padding[2] - dilation[2] \times (K_w - 1) - 1 ) / stride[2]] + 1
    $$

  - When the parameter `transposed = True`:

    $$
    D_{\text{out}}=(D - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
              \times (K_d - 1) + \text {outputPadding}[0] + 1 \\
    H_{\text{out}}=(H - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
              \times (K_h - 1) + \text {outputPadding}[1] + 1 \\
    W_{\text{out}}=(W - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
              \times (K_w - 1) + \text {outputPadding}[2] + 1
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnConvolutionGetWorkspaceSize API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the aclnnConvolution API to execute the computation.

```cpp
aclnnStatus aclnnConvolutionGetWorkspaceSize(
    const aclTensor       *input,
    const aclTensor       *weight,
    const aclTensor       *bias,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    bool                   transposed,
    const aclIntArray     *outputPadding,
    const int64_t          groups,
    aclTensor             *output,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvolution(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnConvolutionGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:220px">Parameter</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:300px">Description</th>
  <th style="width:350px">Instruction</th>
  <th style="width:212px">Data Type</th>
  <th style="width:180px">Data Format</th>
  <th style="width:120px">Dimension (shape)</th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>input (aclTensor*)</td>
  <td>Input</td>
  <td>input in the formula. Indicates the convolution input.</td>
  <td><ul><li>Empty tensors are supported.</li><li>The data type must comply with the data type deduction rule with weight (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
  <td>NCL, NCHW, NCDHW</td>
  <td>3-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight (aclTensor*)</td>
  <td>Input</td>
  <td>weight in the formula. Indicates the convolution weight.</td>
  <td><ul><li>Supports empty tensors.</li><li>The data type must follow the data type deduction rules with input (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#Constraints">Constraints</a>).</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
  <td>NCL, NCHW, NCDHW</td>
  <td>3-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias (aclTensor*)</td>
  <td>Input</td>
  <td>The bias in the formula, indicating the convolution bias.</td>
  <td><ul><li>For the no-bias use case, pass in nullptr.</li><li>When transposed=false, it is one-dimensional and the value equals the first dimension of weight; when transposed=true, it is one-dimensional and the value equals weight.shape[1] * groups.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16</td>
  <td>ND, NCL, NCHW, NCDHW</td>
  <td>1-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>stride (aclIntArray*)</td>
  <td>Input</td>
  <td>Convolution scan stride.</td>
  <td>Array Length must equal the number of input dimensions minus 2, and the value must be greater than 0.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding (aclIntArray*)</td>
  <td>Input</td>
  <td>The padding of input.</td>
  <td>When the dimension of input is 3 and transposed=false, the Array Length of padding must be 1 or 2. When the dimension of input is 3 and transposed=true, the Array Length of padding must be 1. When the dimension of input is 4, the Array Length of padding must be 2 or 4. When the dimension of input is 5, the Array Length of padding must be 3. The value of padding must be greater than or equal to 0.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation (aclIntArray*)</td>
  <td>Input</td>
  <td>Spacing between elements in the convolution kernel.</td>
  <td>The array length must equal the dimension of input minus 2, and the value must be greater than 0.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>transposed (bool)</td>
  <td>Input</td>
  <td>Specifies whether to perform transposed convolution.</td>
  <td>-</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>outputPadding (aclIntArray*)</td>
  <td>Input</td>
  <td>Padding added to all sides of the output in transposed convolution.</td>
  <td>This parameter is ignored in non-transposed convolution. The array length must equal the dimension of input minus 2. The value must be greater than or equal to 0 and less than the corresponding value of stride or dilation.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups (int64_t)</td>
  <td>Input</td>
  <td>Number of blocked connections from input channels to output channels.</td>
  <td>The value must be in the range [1,65535] and must satisfy groups * the C dimension of weight = the C dimension of input.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>output (aclTensor*)</td>
  <td>Output</td>
  <td>Denotes the convolution output.</td>
  <td><ul><li>Supports empty tensors.</li><li>The data type must be consistent with the data type deduced from input and weight.</li><li>The number of channels equals the first dimension of weight, and other dimensions are ≥ 0.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
  <td>NCL, NCHW, NCDHW</td>
  <td>3-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>cubeMathType (int8_t)</td>
  <td>Input</td>
  <td>Determines which computation logic the Cube unit uses for the computation.</td>
  <td><ul><li>If the input data types have a <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>, this parameter processes the deducted data type by default.</li>
    <li>The supported enumeration values are as follows:</li>
    <ul><li> 0 (KEEP_DTYPE): Keeps the input data type for the computation.</li></ul>
    <ul><li> 1 (ALLOW_FP32_DOWN_PRECISION): Allows FLOAT to be computed with reduced precision to improve performance.</li></ul>
    <ul><li> 2 (USE_FP16): Uses FLOAT16 precision for the computation.</li></ul>
    <ul><li> 3 (USE_HF32): Uses HFLOAT32 (mixed precision) for the computation.</li></ul></ul></td>
  <td>INT8</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*)</td>
  <td>Output</td>
  <td>Returns the workspace size to be allocated on the Device.</td>
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

  aclnnStatus: return status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Codes</a>.

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
  </tr>  </thead>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The pointer-type input parameter is a null pointer.</td>
  </tr>
  <tr>
  <td rowspan="11" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="11" align="left">161002</td>
  <td align="left">The data type or data format of input, weight, bias, or output is not supported.</td>
  </tr>
  <tr><td align="left">The data types of input and output are inconsistent; when transposed=false, inconsistent data types of input and output are supported and do not trigger this error.</td></tr>
  <tr><td align="left">The input shapes of stride, padding, dilation, or outputPadding are incorrect.</td></tr>
  <tr><td align="left">The input value of groups is incorrect.</td></tr>
  <tr><td align="left">The shape of output does not match the infershape result.</td></tr>
  <tr><td align="left">The value of outputPadding does not meet the requirement.</td></tr>
  <tr><td align="left">For the empty tensors passed to input, weight, bias, or output, those with some dimensions set to 0 do not meet requirements.</td></tr>
  <tr><td align="left">When transposed=false, the spatial size of input after padding is smaller than that of weight (expanded by dilation, if dilation > 1).</td></tr>
  <tr><td align="left">The shape of bias is not 1 when transposed=true.</td></tr>
  <tr><td align="left">stride or dilation is less than 0, which does not meet the requirement.</td></tr>
  <tr><td align="left">The current processor does not support convolution.</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">Internal API verification error, usually caused by input data or attribute specifications beyond the supported range.</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">An exception occurs when the API calls the NPU runtime API, for example, SocVersion is not supported.</td>
  </tr>
  </table>

## aclnnConvolution

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
  <td>Memory address of the workspace allocated on the Device.</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Input</td>
  <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnConvolutionGetWorkspaceSize.</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>Input</td>
  <td>Op executor that contains the operator computation process.</td>
  </tr>
  <tr>
  <td>stream</td>
  <td>Input</td>
  <td>Stream for executing tasks.</td>
  </tr>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Codes</a>.

## Constraints

- deterministic computation
  - aclnnConvolution defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width:150px">
    <col style="width:450px">
    <col style="width:400px">
    <col style="width:400px">
    <col style="width:550px">
    </colgroup>
   <thead>
    <tr>
     <th>Constraint Type</th>

     <th><term>Atlas A2 products</term>,<term>Atlas A3 products</term></th>


     <th><term>Atlas 200I/500 A2 inference products</term></th>


     <th><term>Atlas inference products</term></th>


     <th><term>Ascend 950PR&950DT products</term></th>

   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">input, weight</th>
     <td>        <ul><li>The data types of input and weight do not support HIFLOAT8 or FLOAT8_E4M3FN.</li></ul>
        <ul><li>For the conv2d and conv3d transposed=true use cases, weight H and W must be in the range [1,255], and the size of each of the other dimensions must be greater than or equal to 1.</li></ul>
        <ul><li>For the conv1d transposed=true use case, weight L must be in the range [1,255], and the size of each of the other dimensions must be greater than or equal to 1.</li></ul>
        <ul><li>For the conv3d forward scenario, weight H and W must be in the range [1,511].</li></ul>
     </td>
     <td>        The data types of input and weight do not support HIFLOAT8 or FLOAT8_E4M3FN.
     </td>
     <td>        <ul><li>The data types of input and weight do not support BFLOAT16, HIFLOAT8, or FLOAT8_E4M3FN.</li></ul>
        <ul><li>The H and W dimension sizes of input must be less than or equal to 4096, and the D, H, and W dimension sizes of weight must be less than or equal to 255.</li></ul>
     </td>
     <td>           The data types of input and weight support FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8.
           <ul>
              <li>When transposed=true: the data type of input additionally supports FLOAT8_E4M3FN. When the data type of input is HIFLOAT8 or FLOAT8_E4M3FN, the data types of output and weight must be the same as that of input. The N dimension must be greater than or equal to 0, and the size of each of the other dimensions must be greater than or equal to 1. The data type of weight additionally supports FLOAT8_E4M3FN, and the size of all dimensions must be greater than or equal to 0 (if the D, H, or W dimension is 0, the corresponding dimension of the inferred output must also be 0).</li>
              <li>When transposed=false: when the data type of input is HIFLOAT8, the data type of weight must be the same as that of input. The N dimension must be greater than or equal to 0. D, H, and W must be greater than or equal to 0 (a value of 0 is supported only when the corresponding D, H, or W dimension of the inferred output is also 0), and C must be greater than or equal to 0 (a value of 0 is supported only when one of the N, C, D, H, and W dimensions of the inferred output is 0). weight H and W must be in the range [1,511]. The N dimension size must be greater than or equal to 0 (a value of 0 is supported only when the N dimensions of bias and output are also 0). The supported C dimension size is the same as that of the C dimension of input.</li>
            </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>        <ul>
          <li>The data type of bias does not support HIFLOAT8 or FLOAT8_E4M3FN. The data type must be the same as those of input and weight.</li>
          <li>In conv1d, conv2d, and conv3d forward scenarios, bias is converted to FLOAT for computation.</li>
        </ul>
     </td>
     <td>        <ul>
          <li>The bias data type does not support HIFLOAT8 or FLOAT8_E4M3FN. The data type must be the same as that of input and weight.</li>
          <li>In conv1d, conv2d, and conv3d forward scenarios, bias is converted to FLOAT for computation.</li>
        </ul>
     </td>
     <td>        The bias data type does not support BFLOAT16, HIFLOAT8, or FLOAT8_E4M3FN. Only FLOAT16 is supported in the Conv3D forward scenario.
     </td>
     <td>          The bias data type does not support HIFLOAT8 or FLOAT8_E4M3FN.
          <ul>
            <li>transposed=true: bias is not supported when input and weight are of data type HIFLOAT8 or FLOAT8_E4M3FN.</li>
            <li>transposed=false: if input and weight are of data type HIFLOAT8, bias is converted to FLOAT for computation.</li>
          </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">stride</th>
     <td colspan="3">        In the conv3d transposed=true scenario, strideD must be greater than or equal to 1, and strideH and strideW must be in the range [1,63]. In conv1d and conv2d transposed=true scenarios, each value must be greater than or equal to 1. In the conv3d forward scenario, strideH and strideW must be in the range [1,63].
     </td>
     <td>        In the conv3d transposed=true scenario, strideD, strideH, and strideW must be greater than or equal to 1. In the conv3d transposed=false scenario, strideH and strideW must be in the range [1,63], and strideD must be in the range [1, 1000000].
     </td>
   </tr>
   <tr>
     <th scope="row">padding</th>
     <td colspan="3">        In the conv3d transposed=true scenario, paddingD must be greater than or equal to 0, and paddingH and paddingW must be in the range [0,255]. In conv1d and conv2d transposed=false scenarios, each value must be in the range [0,255]. In the conv3d forward scenario, paddingH and paddingW must be in the range [0,255].
     </td>
     <td>        In the conv3d transposed=true scenario, paddingD, paddingH, and paddingW must be greater than or equal to 0. In the conv3d transposed=false scenario, paddingH and paddingW must be in the range [0,255], and paddingD must be in the range [0, 1000000].
     </td>
   </tr>
   <tr>
     <th scope="row">dilation</th>
     <td colspan="3">        For conv1d, conv2d, and conv3d with transposed=true, each value must be in the range [1,255]. For conv3d in the Forward Scenario, dilationH and dilationW must be in the range [1,255].
     </td>
     <td>        For conv3d with transposed=true, dilationD, dilationH, and dilationW must be greater than or equal to 1. For conv3d with transposed=false, dilationH and dilationW must be in the range [1,255], and dilationD must be in the range [1, 1000000].
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType</th>
     <td>        <ul>
          <li>If the value is 0(KEEP_DTYPE), FLOAT input is not supported yet.</li>
          <li>If the value is 1(ALLOW_FP32_DOWN_PRECISION), FLOAT input can be converted to HFLOAT32 for computation.</li>
          <li>If the value is 2(USE_FP16), this option is not supported for BFLOAT16 input.</li>
          <li>If the value is 3(USE_HF32), FLOAT input is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
     <td>        <ul>
          <li>If the value is 0(KEEP_DTYPE), FLOAT input is not supported yet.</li>
          <li>If the value is 1(ALLOW_FP32_DOWN_PRECISION), FLOAT input can be converted to HFLOAT32 for computation.</li>
          <li>If the value is 2(USE_FP16), this option is not supported for BFLOAT16 input.</li>
          <li>If the value is 3(USE_HF32), FLOAT input is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
     <td>        <ul>
          <li>If the value is 0(KEEP_DTYPE), FLOAT input is not supported yet.</li>
          <li>If the value is 1(ALLOW_FP32_DOWN_PRECISION), FLOAT input can be converted to FLOAT16 for computation.</li>
          <li>If the value is 2(USE_FP16), this option is not supported for BFLOAT16 input.</li>
          <li>If the value is 3(USE_HF32), it is not supported yet.</li>
        </ul>
     </td>
     <td>        <ul>
          <li>If the value is 1(ALLOW_FP32_DOWN_PRECISION), FLOAT input can be converted to HFLOAT32 for computation.</li>
          <li>If the value is 2(USE_FP16), this option is not supported for BFLOAT16 input.</li>
          <li>If the value is 3(USE_HF32), FLOAT input is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">Other Constraints</th>
     <td>        Each dimension size of each group of tensors in input, weight, and bias must not exceed 1000000.
     </td>
     <td>        <ul>
          <li>Currently, only 2D convolution with transposed=false is supported. 1D and 3D convolution are not supported yet.</li>
          <li>Each dimension size of each group of tensors in input, weight, and bias must not exceed 1000000.</li>
        </ul>
     </td>
     <td>        <ul>
          <li>1D and 2D Convolution are supported. For 3D Convolution, only FLOAT16 is supported in the forward scenario.</li>
          <li>For each group of tensors in input, weight, and bias, the Dimension Size of each dimension should not be greater than 1000000.</li>
        </ul>
     </td>
     <td>        When transposed is true, 1D, 2D, and 3D Convolution are supported, and empty tensors are supported. In this case, the computation behavior of gradients in the padding area depends on the input shape. Depending on the operator optimization strategy, gradients in the padding area may be directly set to 0.
     </td>
   </tr>
   </tbody>
  </table>

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate: resource initialization
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
  // Call aclrtMalloc to request device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
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

int aclnnConvolutionTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
  std::vector<int64_t> shapeInput = {2, 2, 2, 2};
  std::vector<int64_t> shapeWeight = {1, 2, 1, 1};
  std::vector<int64_t> shapeResult = {2, 1, 2, 2};
  std::vector<int64_t> convStrides;
  std::vector<int64_t> convPads;
  std::vector<int64_t> convOutPads;
  std::vector<int64_t> convDilations;

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* result = nullptr;
  std::vector<float> inputData(GetShapeSize(shapeInput), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> outputData(GetShapeSize(shapeResult), 1);
  convStrides = {1, 1};
  convPads = {0, 0};
  convOutPads = {0, 0};
  convDilations = {1, 1};

  // Create the input aclTensor.
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, aclDataType::ACL_FLOAT, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create the weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, aclDataType::ACL_FLOAT, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create the out aclTensor.
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outPadsPtr(outPads, aclDestroyIntArray);
  CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Modify it to the specific API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConvolution.
  ret = aclnnConvolutionGetWorkspaceSize(input, weight, nullptr, strides, pads, dilations, false, outPads, 1, result, 1,
                                         &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnConvolution.
  ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from the device memory to the host. Modify it according to the definition of the specific API.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvolutionTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
