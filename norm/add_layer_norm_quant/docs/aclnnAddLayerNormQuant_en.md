# aclnnAddLayerNormQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:44:50.826Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: The LayerNorm operator is a commonly used normalization operation in large models. The AddLayerNormQuant operator fuses the Add operator before LayerNorm and the LayerNorm normalized output to one or two downstream quantization operators, reducing data movement. The downstream quantization operator of LayerNorm can be a Quantize, AscendQuantV2, or DynamicQuant operator. The specific quantization operator type is determined by the attr parameters divMode and quantMode. When there are two downstream quantization operators, the operator types, input/output dtype combinations, and optional input combinations of the two quantization operators must be completely identical.
- Formula:

  $$
  x = x1 + x2 + biasOptional
  $$

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
  $$

  - When quantMode is set to "static", the outputs outScales1Out and outScales2Out are meaningless. Depending on the divMode input, the fused quantization operator may be Quantize or AscendQuantV2:
    - When divMode is set to true, the fused quantization operator is Quantize, with the following formula:

        $$
        y1Out = round(y / scales1Optional + zeroPoints1Optional)
        $$

        $
        y2Out = round(y / scales2Optional + zeroPoints2Optional), \quad \text{if and only if scales2Optional is present}
        $

    - When divMode is set to false, the fused quantization operator is AscendQuantV2, and the formula is as follows:

        $$
        y1Out = round(y * scales1Optional + zeroPoints1Optional)
        $$

        $
        y2Out = round(y * scales2Optional + zeroPoints2Optional), \quad \text{if and only if scales2Optional is present}
        $

  - When quantMode is set to "dynamic", the inputs zeroPoints1Optional and zeroPoints2Optional have no practical significance. The fused quantization operator is DynamicQuant, and divMode is ineffective in this case:
    - If neither scales1Optional nor scales2Optional is provided, the outputs y2Out and scale2Out have no practical significance and can be ignored. The formula is as follows:

        $$
        outScales1Out = row\_max(abs(y))/127
        $$

        $$
        y1Out = round(y / outScales1Out)
        $$

    - If only scales1Optional is input, the outputs y2Out and scale2Out are meaningless and can be ignored. The corresponding formula is as follows:

        $$
        tmp1 = y * scales1Optional
        $$

        $$
        outScales1Out = row\_max(abs(tmp1))/127
        $$

        $$
        y1Out = round(tmp1 / outScales1Out)
        $$

    - If both scales1Optional and scales2Optional are present, y2Out and scale2Out are valid. The corresponding formula is as follows:

        $$
        tmp1 = y * scales1Optional, \quad tmp2 = y * scales2Optional
        $$

        $$
        outScales1Out = row\_max(abs(tmp1))/127, \quad outScales2Out = row\_max(abs(tmp2))/127
        $$

        $$
        y1Out = round(tmp1 / outScales1Out),\quad y2Out = round(tmp2 / outScales2Out)
        $$

        where row_max denotes finding the maximum value of each row.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnAddLayerNormQuantGetWorkspaceSize` API to obtain the input parameters and the workspace size required by the computation flow, and then call the `aclnnAddLayerNormQuant` API to perform the computation.

```Cpp
aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *biasOptional,
  const aclTensor *scales1Optional,
  const aclTensor *scales2Optional,
  const aclTensor *zeroPoints1Optional,
  const aclTensor *zeroPoints2Optional,
  const char      *quantMode,
  double           epsilon,
  bool             additionalOutput,
  bool             divMode,
  aclTensor       *y1Out,
  aclTensor       *y2Out,
  aclTensor       *xOut,
  aclTensor       *outScales1Out,
  aclTensor       *outScales2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormQuantGetWorkspaceSize

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
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>Input of the addition computation in AddLayerNorm. The operator computes x1 + x2 + biasOptional and applies layer normalization to the result. Corresponds to `x1` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When quantMode = "static", the shape supports 1-8 dimensions, and the supported data types are FLOAT32, FLOAT16, and BFLOAT16.</li><li>When quantMode = "dynamic", the shape supports 2-8 dimensions, and the supported data types are FLOAT16 and BFLOAT16.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Input of the addition computation in AddLayerNorm. The operator computes x1 + x2 + biasOptional and applies layer normalization to the result. Corresponds to `x2` in the formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>When quantMode = "static", the shape supports 1-8 dimensions, and the supported data types are FLOAT32, FLOAT16, and BFLOAT16.</li><li>When quantMode = "dynamic", the shape supports 2-8 dimensions, and the supported data types are FLOAT16 and BFLOAT16.</li><li>The shape is the same as that of `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma(aclTensor*)</td>
      <td>Input</td>
      <td>Represents the gamma parameter in layer normalization. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>When quantMode = "static", the shape supports 1-8 dimensions, and the supported data types are FLOAT32, FLOAT16, and BFLOAT16.</li><li>When quantMode = "dynamic", the shape supports 2-8 dimensions, and the supported data types are FLOAT16 and BFLOAT16.</li><li>The data dimensions must be the same as the last dimensions of `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Corresponds to beta in the LayerNorm formula, representing the beta parameter in layer normalization. Corresponds to `beta` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When quantMode = "static", supported data types are FLOAT32, FLOAT16, and BFLOAT16.</li><li>When quantMode = "dynamic", the shape supports 2-8 dimensions, and supported data types are FLOAT16 and BFLOAT16.</li><li>The shape is the same as that of `gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter. Pass an aclTensor that meets the following constraints, or use nullptr to indicate that this optional input is absent. Indicates the input of the addition computation in AddLayerNorm. The operator computes x1 + x2 + biasOptional and performs layer normalization on the result. Corresponds to `biasOptional` in the corresponding formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>When quantMode = "static", the shape supports 1-8 dimensions, and the data type supports FLOAT32, FLOAT16, and BFLOAT16.</li><li>When quantMode = "dynamic", the shape supports 2-8 dimensions, and the data type supports FLOAT16 and BFLOAT16.</li><li>The shape can be the same as that of `gamma`/`beta` or `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales1Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, representing the scale/smooth input of the first fused quantization operator. Corresponds to `scales1Optional` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>When quantMode = "static", the supported data types are FLOAT32, FLOAT16, and BFLOAT16.</li><li>When quantMode = "dynamic", the supported data types are FLOAT16 and BFLOAT16.</li><li>The shape is the same as that of `gamma`.</li><li>When the optional input parameter is passed in, see the value constraint in <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales2Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, representing the scale/smooth input of the second fused quantization operator. Corresponds to `scales2Optional` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Supports the FLOAT32, FLOAT16, and BFLOAT16 data types when quantMode = "static".</li><li>Supports the FLOAT16 and BFLOAT16 data types when quantMode = "dynamic".</li><li>The shape is the same as `gamma`.</li><li>When the optional input is passed, see the value constraints in <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints1Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input, representing the zeroPoints input of the first fused quantization operator. Valid only when quantMode = "static". Corresponds to `zeroPoints1Optional` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Supports the FLOAT32, FLOAT16, and BFLOAT16 data types when quantMode = "static".</li><li>Supports the FLOAT16 and BFLOAT16 data types when quantMode = "dynamic".</li><li>The shape is the same as `gamma`.</li><li>When the optional input is passed, see the value constraints in <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints2Optional(aclTensor*)</td>
      <td>Input</td>
      <td>Optional input, representing the zeroPoints input of the second fused quantization operator. Valid only when quantMode = "static". Corresponds to `zeroPoints2Optional` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When quantMode = "static", the supported data types are FLOAT32, FLOAT16, and BFLOAT16.</li><li>When quantMode = "dynamic", the supported data types are FLOAT16 and BFLOAT16.</li><li>The shape is the same as `gamma`.</li><li>When the optional input is passed, see <a href="#constraints">Constraints</a> for the value constraint.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode (char*)</td>
      <td>Input</td>
      <td>Quantization mode, used to determine whether the fused quantization operator is static or dynamic. Corresponds to `quantMode` in the formula description. The value can be "static" or "dynamic".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Specifies the epsilon in the corresponding LayerNorm, a value added to the denominator to ensure numerical stability. Corresponds to `epsilon` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additionalOutput (bool)</td>
      <td>Input</td>
      <td>Specifies whether to enable the output of x=x1+x2+biasOptional.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode (bool)</td>
      <td>Input</td>
      <td>Valid when quantMode = "static". Indicates whether scale is processed by multiplication or division in static quantization. When true, the operator performs division on scale in the quantization computation. Corresponds to `divMode` in the formula description.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of quantizing y, the LayerNorm output, with the first quantization operator. Corresponds to `y1Out` in the formula.</td>
      <td>The shape must be the same as that of input x1/x2.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of quantizing y, the LayerNorm output, by the second quantization operator. Corresponds to `y2Out` in the formula.</td>
      <td>The shape must be the same as that of input x1/x2.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Add result output x. Corresponds to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When quantMode = "static", the data type can be FLOAT32, FLOAT16, or BFLOAT16.</li><li>When quantMode = "dynamic", the data type can be FLOAT16 or BFLOAT16.</li><li>The shape must be the same as that of input `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outScales1Out (aclTensor*)</td>
      <td>Output</td>
      <td>outScale result output of the first dynamic quantization computation, valid only when quantMode="dynamic". Corresponds to `outScales1Out` in the formula.</td>
      <td>Shape is the shape of input `x1` with the last dimension removed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outScales2Out(aclTensor*)</td>
      <td>Output</td>
      <td>Output of the outScale result for the second dynamic quantization computation, valid only when quantMode="dynamic". Corresponds to `outScales2Out` in the corresponding formula.</td>
      <td>Shape is the shape of input `x1` with the last dimension removed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If a required input, output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The hardware platform is not in the range of supported products.</td>
    </tr>
    <tr>
      <td>The value of quantMode is not "static" or "dynamic".</td>
    </tr>
    <tr>
      <td>The input data type combination is invalid. For valid data type combinations, see the constraints and descriptions below.</td>
    </tr>
    <tr>
      <td>The shapes of x1, gamma, and outScales1Out satisfy the following conditions:
      <ol>
      <li>The dimensions of gamma are inconsistent with the last few dimensions of x1.</li>
      <li>When the quantization mode is dynamic, that is, the input quantMode is set to "dynamic", the dimensions of x1 are fewer than 2, or the dimensions of gamma are not 1.</li>
      <li>When the quantization mode is dynamic, that is, the input quantMode is set to "dynamic", the shape of outScales1Out is not the shape of input x1 with the last dimension removed.</li></ol></td>
    </tr>
    <tr>
      <td>The shapes of all input tensors satisfy the following equivalence relationships:
      <ol>
      <li>1. The shapes of x1, x2, xOut, and y1 are different; when the optional input scales2Optional is present, this condition is tightened to the shapes of x1, x2, xOut, y1, and y2 being different.</li>
      <li>The shapes of gamma and beta are different; when the optional inputs scales1Optional, scales2Optional, zeroPoints1Optional, and zeroPoints2Optional are present, their shapes are different from that of gamma.</li>
      <li>When biasOptional is present, its shape is different from both that of gamma and that of x1.</li>
      <li>When the quantization mode is dynamic, that is, the input quantMode is set to "dynamic", and the optional input scales2Optional is present, the shape of outScales1Out is different from that of outScales2Out.</li>
      </ol>
      </td>
    </tr>
    <tr>
      <td>The presence status of the optional inputs (scales1Optional, scales2Optional, zeroPoints1Optional, and zeroPoints2Optional) does not satisfy the specific combination relationship.</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNormQuant

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
      <td>Address of the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnAddLayerNormQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: Return status code. For details, see [Error Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Functional dimensions:

  * The optional inputs (scales1Optional, scales2Optional, zeroPoints1Optional, zeroPoints2Optional) support the following combinations:

    | scales1Optional | scales2Optional | zeroPoints1Optional | zeroPoints2Optional | quantMode | Whether Valid |
    | --------------- | --------------- | ------------------- | ------------------- | ----------------- | :------ |
    | T               | T               | T                   | T                   | "static"          | T       |
    | T               | T               | T                   | F                   | "static"          | F       |
    | T               | T               | F                   | T                   | "static"          | F       |
    | T               | T               | F                   | F                   | "static"          | T       |
    | T               | F               | T                   | T                   | "static"          | F       |
    | T               | F               | T                   | F                   | "static"          | T       |
    | T               | F               | F                   | T                   | "static"          | F       |
    | T               | F               | F                   | F                   | "static"          | T       |
    | F               | X               | X                   | X                   | "static"          | F       |
    | T               | T               | F                   | F                   | "dynamic"         | T       |
    | T               | F               | F                   | F                   | "dynamic"         | T       |
    | F               | T               | F                   | F                   | "dynamic"         | F       |
    | F               | F               | F                   | F                   | "dynamic"         | T       |
    | X               | X               | T                   | X                   | "dynamic"         | F       |
    | X               | X               | X                   | T                   | "dynamic"         | F       |

    Where:
    - `T` indicates that the optional input is present and `/` indicates valid.
    - `F` indicates that the optional input is absent, and `/` is invalid.
    - `X` indicates that either case is valid.
- Data type support description:
  - When `quantMode` is "static":

    | x1 Data Type | x2 Data Type | Gamma Data Type | Beta Data Type | Bias Data Type | scale1 Data Type | scale2 Data Type | zeroPoints1 Data Type | zeroPoints2 Data Type | y1 Data Type | y2 Data Type | X Data Type | outScale1 Data Type | outScale2 Data Type |
    | ---------- | --------- | ------------- | ----------- | ------------ | -------------- | -------------- | ------------------ | ------------------- | --------- | ---------- | --------- | ----------------- | :--------------- |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT16        | FLOAT16        | FLOAT16            | FLOAT16             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | BFLOAT16       | BFLOAT16       | BFLOAT16           | BFLOAT16            | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |
    | FLOAT32    | FLOAT32   | FLOAT32       | FLOAT32     | FLOAT32      | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | FLOAT32   | FLOAT32           | FLOAT32          |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |

  - When `quantMode` is "dynamic":

    | x1 Data Type | x2 Data Type | Gamma Data Type | Beta Data Type | Bias Data Type | scale1 Data Type | scale2 Data Type | zeroPoints1 Data Type | zeroPoints2 Data Type | y1 Data Type | y2 Data Type | X Data Type | outScale1 Data Type | outScale2 Data Type |
    | ---------- | --------- | ------------- | ----------- | ------------ | -------------- | -------------- | ------------------ | ------------------- | --------- | ---------- | --------- | ----------------- | :--------------- |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT16        | FLOAT16        | FLOAT16            | FLOAT16             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | BFLOAT16       | BFLOAT16       | BFLOAT16           | BFLOAT16            | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |

- Deterministic computation:
  - aclnnAddLayerNormQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm_quant.h"

#define CHECK_RET(cond, return_expr)\
do {                                \
  if (!(cond)) {                    \
    return_expr;                    \
  }                                 \
} while (0)

#define LOG_PRINT(message, ...)   \
    do {                          \
  printf(message, ##__VA_ARGS__); \
} while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate: resource initialization.
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output in a custom manner based on the API. In this example, the case without the optional bias input and the case with the bias input are each called once.
  float eps = 1e-6;
  bool additionalOut = true;
  bool divMode = true;
  const char* quantMode = "static";

  std::vector<int64_t> xShape = {8, 64};
  std::vector<int64_t> gammaShape = {64};

  std::vector<int64_t> reduceShape = {8,};

  void *x1DeviceAddr = nullptr;
  void *x2DeviceAddr = nullptr;
  void *betaDeviceAddr = nullptr;
  void *gammaDeviceAddr = nullptr;
  void *biasDeviceAddr = nullptr;
  void *s1DeviceAddr = nullptr;
  void *s2DeviceAddr = nullptr;
  void *z1DeviceAddr = nullptr;
  void *z2DeviceAddr = nullptr;

  // For the output Device address without bias
  void *y1DeviceAddr = nullptr;
  void *y2DeviceAddr = nullptr;
  void *xDeviceAddr = nullptr;
  void *outScales1DeviceAddr = nullptr;
  void *outScales2DeviceAddr = nullptr;

  aclTensor *x1 = nullptr;
  aclTensor *x2 = nullptr;
  aclTensor *beta = nullptr;
  aclTensor *gamma = nullptr;
  aclTensor *bias = nullptr;
  aclTensor *s1 = nullptr;
  aclTensor *s2 = nullptr;
  aclTensor *z1 = nullptr;
  aclTensor *z2 = nullptr;

  // For the aclTensor without bias
  aclTensor *y1 = nullptr;
  aclTensor *y2 = nullptr;
  aclTensor *x = nullptr;
  aclTensor *outScales1 = nullptr;
  aclTensor *outScales2 = nullptr;

  int64_t xShapeSize = GetShapeSize(xShape);
  int64_t gammaShapeSize = GetShapeSize(gammaShape);
  int64_t reduceShapeSize = GetShapeSize(reduceShape);

  std::vector<float> x1HostData(xShapeSize, 0x3C00);
  std::vector<float> x2HostData(xShapeSize, 0x3C00);
  std::vector<float> gammaHostData(gammaShapeSize, 0x3C00);
  std::vector<float> betaHostData(gammaShapeSize, 0x3C00);
  std::vector<float> biasHostData(gammaShapeSize, 0x3C00);

  std::vector<float> s1HostData(gammaShapeSize, 0x3C00);
  std::vector<float> s2HostData(gammaShapeSize, 0x3C00);
  std::vector<float> z1HostData(gammaShapeSize, 0x3C00);
  std::vector<float> z2HostData(gammaShapeSize, 0x3C00);

  // For the HostData without bias
  std::vector<int8_t> y1HostData(xShapeSize, 0);
  std::vector<int8_t> y2HostData(xShapeSize, 0);
  std::vector<float> xHostData(xShapeSize, 0);
  std::vector<float> outScales1HostData(reduceShapeSize, 0);
  std::vector<float> outScales2HostData(reduceShapeSize, 0);

  // Create the self aclTensor.
  ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaHostData,  gammaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, gammaShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(s1HostData, gammaShape, &s1DeviceAddr, aclDataType::ACL_FLOAT, &s1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(s2HostData, gammaShape, &s2DeviceAddr, aclDataType::ACL_FLOAT, &s2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(z1HostData, gammaShape, &z1DeviceAddr, aclDataType::ACL_FLOAT, &z1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(z2HostData, gammaShape, &z2DeviceAddr, aclDataType::ACL_FLOAT, &z2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the aclTensor without bias.
  ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales1HostData, reduceShape, &outScales1DeviceAddr, aclDataType::ACL_FLOAT, &outScales1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales2HostData, reduceShape, &outScales2DeviceAddr, aclDataType::ACL_FLOAT, &outScales2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnAddLayerNormQuant API call example, including one with bias and one without bias
  // 3. Call the CANN operator library API. Replace it with the actual API name.

  // 3.1 Example without the optional input bias
  // Call the first-phase API of aclnnAddLayerNormQuant.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  ret = aclnnAddLayerNormQuantGetWorkspaceSize(x1, x2, gamma, beta, bias, s1, s2, nullptr, nullptr, quantMode, eps, additionalOut, divMode, y1, y2, x, outScales1, outScales2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAddLayerNormQuant
  ret = aclnnAddLayerNormQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from device memory to host memory. Modify the code based on the specific API definition.

  auto y1Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY1(y1Size, 0);
  ret = aclrtMemcpy(resultDataY1.data(), resultDataY1.size() * sizeof(resultDataY1[0]), y1DeviceAddr, y1Size * sizeof(resultDataY1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant y1 output");
  for (int64_t i = 0; i < y1Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY1[i]);
  }

  auto y2Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY2(y2Size, 0);
  ret = aclrtMemcpy(resultDataY2.data(), resultDataY2.size() * sizeof(resultDataY2[0]), y2DeviceAddr, y2Size * sizeof(resultDataY2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant y2 output");
  for (int64_t i = 0; i < y2Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY2[i]);
  }

  auto xSize = GetShapeSize(xShape);
  std::vector<float> resultDataX(xSize, 0);
  ret = aclrtMemcpy(resultDataX.data(), resultDataX.size() * sizeof(resultDataX[0]), xDeviceAddr, xSize * sizeof(resultDataX[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant x output");
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataX[i]);
  }

  auto outScale1Size = GetShapeSize(reduceShape);
  std::vector<float> resultDataOutScale1(outScale1Size, 0);
  ret = aclrtMemcpy(resultDataOutScale1.data(), resultDataOutScale1.size() * sizeof(resultDataOutScale1[0]), outScales1DeviceAddr, outScale1Size * sizeof(resultDataOutScale1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant outScale1 output");
  for (int64_t i = 0; i < outScale1Size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataOutScale1[i]);
  }

  auto outScale2Size = GetShapeSize(reduceShape);
  std::vector<float> resultDataOutScale2(outScale2Size, 0);
  ret = aclrtMemcpy(resultDataOutScale2.data(), resultDataOutScale2.size() * sizeof(resultDataOutScale2[0]), outScales2DeviceAddr, outScale2Size * sizeof(resultDataOutScale2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant outScale2 output");
  for (int64_t i = 0; i < outScale2Size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataOutScale2[i]);
  }


  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(beta);
  aclDestroyTensor(gamma);
  aclDestroyTensor(bias);
  aclDestroyTensor(s1);
  aclDestroyTensor(s2);
  aclDestroyTensor(z1);
  aclDestroyTensor(z2);

  aclDestroyTensor(y1);
  aclDestroyTensor(y2);
  aclDestroyTensor(x);
  aclDestroyTensor(outScales1);
  aclDestroyTensor(outScales2);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(s1DeviceAddr);
  aclrtFree(s2DeviceAddr);
  aclrtFree(z1DeviceAddr);
  aclrtFree(z2DeviceAddr);

  aclrtFree(y1DeviceAddr);
  aclrtFree(y2DeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(outScales1DeviceAddr);
  aclrtFree(outScales2DeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
