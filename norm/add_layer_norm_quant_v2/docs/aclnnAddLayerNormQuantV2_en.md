# aclnnAddLayerNormQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:40:13.151Z -->

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

- Description: The LayerNorm operator is a normalization operation commonly used in large models. The AddLayerNormQuantV2 operator fuses the Add operator before LayerNorm and the normalized output of LayerNorm into 1 or 2 downstream quantization operators, reducing move-in and move-out operations. The quantization operators downstream of LayerNorm can be Quantize, AscendQuantV2, or DynamicQuant operators, and the specific quantization operator type is determined by the attr parameters divMode and quantMode. When there are 2 downstream quantization operators, the two quantization operators must have exactly the same operator types, input/output dtype combinations, and optional input combinations. Compared with [aclnnAddLayerNormQuant](../../add_layer_norm_quant/docs/aclnnAddLayerNormQuant.md), this API adds the LayerNorm output. Select an appropriate API based on actual requirements.

- Formula:

  $$
  x = x1 + x2 + biasOptional
  $$

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
  $$

  - When the quantMode input is "static", the outputs outScales1Out and outScales2Out have no practical meaning. Depending on the divMode input, the fused quantization operator may be Quantize or AscendQuantV2.

    - When divMode is set to true, the fused quantization operator is Quantize, with the following formula:

      $$
      y1Out = round(y / scales1Optional + zeroPoints1Optional)
      $$

      $$
      y2Out = round(y / scales2Optional + zeroPoints2Optional), \quad \text{ if and only if scales2Optional exists}
      $$

    - When divMode is set to false, the fused quantization operator is AscendQuantV2, and the formula is as follows:

      $$
      y1Out = round(y * scales1Optional + zeroPoints1Optional)
      $$

      $$
      y2Out = round(y * scales2Optional + zeroPoints2Optional), \quad \text{ if and only if scales2Optional exists}
      $$

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

      where row\_max denotes the maximum value of each row.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnAddLayerNormQuantV2GetWorkspaceSize` API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the `aclnnAddLayerNormQuantV2` API to perform the computation.

```Cpp
aclnnStatus aclnnAddLayerNormQuantV2GetWorkspaceSize(
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
  aclTensor       *layernormRes,
  aclTensor       *outScales1Out,
  aclTensor       *outScales2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNormQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormQuantV2GetWorkspaceSize

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
      <td>Input of the addition in AddLayerNorm. The operator computes x1 + x2 + biasOptional and performs layer normalization on the result. Corresponds to `x1` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>When quantMode = "static", the shape supports 1-8 dimensions, and the data type supports FLOAT32, FLOAT16, and BFLOAT16.</li><li>When quantMode = "dynamic", the shape supports 2-8 dimensions, and the data type supports FLOAT16 and BFLOAT16.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Input of the addition in AddLayerNorm. The operator computes x1 + x2 + biasOptional and performs layer normalization on the result. Corresponds to `x2` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape need to be consistent with `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the gamma parameter in layer normalization. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When quantMode = "static", the shape supports 1-8 dimensions.</li><li>When quantMode = "dynamic", the shape supports 2-8 dimensions.</li><li>The data type needs to be consistent with `x1`.</li><li>The data dimensions need to be the same as the last few dimensions of `x1` or be [1, the last dimension of x1]. In per-tensor mode, the shape is [1, the last dimension of x1].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Corresponds to beta in the LayerNorm formula, representing the beta parameter in layer normalization. Corresponds to `beta` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>When quantMode = "static", the shape supports 1-8 dimensions.</li><li>When quantMode = "dynamic", the shape supports 2-8 dimensions.</li><li>The data type needs to be consistent with `x1`.</li><li>The shape needs to be consistent with `gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input. Pass an aclTensor that meets the constraints below, or use nullptr as a placeholder to indicate that this optional input does not exist. Represents the input for the addition computation in AddLayerNorm. The operator computes x1 + x2 + biasOptional internally and performs layer normalization on the result. Corresponds to `biasOptional` in the formula.</td>
      <td><ul><li>Empty tensor is supported.</li><li>When quantMode = "static", shape supports 1-8 dimensions.</li><li>When quantMode = "dynamic", shape supports 2-8 dimensions.</li><li>The data type needs to be consistent with `x1`.</li><li>The shape can be the same as `gamma` or `x1`. In per-tensor mode, the shape is the same as `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales1Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input, indicating the scale/smooth input of the first quantization operator to be fused. Corresponding to `scales1Optional` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape can remain consistent with `gamma` or be [1]; the shape is [1] in per-tensor mode.</li><li>When this optional input is passed, see <a href="#constraints">Constraints</a> for the data type and value constraint.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales2Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input, indicating the scale/smooth input of the second quantization operator to be fused. Corresponding to `scales2Optional` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape need to be consistent with `scales1Optional`.</li><li>For value constraints when optional input parameters are passed in, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints1Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, indicating the zeroPoints input in the first fused quantization operator, valid only when quantMode = "static". Corresponds to `zeroPoints1Optional` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape need to be consistent with `scales1Optional`.</li><li>For value constraints when optional input parameters are passed in, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints2Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, representing the zeroPoints input of the second fused quantization operator. Valid only when quantMode = "static". Corresponds to `zeroPoints2Optional` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>The data type and shape need to remain consistent with `scales1Optional`.</li><li>When this optional input parameter is passed in, see <a href="#constraints">Constraints</a> for the value constraint.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode (char*)</td>
      <td>Input</td>
      <td>Quantization mode, which determines whether the fused operator is a static or dynamic quantization operator. Corresponding to `quantMode` in the formula description. The value can be "static" or "dynamic".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Represents the epsilon in the corresponding LayerNorm, a value added to the denominator to ensure numerical stability. It corresponds to `epsilon` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additionalOutput (bool)</td>
      <td>Input</td>
      <td>Represents whether to enable the output of x=x1+x2+biasOptional.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode (bool)</td>
      <td>Input</td>
      <td>Valid only when quantMode = "static". Indicates whether scale is processed by multiplication or division in static quantization. When true is passed in, the operator performs division on scale in the quantization computation. Corresponds to `divMode` in the formula description.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of quantizing y, the output of LayerNorm, with the first quantization operator. It corresponds to `y1Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>shape needs to be consistent with input x1/x2.</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of quantizing the LayerNorm result output y by the second quantization operator. Corresponds to `y2Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape needs to be consistent with that of the inputs x1/x2.</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the output x of the Add operation. It corresponds to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape need to be consistent with the inputs `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>layernormRes (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the layernorm output layernormRes. It corresponds to `y` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape need to be consistent with input `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outScales1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Specifies the outScale result output of the first dynamic quantization computation, valid only when quantMode="dynamic". Corresponds to `outScales1Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is the shape of input `x1` with the last dimension removed.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outScales2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the outScale result output of the second dynamic quantization, valid only when quantMode="dynamic". Corresponding to `outScales2Out` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape is the shape of input `x1` with the last dimension removed.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation process of the operator.</td>
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
      <td>A passed-in required input, output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The value of quantMode is not "static" or "dynamic".</td>
    </tr>
    <tr>
      <td>The input data type combination is invalid. For valid data type combinations, see the Constraints section below.</td>
    </tr>
    <tr>
      <td>The shapes of x1, gamma, and outScales1Out meet any of the following conditions:
      <ol>
      <li>The dimensions of gamma do not match the trailing dimensions of x1.</li>
      <li>When the quantization mode is dynamic, that is, the value of the input quantMode is "dynamic", the number of dimensions of x1 is less than 2, or the number of dimensions of gamma is 1.</li>
      <li>When the quantization mode is dynamic, that is, the value of the input quantMode is "dynamic", the shape of outScales1Out is not the shape of the input x1 with the last dimension removed.</li></ol></td>
    </tr>
    <tr>
      <td>The shapes of all input tensors satisfy any of the following equality relations:
      <ol>
      <li>The shapes of x1, x2, xOut, layernormRes, and y1 are not the same; when the optional input scales2Optional exists, this condition is tightened to the shapes of x1, x2, xOut, layernormRes, y1, and y2 being not the same.</li>
      <li>The shapes of gamma and beta are not the same.</li>
      <li>When biasOptional exists, its shape is different from both gamma and x1.</li>
      <li>When the quantization mode is dynamic, that is, the value of the input quantMode is "dynamic", and the optional input scales2Optional exists at the same time, the shape of outScales1Out is different from that of outScales2Out.</li>
      </ol>
      </td>
    </tr>
    <tr>
      <td>The presence of the optional inputs (scales1Optional, scales2Optional, zeroPoints1Optional, zeroPoints2Optional) does not meet the specific combination relationship.</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNormQuantV2

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained by the first-phase API aclnnAddLayerNormQuantV2GetWorkspaceSize.</td>
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
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Product constraints:

  <!-- npu="A3,910b" id7 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - In the current scenario, `quantMode`, `additionalOutput`, and `divMode` only support the values `static`, `false`, and `false`, respectively.
    - In the current scenario, only empty `scales2Optional` and `zeroPoints2Optional` are supported.
    - In the current scenario, only the per-tensor quantization mode is supported, that is, the shape of `scales1Optional` and `zeroPoints1Optional` is [1], and the shape of `gamma` and `beta` is [1, the last dimension of x1].
    - In the current scenario, the data type of `scales1Optional` and `zeroPoints1Optional` must remain consistent with that of `x1`.
  <!-- end id7 -->

  <!-- npu="310p" id8 -->
  - <term>Atlas inference products</term>:
    - In the current scenario, `quantMode`, `additionalOutput`, and `divMode` only support `static`, `false`, and `false`, respectively.
    - In the current scenario, `scales2Optional` and `zeroPoints2Optional` only support empty.
    - In the current scenario, only the per-tensor quantization mode is supported, that is, the shape of `scales1Optional` and `zeroPoints1Optional` is [1], and the shape of `gamma` and `beta` is [1, the last dimension of x1].
    - In the current scenario, the data type of `scales1Optional` and `zeroPoints1Optional` must remain consistent with that of `x1`.
    - In the current scenario, the BFLOAT16 type is not supported.
    - The norm axis length of input parameters `x1` and `x2` must be greater than or equal to 32 bytes.
  <!-- end id8 -->

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

    - `T` indicates that the optional input exists or the combination mode is valid.
    - `F` indicates that the optional input does not exist or the combination is invalid.
    - `X` indicates that either case is valid.
- Data type support description:

  - When `quantMode` is "static":

    | Data Type Of x1 | x2 Data Type | Gamma Data Type | Beta Data Type | Data Type Of Bias | Data Type Of scale1 | Data Type Of scale2 | Data Type Of zeroPoints1 | Data Type Of zeroPoints2 | Data Type Of y1 | Data Type Of y2 | Data Type Of X | Data Type Of layernormRes | Data Type Of outScale1 | Data Type Of outScale2 |
    | ---------- | --------- | ------------- | ----------- | ------------ | -------------- | -------------- | ------------------ | -------------------| ------------------- | --------- | ---------- | --------- | ----------------- | :--------------- |
    | FLOAT16    | FLOAT16   | FLOAT16    | FLOAT16    | FLOAT16    | FLOAT16        | FLOAT16        | FLOAT16            | FLOAT16         |     INT8      | INT8       | FLOAT16     | FLOAT16       | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16   | BFLOAT16   | BFLOAT16   | BFLOAT16       | BFLOAT16       | BFLOAT16           | BFLOAT16         |     INT8      | INT8       | BFLOAT16    | BFLOAT16      | FLOAT32           | FLOAT32          |
    | FLOAT32    | FLOAT32   | FLOAT32    | FLOAT32    | FLOAT32    | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32          |     INT8      | INT8       | FLOAT32     | FLOAT32       | FLOAT32           | FLOAT32          |
    | FLOAT16    | FLOAT16   | FLOAT16    | FLOAT16    | FLOAT16    | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32          |     INT8      | INT8       | FLOAT16     | FLOAT16       | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16   | BFLOAT16   | BFLOAT16   | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32          |     INT8      | INT8       | BFLOAT16    | BFLOAT16       | FLOAT32           | FLOAT32          |

  - When `quantMode` is "dynamic":

    | Data Type Of x1 | x2 Data Type | Gamma Data Type | Beta Data Type | Data Type Of Bias | Data Type Of scale1 | Data Type Of scale2 | Data Type Of zeroPoints1 | Data Type Of zeroPoints2 | Data Type Of y1 | Data Type Of y2 | Data Type Of X | Data Type Of layernormRes | Data Type Of outScale1 | Data Type Of outScale2 |
    | ---------- | --------- | ------------- | ----------- | ------------ | -------------- | -------------- | ------------------ |     ------------------- | ------------------- | --------- | ---------- | --------- | ----------------- | :--------------- |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT16        | FLOAT16      | FLOAT16         | FLOAT16           |     INT8      | INT8       | FLOAT16     | FLOAT16           | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | BFLOAT16       | BFLOAT16     | BFLOAT16        | BFLOAT16          |     INT8      | INT8       | BFLOAT16    | BFLOAT16          | FLOAT32           | FLOAT32          |

- deterministic computation:

  - aclnnAddLayerNormQuantV2 defaults to a deterministic implementation.

## Example

The sample code is as follows and is for reference only. For details about how to compile and run the sample, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm_quant_v2.h"

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

uint16_t FloatToFp16(float value) {
  uint32_t x;
  memcpy(&x, &value, sizeof(x));
  uint16_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = x & 0x7FFFFF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | (mantissa >> 13);
}

float Fp16ToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    f = sign;
  } else if (exp == 31) {
    f = sign | 0x7F800000 | (mantissa << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
  }
  float result;
  memcpy(&result, &f, sizeof(result));
  return result;
}

std::vector<uint16_t> MakeFp16Data(size_t size, float value) {
  uint16_t fp16Val = FloatToFp16(value);
  return std::vector<uint16_t>(size, fp16Val);
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // boilerplate, resource initialization
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
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs, which need to be custom-constructed according to the API definition. In this example, a case without the optional bias input and a case with the bias input are each called once.
  float eps = 1e-6;
  bool additionalOut = true;
  bool divMode = false;
  const char* quantMode = "static";

  std::vector<int64_t> xShape = {1, 32};
  std::vector<int64_t> gammaShape = {1, 32};
  std::vector<int64_t> scaleShape = {1};

  std::vector<int64_t> reduceShape = {1};

  void *x1DeviceAddr = nullptr;
  void *x2DeviceAddr = nullptr;
  void *betaDeviceAddr = nullptr;
  void *gammaDeviceAddr = nullptr;
  void *biasDeviceAddr = nullptr;
  void *s1DeviceAddr = nullptr;
  void *z1DeviceAddr = nullptr;

  // Device address of the output for the case without bias.
  void *y1DeviceAddr = nullptr;
  void *y2DeviceAddr = nullptr;
  void *xDeviceAddr = nullptr;
  void *layernormResDeviceAddr = nullptr;
  void *outScales1DeviceAddr = nullptr;
  void *outScales2DeviceAddr = nullptr;

  aclTensor *x1 = nullptr;
  aclTensor *x2 = nullptr;
  aclTensor *beta = nullptr;
  aclTensor *gamma = nullptr;
  aclTensor *bias = nullptr;
  aclTensor *s1 = nullptr;
  aclTensor *z1 = nullptr;

  // aclTensor for the case without bias.
  aclTensor *y1 = nullptr;
  aclTensor *y2 = nullptr;
  aclTensor *x = nullptr;
  aclTensor *layernormRes = nullptr;
  aclTensor *outScales1 = nullptr;
  aclTensor *outScales2 = nullptr;

  int64_t xShapeSize = GetShapeSize(xShape);
  int64_t gammaShapeSize = GetShapeSize(gammaShape);
  int64_t scaleShapeSize = GetShapeSize(scaleShape);
  int64_t reduceShapeSize = GetShapeSize(reduceShape);

  std::vector<uint16_t> x1HostData = MakeFp16Data(xShapeSize, 1.0f);
  std::vector<uint16_t> x2HostData = MakeFp16Data(xShapeSize, 1.0f);
  std::vector<uint16_t> gammaHostData = MakeFp16Data(gammaShapeSize, 1.0f);
  std::vector<uint16_t> betaHostData = MakeFp16Data(gammaShapeSize, 1.0f);
  std::vector<uint16_t> biasHostData = MakeFp16Data(gammaShapeSize, 1.0f);

  std::vector<uint16_t> s1HostData = MakeFp16Data(scaleShapeSize, 1.0f);
  std::vector<uint16_t> z1HostData = MakeFp16Data(scaleShapeSize, 1.0f);

  // HostData for the case without bias.
  std::vector<int8_t> y1HostData(xShapeSize, 0);
  std::vector<int8_t> y2HostData(xShapeSize, 0);
  std::vector<uint16_t> xHostData(xShapeSize, 0);
  std::vector<uint16_t> layernormResHostData(xShapeSize, 0);
  std::vector<float> outScales1HostData(reduceShapeSize, 0);
  std::vector<float> outScales2HostData(reduceShapeSize, 0);

  // Create the self aclTensor.
  ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaHostData,  gammaShape, & betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, gammaShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(s1HostData, scaleShape, &s1DeviceAddr, aclDataType::ACL_FLOAT16, &s1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(z1HostData, scaleShape, &z1DeviceAddr, aclDataType::ACL_FLOAT16, &z1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the aclTensor without bias.
  ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(layernormResHostData, xShape, &layernormResDeviceAddr, aclDataType::ACL_FLOAT16, &layernormRes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales1HostData, reduceShape, &outScales1DeviceAddr, aclDataType::ACL_FLOAT, &outScales1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales2HostData, reduceShape, &outScales2DeviceAddr, aclDataType::ACL_FLOAT, &outScales2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnAddLayerNormQuantV2 API call example

  // 1. Example without the optional input bias
  // Call the first-phase API of aclnnAddLayerNormQuantV2.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  ret = aclnnAddLayerNormQuantV2GetWorkspaceSize(x1, x2, gamma, beta, nullptr, s1, nullptr, z1, nullptr, quantMode, eps, additionalOut, divMode, y1, y2, x, layernormRes, outScales1, outScales2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 2. Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 3. Call the second-phase API of aclnnAddLayerNormQuantV2.
  ret = aclnnAddLayerNormQuantV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuantV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. The code needs to be modified based on the actual API definition.

  auto y1Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY1(y1Size, 0);
  ret = aclrtMemcpy(resultDataY1.data(), resultDataY1.size() * sizeof(resultDataY1[0]), y1DeviceAddr, y1Size * sizeof(resultDataY1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuantV2 y1 output\n");
  for (int64_t i = 0; i < y1Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY1[i]);
  }

  auto y2Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY2(y2Size, 0);
  ret = aclrtMemcpy(resultDataY2.data(), resultDataY2.size() * sizeof(resultDataY2[0]), y2DeviceAddr, y2Size * sizeof(resultDataY2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuantV2 y2 output\n");
  for (int64_t i = 0; i < y2Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY2[i]);
  }

  auto xSize = GetShapeSize(xShape);
  std::vector<uint16_t> resultDataX(xSize, 0);
  ret = aclrtMemcpy(resultDataX.data(), resultDataX.size() * sizeof(resultDataX[0]), xDeviceAddr, xSize * sizeof(resultDataX[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuantV2 x output\n");
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultDataX[i]));
  }

  auto layernormResSize = GetShapeSize(xShape);
  std::vector<uint16_t> resultDataLayernormRes(layernormResSize, 0);
  ret = aclrtMemcpy(resultDataLayernormRes.data(), resultDataLayernormRes.size() * sizeof(resultDataLayernormRes[0]), layernormResDeviceAddr, layernormResSize * sizeof(resultDataLayernormRes[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuantV2 layernormRes output\n");
  for (int64_t i = 0; i < layernormResSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultDataLayernormRes[i]));
  }

  // 6. Release aclTensor and aclScalar. The code needs to be modified based on the actual API definition.
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(beta);
  aclDestroyTensor(gamma);
  aclDestroyTensor(bias);
  aclDestroyTensor(s1);
  aclDestroyTensor(z1);

  aclDestroyTensor(y1);
  aclDestroyTensor(y2);
  aclDestroyTensor(x);
  aclDestroyTensor(layernormRes);
  aclDestroyTensor(outScales1);
  aclDestroyTensor(outScales2);

  // 7. Release device resources. The code needs to be modified based on the actual API definition.
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(s1DeviceAddr);
  aclrtFree(z1DeviceAddr);

  aclrtFree(y1DeviceAddr);
  aclrtFree(y2DeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(layernormResDeviceAddr);
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
