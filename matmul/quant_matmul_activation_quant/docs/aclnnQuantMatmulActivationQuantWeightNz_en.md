# aclnnQuantMatmulActivationQuantWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:49:58.961Z -->

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

- Description: fused quantized matrix multiplication, activation, and dynamic quantization computation. Currently, gelu (including gelu_tanh and gelu_erf) is supported as the activation, and the quantization mode is MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md). The minimum supported input dimension is 2 and the maximum is 6. Similar APIs include aclnnMm (matrix multiplication that only supports 2D tensors as input) and aclnnBatchMatMul (matrix multiplication that only supports 3D inputs, where the first dimension is the batch dimension). Building on these APIs, this API further fuses gelu activation and dynamic MX quantization to improve hardware computing efficiency.

- Formula:

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    Supports MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md). For combinations of input and output data types corresponding to different quantization modes, see [Constraints](#constraints).

    <details>

    <summary><strong>QuantMatmul MX quantization mode</strong></summary>

      $$
      matmulOut[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
      $$

      where gsM, gsN, and gsK represent groupSizeM, groupSizeN, and groupSizeK, respectively; x1Slice represents the vector with a length of groupSizeK in row m of x1, and x2Slice represents the vector with a length of groupSizeK in column n of x2; both are sliced along the K axis starting from j*groupSizeK, j takes values in the range [0, kLoops), kLoops = ceil(K / groupSizeK), and K is the length of the K axis; supports the last slice with a length less than groupSizeK. bias is included only in MX quantization mode.

    </details>

    <details>

    <summary><strong>Activation Formula</strong></summary>

      - gelu_tanh (high-performance approximation):

        $$
        activationOut=GELU(matmulOut)=matmulOut × Φ(matmulOut)=0.5 * matmulOut * (1 + tanh( \sqrt{2 / \pi} * (matmulOut + 0.044715 * matmulOut^{3})))
        $$

      - gelu_erf:

        $$
        activationOut=GELU(matmulOut)=0.5 * matmulOut * (1 + erf(matmulOut / \sqrt{2}))
        $$

    </details>

    <details>

    <summary><strong>Dynamic Quantization Formula</strong></summary>

      - Scenario 1, when scaleAlg is 0:
        - The input activationOut is grouped along the last axis by $k=32$ values, and each group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized into $\{mxscale1, \{P_i\}_{i=1}^{k}\},\space k = 32$

        $$
        shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
        mxscale = 2^{shared\_exp}\\
        P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space k\\
        $$

        - The quantized $P_{i}$ forms the output yOut at the positions of the corresponding $V_{i}$, and mxscale is output as yScaleOut according to the grouping on the last axis.

        - emax: The exponent of the maximum normal number of the corresponding data type.

            | DataType | Emax |
            | :-----------: | :--: |
            |  FLOAT4_E2M1  |  2   |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |

      - Scenario 2: When scaleAlg is 1, only FP8 types are involved:
        - The input activationOut is divided into blocks of $k = 32$ values along the last axis. A block scaling factor $S_{fp32}^b$ is computed for each block, and then all elements in the block are mapped to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block has fewer than $k = 32$ elements, the missing values are treated as 0 and the block is processed as a complete block.
        - Find the maximum absolute value in the block:

          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$

        - Map FP32 to within the range representable by the target data type FP8, where $Amax(DType)$ is the maximum value representable by the target precision:

          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$

        - Convert the block scale factor $S_{fp32}^b$ to the representable scale value $S_{ue8m0}^b$ in FP8 format.
        - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block floating-point scale factor $S_{fp32}^b$.
        - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:

          $$
          E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a subnormal number, and} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{ otherwise} \end{cases}
          $$

        - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
        - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - Apply to the final quantization step. For each element within a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ represents the block scale factor, i.e., $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ represents the quantized data within the block.

      - Use case 3: when scaleAlg is 2, only the FP4_E2M1 type is involved:
        - When dstTypeMax = 0.0/6.0/7.0:
          - Group the input x along the axis dimension into groups of k = blocksize elements, and each group of k numbers  $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize:
          $$
          shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{ if} the one/two bits preceding the high bit of the mantissa \text{ are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{ otherwise} \end{cases} \\
          $$
          $$
          P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
          $$
          - ​The quantized $P_{i}$ form the output yOut according to the positions of the corresponding $V_{i}$, and mxscale forms the output mxscaleOut according to the grouping along the corresponding axis dimension.
        - When dstTypeMax != 0.0/6.0/7.0:
          - Divides the long vector into blocks, each of length k, and computes a block scale factor $S_{fp32}^b$ separately for each block. Then all elements in a block are mapped to the target low-precision type using the same $S_{fp32}^b$. If the last block contains fewer than k elements, the missing values are treated as 0 and the block is processed as a complete one.
          - Find the maximum absolute value in the block:
          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$
          - Maps FP32 to the range representable by the target data type, where $Amax(DType)$ is the maximum value representable by the target precision when dst_max_value=0, and the value passed in for dst_max_value when dst_max_value!=0.
          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$
          - Converts the block scale factor $S_{fp32}^b$ into the scale value $S_{ue8m0}^b$ representable in FP8 format.
          - Extracts the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scale factor $S_{fp32}^b$.
          - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:
            $$
            E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{ otherwise} \end{cases}
            $$
          - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
          - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
          - Apply to the final quantization step. For each element within a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ represents the block scale factor, i.e., $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ represents the quantized data within the block.
          - ​The quantized $P_{i}$ form the output yOut according to the positions of the corresponding $V_{i}$, and mxscale forms the output mxscaleOut according to the grouping on the corresponding axis dimension.

    </details>

  <!-- end id7 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize" API to obtain the workspace size required for the calculation and the executor that includes the calculation process of the operator, and then call the "aclnnQuantMatmulActivationQuantWeightNz" API to perform the calculation.

```cpp
aclnnStatus aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *x1ScaleOptional,
  const aclTensor *x2Scale,
  const aclTensor *biasOptional,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  char            *activationType,
  char            *quantMode,
  char            *roundMode,
  int64_t          scaleAlg,
  double           dstTypeMax,
  aclTensor       *yOut,
  aclTensor       *yScaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulActivationQuantWeightNz(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 248px">
  <col style="width: 121px">
  <col style="width: 210px">
  <col style="width: 327px">
  <col style="width: 250px">
  <col style="width: 115px">
  <col style="width: 138px">
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
        <td>x1</td>
        <td>Input</td>
        <td>Input x1 in the formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Supports <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensors</a> where the last two axes are transposed; non-contiguous tensors in other axis directions are not supported.</li>
            <li>When transposeX1 is false, each dimension of x1 is (batch, m, k), where batch may not exist.</li>
            <li>When transposeX1 is true, each dimension of x1 is (batch, k, m), where batch may not exist.</li>
            <li>When the data types of x1, x2, and y are all FLOAT4_E2M1 (MXFP4 use case), x1 does not support transposition, that is, transposeX1 cannot be true.</li>
          </ul>
        </td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td>          <ul>
            <li>Does not support empty tensors.</li>
            <li>In NZ format, the shape supports 4-8 dimensions.</li>
            <li>When transposeX2 is true, each dimension of x2 indicates (batch, k1, n1, n0, k0), where batch may not exist, k0 = 32, and n0 = 16.</li>
            <li>When transposeX2 is false, each dimension of x2 indicates (batch, n1, k1, k0, n0), where batch may not exist, k0 = 16, and n0 = 32.</li>
            <li>k in the x1 shape and k1 in the x2 shape must satisfy ceil(k / k0) = k1, and n1 in the x2 shape and n in y must satisfy ceil(n / n0) = n1.</li>
            <li>Use the aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight APIs to convert the input format from ND to NZ.</li>
          </ul>
        </td>
        <td>FLOAT8_E4M3FN, FLOAT4_E2M1</td>
        <td>NZ</td>
        <td>4-8</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x1ScaleOptional</td>
        <td>Optional input</td>
        <td>Input x1Scale in the formula.</td>
        <td>          <ul>
            <li>Does not support empty tensors.</li>
            <li>When x1Scale is FLOAT8_E8M0, x1Scale is 3D. Each dimension indicates: (m, ceil(k / 64), 2) if transposeX1 is false, and (ceil(k / 64), m, 2) if transposeX1 is true.</li>
          </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3-7</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x2Scale</td>
        <td>Input</td>
        <td>Quantization parameter, the input x2Scale in the formula.</td>
        <td>          <ul>
            <li>Does not support empty tensors.</li>
            <li>When x2Scale is FLOAT8_E8M0, x2Scale is 3-dimensional, and each dimension is described as follows: (ceil(k / 64), n, 2) if transposeX2 is false, and (n, ceil(k / 64), 2) if transposeX2 is true.</li>
          </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3-7</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Optional input</td>
        <td>The input bias in the formula.</td>
        <td>          <ul>
            <li>Does not support empty tensors.</li>
            <li>Supports 1D (n,) or 3D (batch, 1, n) shape, where n is the same as that of x2.</li>
            <li>When the shape of y is 2D, 4D, 5D, or 6D, the shape of bias only supports 1D (n,).</li>
            <li>If this parameter is not used, pass nullptr.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1, 3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX1</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x1 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x2 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize</td>
        <td>Optional input</td>
        <td>Used for the quantization group sizes along the m, n, and k directions of the input.</td>
        <td>Composed of the 3 values groupSizeM, groupSizeN, and groupSizeK concatenated together. Each value occupies 16 bits, taking up the lower 48 bits of the int64_t type groupSize (the value of the upper 16 bits in groupSize is invalid). For the calculation formula, see Formula 1 below the table. For use cases that do not support groupSize, pass in 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activationType</td>
        <td>Input</td>
        <td>Activation type.</td>
        <td>Supports {"gelu_tanh", "gelu_erf"}.</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>Input</td>
        <td>Quantization mode.</td>
        <td>Supports {"mx"}.</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>roundMode</td>
        <td>Input</td>
        <td>Indicates the data conversion mode, corresponding to round_mode in the formula.</td>
        <td>          <ul>
            <li>If yDtype is FLOAT4_E2M1, {"rint", "floor", "round"} are supported.</li>
            <li>If yDtype is FLOAT8_E4M3FN/FLOAT8_E5M2, only {"rint"} is supported.</li>
          </ul>
        </td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleAlg</td>
        <td>Input</td>
        <td>Indicates the calculation method of mxscaleOut, corresponding to scaleAlg in the formula.</td>
        <td>          <ul>
            <li>Supports values 0, 1, and 2. The value 0 represents scenario 1, 1 represents scenario 2, and 2 represents scenario 3.</li>
            <li>When yDtype is FLOAT4_E2M1, only values 0 and 2 are supported.</li>
            <li>When yDtype is FLOAT8_E4M3FN/FLOAT8_E5M2, values 0 and 1 are supported.</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstTypeMax</td>
        <td>Input</td>
        <td>Represents the value of maxType, corresponding to Amax(DType) in the formula.</td>
        <td>          <ul>
            <li>When scaleAlg is 0 or 1, dstTypeMax does not take effect and no check is performed. The recommended value is 0.0.</li>
            <li>When scaleAlg is 2, dstTypeMax supports values 0.0 and 6.0-12.0. The value 0.0 indicates that the default maximum value of the target type is used, and a value of 6.0-12.0 indicates that the specified value is used.</li>
          </ul>
        </td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
            <tr>
        <td>yOut</td>
        <td>Output</td>
        <td>Output y in the formula, that is, the matmul and activation calculation result after dynamic quantization.</td>
        <td>          <ul>
            <li>Does not support empty Tensor.</li>
            <li>The shape is the same as that of the matmul calculation result, (batch, m, n); batch may not exist.</li>
          </ul>
        </td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2-6</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yScaleOut</td>
        <td>Output</td>
        <td>Quantization scale corresponding to each group after dynamic quantization.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The value on the last axis is the value of the corresponding axis of y divided by 32 and rounded up, and is padded to an even number with a pad value of 0.</li>
            <li>The yScale output needs to be interleaved every two rows of data.</li>
            <li>The shape is the same as that of the matmul computation result, (batch, m, ceil(n/64), 2), and batch may not exist.</li>
          </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-8</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be applied on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor</td>
        <td>Output</td>
        <td>Returns the op executor, which includes the calculation process of the operator.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

  - Note: An optional input refers to an optional quantization parameter, and passing a null pointer is supported.

  - Formula 1: <a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

    - x2 supports the [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) where the last two axes are transposed; the [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) in other scenarios is not supported.
    - Passing a non-zero value for groupSize is supported.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
      <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x1, x2, x1Scale, x2Scale, yOut, or yScaleOut is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>The data type and data format of x1, x2, biasOptional, x1ScaleOptional, x2Scale, yOut, or yScaleOut are not within the supported range.</td>
      </tr>
      <tr>
        <td>The shape of x1, x2, biasOptional, x1ScaleOptional, x2Scale, yOut, or yScaleOut does not meet the validation conditions.</td>
      </tr>
      <tr>
        <td>x1, x2, biasOptional, x1ScaleOptional, x2Scale, yOut, or yScaleOut is an empty tensor.</td>
      </tr>
      <tr>
        <td>The value of activationType, quantMode, roundMode, scaleAlg, dstTypeMax, groupSize, or transposeX1 is not within the supported range.</td>
      </tr>
    </tbody></table>

## aclnnQuantMatmulActivationQuantWeightNz

- **Parameters**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
    <td>Workspace memory address applied on the Device side.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Op executor that includes the calculation flow of the operator.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream for executing tasks.</td>
  </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnQuantMatmulActivationQuantWeightNz defaults to a deterministic implementation.

- **General Constraints:**
  <a id="general-constraints"></a>
  - x1 only supports the ND format, and x2 only supports the NZ format.
  - When k or n is 1, the weightNz feature cannot be used. This API does not support this use case.
  - Before calling this API, you can convert x2 from ND format to the NZ format by calling [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) or [aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md). During use, the data must be padded with 0 to prevent dirty data.
  - When transposeX1 is false, the shape of x1 is (batch, m, k). When transposeX1 is true, the shape of x1 is (batch, k, m). Here, batch represents the first 0 to 4 dimensions, where 0 dimension indicates that batch does not exist.
  - Shape of x2 when transposeX2 is false: (batch, n1, k1, k0, n0). Shape of x2 when transposeX2 is true: (batch, k1, n1, n0, k0). batch represents the preceding 0-4 dimensions, and 0 indicates that batch does not exist. k is the same as k in the shape of x1.
  - x1 supports the non-contiguous tensor with the last two axes transposed. The non-contiguous tensor in other use cases is not supported.
  - x2 supports the non-contiguous tensor with the last two axes transposed. The non-contiguous tensor in other use cases is not supported.
  - If the data type of x1 is FLOAT8_E5M2, the data type of x2 must be FLOAT8_E4M3FN.
  - bias-related constraints:
    - Optional parameter. Passing in a null pointer is supported, but an empty tensor is not supported.
    - The data type must be FLOAT32.
    - The shape can be 1D (n,) or 3D (batch, 1, n), where n is the same as n in x2.
    - If y has a 2D, 4D, 5D, or 6D shape, the shape of bias only supports 1D (n,).
  - Constraints on groupSize:
    - Takes effect only in the MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md).
    - The input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. Currently, only [0, 0, 0] or [1, 1, 32] is supported, and [0, 0, 0] is automatically deduced as [1, 1, 32].

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

  - The shape of y supports 2-6 dimensions, in the format of (batch, m, n). batch may not exist. m must be the same as the m of x1, and n must be the same as the n of x2.

  <details>
    <summary><strong>MX quantization use case constraints:</strong></summary>
    <a id="mx-quantization"></a>

    - The input and output support the following data type combinations:
    <a id="input-and-output-support-the-following-data-type-combinations-mx"></a>

        | x1 | x2 | x1Scale | x2Scale | Bias | y | yScale |
        |---------------|---------------|-------------|-------------|--------------|---------------------------|-------------|
        | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E4M3FN             | FLOAT8_E8M0 |
        | FLOAT8_E5M2   | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E5M2               | FLOAT8_E8M0 |
        | FLOAT4_E2M1   | FLOAT4_E2M1   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT4_E2M1               | FLOAT8_E8M0 |

    - Relationship among the x1 data type, x2 data type, x1, x2, x1Scale, x2Scale, and groupSize:

        | Quantization Type | x1 Data Type | x2 Data Type | x1 Shape | x2 Shape | x1Scale Shape | x2Scale Shape | Bias Shape | yScale Shape | [gsM, gsN, gsK] | groupSize |
        |-------|--------|--------|--------|--------|-------------|-------------|------------|---------------------------------------|--|--|
        |MX full quantization|FLOAT8_E4M3FN|FLOAT8_E4M3FN|<li>Non-transposed: (batch_x1, m, k)</li>|<li>Non-transposed: (batch_x2, k, n)</li><li>Transposed: (batch_x2, n, k)</li>|<li>Non-transposed: (batch_x1, m, ceil(k / 64), 2)</li>|<li>Non-transposed: (batch_x2, ceil(k / 64), n, 2)</li><li>Transposed: (batch_x2, n, ceil(k / 64), 2)</li>|(n,) or (batch_max, 1, n)|null|[1, 1, 32]|4295032864|
        |MX full quantization|FLOAT8_E5M2|FLOAT8_E4M3FN|<li>Non-transposed: (batch_x1, m, k)</li>|<li>Non-transposed: (batch_x2, k, n)</li><li>Transposed: (batch_x2, n, k)</li>|<li>Non-transposed: (batch_x1, m, ceil(k / 64), 2)</li>|<li>Non-transposed: (batch_x2, ceil(k / 64), n, 2)</li><li>Transposed: (batch_x2, n, ceil(k / 64), 2)</li>|(n,) or (batch_max, 1, n)|null|[1, 1, 32]|4295032864|
        |MX full quantization|FLOAT4_E2M1|FLOAT4_E2M1|<li>Non-transposed: (batch_x1, m, k)</li>|<li>Non-transposed: (batch_x2, k, n)</li><li>Transposed: (batch_x2, n, k)</li>|<li>Non-transposed: (batch_x1, m, ceil(k / 64), 2)</li>|<li>Non-transposed: (batch_x2, ceil(k / 64), n, 2)</li><li>Transposed: (batch_x2, n, ceil(k / 64), 2)</li>|(n,) or (batch_max, 1, n)|null|[1, 1, 32]|4295032864|

    - Note: In the table above, gsM, gsK, and gsN represent groupSizeM, groupSizeK, and groupSizeN, respectively.
    - In the MX quantization use case, the transpose attribute of x1 must be consistent with that of x1Scale, and the transpose attribute of x2 must be consistent with that of x2Scale.
    - The data type of yOut is determined by the data type of x1, and the two must be the same.
    - MXFP4 use case constraints (the data types of x1, x2, and y are all FLOAT4_E2M1):
      - x1 does not support transposition, that is, transposeX1 cannot be set to true.
      - scaleAlg only supports values 0 and 2.
      - If scaleAlg is 2, dstTypeMax supports values 0.0 and 6.0-12.0.
      - roundMode supports {"rint", "floor", "round"}.
      - The inner axis of x1 and x2 must be even, and k must be greater than 2.
      - The N dimension of y must be even.
  </details>

## Example

The following sample code is for reference only. For details about how to compile and run, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

x1 is FLOAT8_E4M3FN, x2 is FLOAT8_E4M3FN (NZ format), x1Scale is FLOAT8_E8M0, x2Scale is FLOAT8_E8M0, the activation function is gelu_tanh, and scaleAlg is 0 (use case 1).

  ```cpp
  #include <iostream>
  #include <memory>
  #include <cmath>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_quant_matmul_activation_quant_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)
  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
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
                      aclDataType dataType, aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      // Calculate the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      // Call the aclCreateTensor API to create an aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }
  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
      std::unique_ptr<const aclIntArray, aclnnStatus (*)(const aclIntArray*)> mat2SizePtr(mat2Size, aclDestroyIntArray);
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate device-side memory (aligned size)
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Copy only the original ND data
      auto originalSize = static_cast<size_t>(GetShapeSize(shape) * sizeof(T));
      ret = aclrtMemcpy(*deviceAddr, originalSize, hostData.data(), originalSize, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      // Clear the remaining area to prevent dirty data
      if (size > originalSize) {
          ret = aclrtMemset(static_cast<uint8_t*>(*deviceAddr) + originalSize, size - originalSize, 0, size - originalSize);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset failed. ERROR: %d\n", ret); return ret);
      }

      // Calculate the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor API to create an aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int AclnnQuantMatmulWeightNzActivationQuantTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. Construct the inputs and outputs as required by the API.
      int64_t m = 5;
      int64_t k = 64;
      int64_t n = 128;
      bool transposeX1 = false;
      bool transposeX2 = true;
      constexpr int64_t MX_SCALE_BLOCK_SIZE = 64; // Scale block alignment size, used for calculating the scale shape.
      std::vector<int64_t> x1Shape = {m, k};
      std::vector<int64_t> x2Shape = {n, k};
      std::vector<int64_t> x1ScaleShape = {m, (k + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, 2};
      std::vector<int64_t> x2ScaleShape = {n, (k + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, 2};
      std::vector<int64_t> outShape = {m, n};
      std::vector<int64_t> outScaleShape = {m, (n + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, 2};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      void* outScaleDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      aclTensor* outScale = nullptr;
      std::vector<uint8_t> x1HostData(m * k, 0b00111000);                  // 1.0 in float8_e4m3
      std::vector<uint8_t> x2HostData(n * k, 0b00111000);                  // 1.0 in float8_e4m3
      std::vector<uint8_t> x1ScaleHostData(m * (k + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE * 2, 0b01111111); // 1.0 in float8_e8m0
      std::vector<uint8_t> x2ScaleHostData(n * (k + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE * 2, 0b01111111); // 1.0 in float8_e8m0
      std::vector<uint8_t> outHostData(m * n, 0);
      std::vector<uint8_t> outScaleHostData(m * (n + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE * 2, 0);
      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor in the data layout format compatible with the AI processor.
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the mxscale aclTensor.
      ret = CreateAclTensor(outScaleHostData, outScaleShape, &outScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                            &outScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outScaleTensorPtr(outScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outScaleDeviceAddrPtr(outScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 3. Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // Call the first-phase API aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceTransAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceTransAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceTransAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceTransAddrPtr.reset(workspaceTransAddr);
      }
      // Call the second-phase API aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceTransAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      workspaceSize = 0;
      executor = nullptr;
      int64_t groupSizeValue = 4295032864;
      char activationType[] = "gelu_tanh";
      char quantMode[] = "mx";
      char roundMode[] = "rint";
      ret = aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize(
          x1, x2, x1Scale, x2Scale, bias, transposeX1, transposeX2, groupSizeValue, activationType, quantMode, roundMode,
          0, 0.0, out, outScale, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API aclnnQuantMatmulActivationQuantWeightNz.
      ret = aclnnQuantMatmulActivationQuantWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulActivationQuantWeightNz failed. ERROR: %d\n", ret);
                return ret);
      // 4. (Boilerplate) Synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<uint8_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }
      size = GetShapeSize(outScaleShape);
      std::vector<uint8_t> scaleData(size, 0);
      ret = aclrtMemcpy(scaleData.data(), scaleData.size() * sizeof(scaleData[0]), outScaleDeviceAddr,
                        size * sizeof(scaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("scale[%ld] is: %d\n", i, scaleData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) device/stream initialization. For details, see the acl API documentation.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnQuantMatmulWeightNzActivationQuantTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("AclnnQuantMatmulWeightNzActivationQuantTest failed. ERROR: %d\n", ret);
                    return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```
