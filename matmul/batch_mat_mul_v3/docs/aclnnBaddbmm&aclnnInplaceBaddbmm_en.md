# aclnnBaddbmm&aclnnInplaceBaddbmm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:00:18.696Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description:
Computes the product of α and the matrix multiplication result of batch1 and batch2, then adds the product of β and self.
Note: batch1 and batch2 must be 3D Tensors. Broadcast of the two shapes is supported only in aclnnBaddbmm and will be intercepted in aclnnInplaceBaddbmm;
self must be able to broadcast to the result shape of batch1@batch2. Broadcast forms such as N direction, M direction, scalar, and per-batch scalar are supported. Broadcast means extending a smaller shape to a larger shape so that the shapes of the two Tensors are compatible, that is, each corresponding dimension must be the same or one of them must be 1.

- Formula:

  $
  out = βself+α(batch1@batch2)
  $
  Note: If β is 0, self is ignored and does not participate in the computation.

- Example:

  The shape of self is [1, M, K], the shape of batch1@batch2 is [A, M, 1], and the shape of the output out is [A, M, K]. The numbers in each dimension must be identical or one of them must be 1. In this case, if the shape of self is [2, M, K], the broadcast condition is not met and an error is reported.

## Function Prototype

- aclnnBaddbmm and aclnnInplaceBaddbmm implement the same functionality. The differences are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnBaddbmm: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceBaddbmm: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnBaddbmmGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size according to the process, and then call the "aclnnBaddbmm" API to perform the computation.

```cpp
aclnnStatus aclnnBaddbmmGetWorkspaceSize(
  const aclTensor*    self,
  const aclTensor*    batch1,
  const aclTensor*    batch2,
  const aclScalar*    beta,
  const aclScalar*    alpha,
  aclTensor*          out,
  int8_t              cubeMathType,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnBaddbmm(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

```cpp
aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(
  const aclTensor*    selfRef,
  const aclTensor*    batch1,
  const aclTensor*    batch2,
  const aclScalar*    beta,
  const aclScalar*    alpha,
  int8_t              cubeMathType,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnInplaceBaddbmm(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  aclrtStream       stream)
```

## aclnnBaddbmmGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <td>The input self in the formula.</td>
      <td><ul><li>The data type needs to be consistent with out.</li>
      <li>The shape needs to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the shape of the last two dimensions of batch1@batch2 (Note: only self can be broadcast to the same shape as the last two dimensions of batch1@batch2. The last two dimensions of batch1@batch2 cannot be broadcast to the same shape as self. For example: self: [2, 3, 5], batch1@batch2: [1, 1, 1] causes an error.).</li>
      <li>Supports empty tensors.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-3 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch1</td>
      <td>Input</td>
      <td>The input batch1 in the formula.</td>
      <td><ul><li>The data type needs to satisfy the data type derivation rule with the data types of self and batch2 (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The shape needs to satisfy the bmm input constraint relationship with batch2.</li><li>Supports empty Tensor.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch2</td>
      <td>Input</td>
      <td>Input batch2 in the formula.</td>
      <td><ul><li>The data type needs to satisfy the data type derivation rule with the data types of self and batch1 (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The shape needs to satisfy the bmm input constraint relationship with batch1.</li><li>Supports empty Tensor.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>The input β in the formula.</td>
      <td>The data type needs to be convertible to the data type of self and batch1@batch2 after derivation (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>The input α in the formula.</td>
      <td>The data type needs to be convertible to the data type after derivation from self and batch1@batch2 (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the Formula.</td>
      <td><ul><li>The data type needs to keep consistent with self.</li><li>The shape needs to keep consistent with the last two dimensions of batch1@batch2.</li><li>The format needs to keep consistent with self and batch1@batch2.</li>
      </ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter processes the data type after derivation by default. The supported enumeration values are as follows:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports precision reduction of the input data for computation.</li>
        <li>2: USE_FP16, supports converting the input to FLOAT16 for computation.</li>
        <li>3: USE_HF32, supports converting the input to the data type HFLOAT32 for computation.</li>
        <li>4: USE_FP32_ADD, supports computation in high-precision mode.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 and 4 are not supported.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - cubeMathType=0: When the input data type is FLOAT16 or BFLOAT16, the matrix multiplication uses FLOAT16/BFLOAT16 input and FLOAT32 output for transmission. When the input is of other data types, no processing is performed.
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: When the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation; when the input is of other data types, no processing is performed.
    - cubeMathType=4: When self, batch1, and batch2 are all FLOAT16 or all BFLOAT16, the batched matrix multiplication and bias addition are computed using FLOAT32 intermediate results. Broadcasting self against the batched matrix multiplication result is supported. The output data type is specified by out.

  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: when the input data type is FLOAT32, it is converted to FLOAT16 for computation; when the input is of other data types, no processing is performed.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation; when the input is of other data types, no processing is performed.
    - No processing is performed when cubeMathType is set to 4.

  <!-- end id9 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
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
      <td>The input self, batch1, or batch2 is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of self, batch1, batch2, or out is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data format of self, batch1, batch2, or out is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self, batch1, batch2, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The data formats of self, batch1, batch2, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>self cannot be broadcast with batch1@batch2.</td>
    </tr>
    <tr>
      <td>self and batch1@batch2 must be empty tensors or non-empty tensors at the same time.</td>
    </tr>
    <tr>
      <td>The data types of self, batch1, batch2, and out do not satisfy the derivation rule.</td>
    </tr>
  </tbody>
  </table>

## aclnnBaddbmm

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>The workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnBaddbmmGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, includes the operator sub-computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceBaddbmmGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <td>selfRef</td>
      <td>Input|Output</td>
      <td>Input/output tensor, which is the input self and out in the formula.</td>
      <td><ul>
      <li>The data type and the data type of batch1@batch2 need to satisfy the data type derivation rule (See <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#Constraints">Constraints</a>).</li><li>The last two dimensions of shape need to be consistent with the last two dimensions of batch1@batch2.</li>
      <li>Support empty Tensor.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch1</td>
      <td>Input</td>
      <td>Input batch1 in the formula.</td>
      <td><ul><li>The data type needs to satisfy the data type derivation rule with the data types of self and batch2 (See <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The shape needs to satisfy the bmm input constraint relationship with batch2.</li><li>Supports empty tensors.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch2</td>
      <td>Input</td>
      <td>The input batch2 in the Formula.</td>
      <td><ul><li>The data type needs to satisfy the data type derivation rule with the data types of self and batch1 (See <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The shape needs to satisfy the bmm input constraint relationship with batch1.</li><li>Supports empty tensors.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>The input β in the formula.</td>
      <td>The data type needs to be convertible to the data type of self and batch1@batch2 after derivation (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Input α in the formula.</td>
      <td>The data type needs to be convertible to the data type of self and batch1@batch2 after derivation (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter processes the data type after derivation by default. The supported enumeration values are as follows:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports converting the input data to a lower precision for computation.</li>
        <li>2: USE_FP16, supports converting the input to FLOAT16 for computation.</li>
        <li>3: USE_HF32, supports converting the input to the data type HFLOAT32 for computation.</li>
        <li>4: USE_FP32_ADD, supports computation in high-precision mode.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator sub-computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id10 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 and 4 are not supported.
  <!-- end id10 -->
  <!-- npu="910,310p" id11 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 is not supported.
    - No processing is performed when cubeMathType is set to 4.
  <!-- end id11 -->
  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: When the input data type is FLOAT32, it will be converted to FLOAT16 for computation; when the input is of other data types, no processing is performed.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation; when the input is of other data types, no processing is performed.
    - cubeMathType=4: When selfRef, batch1, and batch2 are all FLOAT16 or all BFLOAT16, the batched matrix multiplication and bias addition process uses FLOAT32 intermediate results for computation.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: When the input data type is FLOAT32, it will be converted to FLOAT16 for computation; when the input is of other data types, no processing is performed.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation; when the input is of other data types, no processing is performed.
    - No processing is performed when cubeMathType is set to 4.

  <!-- end id13 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
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
      <td>The input selfRef, batch1, or batch2 is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of selfRef, batch1, or batch2 is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data format of selfRef, batch1, or batch2 is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of selfRef, batch1, and batch2 are inconsistent.</td>
    </tr>
    <tr>
      <td>selfRef cannot be broadcast with batch1@batch2.</td>
    </tr>
    <tr>
      <td>The first dimension of batch1 is not equal to that of batch2.</td>
    </tr>
    <tr>
      <td>The dimensions of batch1 and batch2 are not three-dimensional.</td>
    </tr>
    <tr>
      <td>The last dimension of batch1 is not equal to the second-to-last dimension of batch2.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceBaddbmm

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>The workspace memory address applied on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size applied on the device side, obtained by the first-phase API aclnnInplaceBaddbmmGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, includes the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
aclnnBaddbmm&aclnnInplaceBaddbmm default to a deterministic implementation.

<!-- npu="910,310p" id14 -->
- <term>Atlas training products</term>, <term>Atlas inference products</term>: The Cube unit does not support FLOAT32 computation. When the input is FLOAT32, set cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) to allow the API to internally cast to FLOAT16 for computation.
<!-- end id14 -->
<!-- npu="A3,910b" id15 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - Data type derivation is not supported when one of the two inputs batch1 and batch2 is BFLOAT16 and the other is FLOAT or FLOAT16.
  - In the precision promotion use case, currently only the following is supported: the data types of inputs batch1 and batch2 are both FLOAT16 or BFLOAT16, and the data type of out is FLOAT32.
  - In use cases where self does not need to be broadcast, inputs whose data formats of batch1, batch2, and out are all FRACTAL_NZ are supported. The internal implementation converts the NZ format to ND. Directly using inputs in ND format is recommended.

<!-- end id15 -->

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_baddbmm.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate, resource initialization
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
  // Call aclrtMalloc to apply for memory on the Device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy Host-side data to Device-side memory.
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

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API Reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Process the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {1, 2, 4};
  std::vector<int64_t> batch1Shape = {1, 2, 3};
  std::vector<int64_t> batch2Shape = {1, 3, 4};
  std::vector<int64_t> outShape = {1, 2, 4};
  void* selfDeviceAddr = nullptr;
  void* batch1DeviceAddr = nullptr;
  void* batch2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* batch1 = nullptr;
  aclTensor* batch2 = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* beta = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> batch1HostData = {1, 1, 1, 2, 2, 2};
  std::vector<float> batch2HostData = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  std::vector<float> outHostData(8, 0);
  int8_t cubeMathType = 1;
  float alphaValue = 1.2f;
  float betaValue = 1.0f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the batch1 aclTensor.
  ret = CreateAclTensor(batch1HostData, batch1Shape, &batch1DeviceAddr, aclDataType::ACL_FLOAT, &batch1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the batch2 aclTensor.
  ret = CreateAclTensor(batch2HostData, batch2Shape, &batch2DeviceAddr, aclDataType::ACL_FLOAT, &batch2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create the beta aclScalar.
  beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // Example of the aclnnBaddbmm API call.
  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnBaddbmm.
  ret = aclnnBaddbmmGetWorkspaceSize(self, batch1, batch2, beta, alpha, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBaddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnBaddbmm.
  ret = aclnnBaddbmm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBaddbmm failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize to wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the Device memory to the Host memory. Modify this step based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceBaddbmm API call example
  // 3. Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceBaddbmm\n");
  // Call the first-phase API aclnnInplaceBaddbmm.
  ret = aclnnInplaceBaddbmmGetWorkspaceSize(self, batch1, batch2, beta, alpha, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBaddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnInplaceBaddbmm.
  ret = aclnnInplaceBaddbmm(inplaceWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBaddbmm failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize to wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the Device memory to the Host. Modify the code based on the specific API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(batch1);
  aclDestroyTensor(batch2);
  aclDestroyScalar(alpha);
  aclDestroyScalar(beta);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(batch1DeviceAddr);
  aclrtFree(batch2DeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtFree(inplaceWorkspaceAddr);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
