# aclnnAddbmm&aclnnInplaceAddbmm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:42:43.106Z -->

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

- Description: First performs matrix multiplication of batch1 and batch2, then adds the results in batches along the first dimension (batch dimension), compressing the 3D vector into a 2D vector (with the shape size being the last two dimensions). The result is then multiplied by α, and summed with the product of β and self to obtain the final result.

- Formula:

  $$
  out = βself+α(\sum_{i=0}^{b-1} batch1_{i}@batch2_{i})
  $$

  Note: If β is 0, self is ignored and does not participate in the calculation. If α is 0, batch1 and batch2 are ignored and do not participate in the calculation.

- Example:

  The shape of self is [3,5], the shape of batch1 is [10,3,4], the shape of batch2 is [10,4,5], and the shape of the computed output out is [3,5].

## Function Prototype

- aclnnAddbmm and aclnnInplaceAddbmm implement the same functionality. The differences are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnAddbmm: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceAddbmm: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnAddbmmGetWorkspaceSize" or "aclnnInplaceAddbmmGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation process. Then call the "aclnnAddbmm" or "aclnnInplaceAddbmm" API to perform the computation.

```cpp
aclnnStatus aclnnAddbmmGetWorkspaceSize(
  const aclTensor     *self,
  const aclTensor     *batch1,
  const aclTensor     *batch2,
  const aclScalar     *beta,
  const aclScalar     *alpha,
  aclTensor           *out,
  int8_t               cubeMathType,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnAddbmm(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  aclrtStream        stream)
```

```cpp
aclnnStatus aclnnInplaceAddbmmGetWorkspaceSize(
  aclTensor         *selfRef,
  const aclTensor   *batch1,
  const aclTensor   *batch2,
  const aclScalar   *beta,
  const aclScalar   *alpha,
  int8_t             cubeMathType,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnInplaceAddbmm(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnAddbmmGetWorkspaceSize

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
      <td>Input self in the formula.</td>
      <td><ul><li>The data type must be the same as that of out.</li>
      <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the shape of the last two dimensions of batch1@batch2 (note: only self can be broadcast to the same shape as the last two dimensions of batch1@batch2; the last two dimensions of batch1@batch2 cannot be broadcast to the same shape as self. For example, self:[3,5] and batch1@batch2:[1,1,1] will fail).</li>
      <li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-2 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch1</td>
      <td>Input</td>
      <td>Input batch1 in the formula.</td>
      <td><ul><li>The data types of this parameter, self, and batch2 must conform to the data type derivation rules (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li><li>Only 3D shapes are supported, and the first dimension of batch1 must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the batch dimension of batch2.</li><li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch2</td>
      <td>Input</td>
      <td>Input batch2 in the formula.</td>
      <td><ul><li>The data types of this parameter, self, and batch1 must conform to the data type derivation rules (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li><li>Only 3D shapes are supported, and the first dimension of batch2 must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the batch dimension of batch1.</li><li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Input β in the formula.</td>
      <td>The data type must be convertible to the deduced data type of self and batch1@batch2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Input α in the formula.</td>
      <td>The data type must be convertible to the derived data type of self and batch1@batch2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output out in the formula.</td>
      <td><ul><li>The data type must be consistent with self.</li><li>The shape must be consistent with the last two dimensions of batch1@batch2.</li><li>Empty tensors are supported.</li>
      </ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter processes the data type after deduction by default. The supported enum values are as follows: <ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports computation with reduced input data precision.</li>
        <li>2: USE_FP16, supports reducing the input to FLOAT16 for computation.</li>
        <li>3: USE_HF32, supports reducing the input to the data type HFLOAT32 for computation.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output Parameter</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output Parameter</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="950,A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: When the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation; when the input is of other data types, no processing is performed.
  <!-- end id7 -->
  <!-- npu="910,310p" id8 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 is not supported.

  <!-- end id8 -->

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
      <td>self, batch1, or batch2 is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of self, batch1, batch2, or out is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data type of self, batch1, batch2, or out does not meet the derivation rule.</td>
    </tr>
    <tr>
      <td>self cannot perform a broadcast operation with batch1@batch2.</td>
    </tr>
    <tr>
      <td>The first dimensions of batch1 and batch2 are not equal, or a broadcast operation cannot be performed.</td>
    </tr>
    <tr>
      <td>The dimensions of batch1 and batch2 are not 3D.</td>
    </tr>
    <tr>
      <td>The last dimension of batch1 is not equal to the second-to-last dimension of batch2.</td>
    </tr>
    <tr>
      <td>self and batch1@batch2 must be both empty tensors or both non-empty tensors.</td>
    </tr>
  </tbody>
  </table>

## aclnnAddbmm

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnAddbmmGetWorkspaceSize.</td>
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
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddbmmGetWorkspaceSize

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
      <td>Input and output tensor, that is, the input self and out in the formula.</td>
      <td><ul>
      <li>The data type must comply with the data type derivation rule of batch1@batch2 (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The shape must be the same as the shape of the last two dimensions of batch1@batch2.</li>
      <li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch1</td>
      <td>Input</td>
      <td>Input batch1 in the formula.</td>
      <td><ul><li>The data type of this parameter must satisfy the data type derivation rule together with the data types of self and batch2 (see the <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and the <a href="#constraints">Constraints</a>).</li><li>shape supports only three dimensions, and the first dimension batch1 must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the batch dimension of batch2.</li><li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batch2</td>
      <td>Input</td>
      <td>batch2 in the formula.</td>
      <td><ul><li>The data type of this parameter must satisfy the data type derivation rule together with the data types of self and batch1 (see the <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and the <a href="#constraints">Constraints</a>).</li><li>shape supports only three dimensions, and the first dimension batch must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the batch dimension of batch1.</li><li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Input β in the formula.</td>
      <td>The data type must be convertible to the data type derived from self and batch1@batch2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Input α in the formula.</td>
      <td>The data type must be convertible to the data type deduced from self and batch1@batch2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter processes the data type after deduction by default. The supported enumeration values are as follows: <ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports downgrading input data to a lower precision for computation.</li>
        <li>2: USE_FP16, supports downgrading input to FLOAT16 for computation.</li>
        <li>3: USE_HF32, supports downgrading input to data type HFLOAT32 for computation.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output Parameter</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output Parameter</td>
      <td>Returns the op Executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="950,A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: When the input data type is FLOAT32, it will be converted to FLOAT16 for computation; when the input is of other data types, no processing is performed.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation; when the input is of other data types, no processing is performed.
  <!-- end id9 -->
  <!-- npu="910,310p" id10 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 is not supported.

  <!-- end id10 -->

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
      <td>The passed-in selfRef, batch1, or batch2 is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of selfRef, batch1, or batch2 is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data type of selfRef, batch1, or batch2 does not meet the derivation rule.</td>
    </tr>
    <tr>
      <td>selfRef cannot be broadcast with batch1@batch2.</td>
    </tr>
    <tr>
      <td>The first dimensions of batch1 and batch2 are not equal, or cannot be broadcast against each other.</td>
    </tr>
    <tr>
      <td>The dimensions of batch1 and batch2 are not 3.</td>
    </tr>
    <tr>
      <td>The last dimension of batch1 and the second-to-last dimension of batch2 are not equal.</td>
    </tr>
    <tr>
      <td>selfRef and batch1@batch2 must both be empty tensors or both be non-empty tensors.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAddbmm

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
      <td>The workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size allocated on the Device, obtained by the first-phase API aclnnInplaceAddbmmGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation description:

  <!-- npu="910,310p" id11 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>: aclnnAddbmm&aclnnInplaceAddbmm default to a deterministic implementation.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnAddbmm&aclnnInplaceAddbmm default to a deterministic implementation.

  <!-- end id12 -->

<!-- npu="A3,910b" id13 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Do not support data type deduction when one of the batch1 and batch2 inputs is BFLOAT16 and the other is FLOAT or FLOAT16.
<!-- end id13 -->
<!-- npu="910,310p" id14 -->
- <term>Atlas training products</term>, <term>Atlas inference products</term>: The Cube unit does not support FLOAT32 computation. When the input is FLOAT32, set cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) to allow the API to internally cast to FLOAT16 for computation.

<!-- end id14 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addbmm.h"

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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs as required by the API.
  std::vector<int64_t> selfShape = {2, 4};
  std::vector<int64_t> batch1Shape = {1, 2, 3};
  std::vector<int64_t> batch2Shape = {1, 3, 4};
  std::vector<int64_t> outShape = {2, 4};
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
  // Create batch2 aclTensor.
  ret = CreateAclTensor(batch2HostData, batch2Shape, &batch2DeviceAddr, aclDataType::ACL_FLOAT, &batch2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create beta aclScalar.
  beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // Call the first-phase API of aclnnAddbmm.
  ret = aclnnAddbmmGetWorkspaceSize(self, batch1, batch2, beta, alpha, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAddbmm.
  ret = aclnnAddbmm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddbmm failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify it according to the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceAddbmm
  // step3: Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceAddbmm\n");
  // Call the first-phase API of aclnnInplaceAddbmm.
  ret = aclnnInplaceAddbmmGetWorkspaceSize(self, batch1, batch2, beta, alpha, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceAddbmm.
  ret = aclnnInplaceAddbmm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddbmm failed. ERROR: %d\n", ret); return ret);

  // step4 (boilerplate): Synchronously wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // step5: Obtain the output value and copy the result from the device side to the host side.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(batch1);
  aclDestroyTensor(batch2);
  aclDestroyScalar(alpha);
  aclDestroyScalar(beta);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(batch1DeviceAddr);
  aclrtFree(batch2DeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
