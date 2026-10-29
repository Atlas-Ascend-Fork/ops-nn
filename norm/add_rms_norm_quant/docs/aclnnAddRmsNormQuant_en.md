# aclnnAddRmsNormQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:51:24.129Z -->

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

- Description: The RMSNorm operator is a common normalization operation in large models. Compared with the LayerNorm operator, it removes the part of subtracting the mean. The AddRmsNormQuant operator fuses the Add operator before RMSNorm and the Quantize operator after RMSNorm, reducing data movement operations.
- Formula:

  $$
  x_i={x1}_{i}+{x2}_{i}
  $$

  $$
  y_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - When divMode is true:

    $$
    y1Out=round((y/scales1)+zero\_points1)
    $$

    $$
    y2Out=round((y/scales2)+zero\_points2)
    $$

  - When divMode is false:

    $$
    y1Out=round((y*scales1)+zero\_points1)
    $$

    $$
    y2Out=round((y*scales2)+zero\_points2)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnAddRmsNormQuantGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnAddRmsNormQuant" API to execute the computation.

```Cpp
aclnnStatus aclnnAddRmsNormQuantGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *scales1,
  const aclTensor *scales2Optional,
  const aclTensor *zeroPoints1Optional,
  const aclTensor *zeroPoints2Optional,
  int64_t          axis,
  double           epsilon,
  bool             divMode,
  aclTensor       *y1Out,
  aclTensor       *y2Out,
  aclTensor       *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormQuantGetWorkspaceSize

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
      <td>Source data tensor in the normalization process. Corresponds to `x1` in the corresponding formula.</td>
      <td><ul><li>Supports null Tensor.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process. Corresponds to `x2` in the corresponding formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape and data type keep consistent with `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the weight tensor in the normalization process. Corresponds to `g` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape keeps consistent with the dimensions of `x1` that require norm (layer normalization), and the data type keeps consistent with `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales1 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the scales tensor used to obtain y1Out in the quantization process, corresponding to `scales1` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>The shape must keep consistent with `gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales2Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Scales tensor used to obtain y2Out in the quantization process. Corresponds to `scales2` in the formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>Optional parameter. Supports passing in a null pointer. The shape and data type keep consistent with `scales1`. For combination constraints with `zeroPoints2Optional`, see <a href="#constraints">Constraints</a>.</li><li>If the value of `divMode` is True, the value of this parameter cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints1Optional (aclTensor*)</td>
      <td>Input</td>
      <td>Offset tensor used to obtain y1Out in the quantization process. It corresponds to `zero_points1` in the corresponding formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>Optional parameter, supports passing in a null pointer. The shape must be kept consistent with `scales1`.</li></ul></td>
      <td>INT32, FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints2Optional(aclTensor*)</td>
      <td>Input</td>
      <td>Offset tensor used to obtain y2Out in the quantization process. It corresponds to `zero_points2` in the corresponding formula.</td>
      <td><ul><li>Supports null tensor.</li><li>Optional parameter. Supports passing in a null pointer. The shape keeps consistent with `scales1`, and the data type keeps consistent with `zeroPoints1Optional`. For the constraints combined with `scales2Optional`, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>INT32, FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Indicates the elewise axis on which quantization is performed. Other axes are broadcast. The specified axis cannot exceed the number of dimensions of the input `x1`. Currently, only -1 is supported. Other values do not take effect.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Input eps in the formula, used to prevent division-by-zero errors. The data type is double. A small positive value is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode(bool)</td>
      <td>Input</td>
      <td>Parameter that determines whether division is used in the quantization formula. The data type is bool.</td>
      <td>Supports True and False.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization output tensor, corresponding to `y1Out` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The shape is the same as input `x1`/`x2`.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization output tensor, corresponding to `y2Out` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>Shape and data type are consistent with `y1Out`.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the sum of x1 and x2, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>Shape and data type are consistent with `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>:
    - Data type:
      - The input parameters `x1`, `x2`, and `gamma`, and the output `xOut` only support FLOAT16.
      - The input parameters `scales1` and `scales2Optional` only support FLOAT32.
      - The optional parameters `zeroPoints1Optional` and `zeroPoints2Optional` only support INT32.
      - The output parameters `y1Out` and `y2Out` only support INT8.
    - The input parameter `divMode` only supports True.
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Data type:
      - The input parameters `x1`, `x2`, `gamma` and the output parameter `xOut` only support FLOAT16 and BFLOAT16.
      - The input parameters `scales1` and `scales2Optional` only support FLOAT32 and BFLOAT16.
      - The optional parameters `zeroPoints1Optional` and `zeroPoints2Optional` only support INT32 and BFLOAT16.
      - The output parameters `y1Out` and `y2Out` only support INT8.
    - The input parameter `divMode` only supports True.
  <!-- end id8 -->

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
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If the passed parameter is a required input, output, or attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is out of the supported range, or the optional parameter combination is not supported by the current product.</td>
    </tr>
    </tbody></table>

## aclnnAddRmsNormQuant

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
      <td>The workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size requested on the Device side, obtained by calling the first-phase API aclnnAddRmsNormQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Each dimension size of the shapes of parameters `x1`, `x2`, `gamma`, `scales1`, `scales2Optional`, `zeroPoints1Optional`, `zeroPoints2Optional`, `y1Out`, `y2Out`, and `xOut` must not be greater than the INT32 maximum value 2147483647.

<!-- npu="950" id13 -->
- <term>Ascend 950PR&950DT products</term>: supports passing in a null pointer for `scales2Optional` while `zeroPoints2Optional` is not null. In this case, `scales2` of the second quantization is processed as 1; that is, regardless of whether `divMode` is True or False, $y2Out=round(y+zero\_points2)$ always holds, and the shape of `y2Out` keeps consistent with that of `x1`.
<!-- end id13 -->

<!-- npu="A3,910b,310p" id14 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>: do not support passing in `zeroPoints2Optional` without `scales2Optional`. When `zeroPoints2Optional` is not null, `scales2Optional` must also be not null.
<!-- end id14 -->

<!-- npu="310p" id9 -->
- <term>Atlas inference products</term>: The norm-axis lengths of `x1`, `x2`, `y1Out`, `y2Out`, and `xOut`, and the lengths of `gamma`, `scales1`, `scales2Optional`, `zeroPoints1Optional`, and `zeroPoints2Optional` must be greater than or equal to 32 Bytes.
<!-- end id9 -->

- Data format description:

  It is recommended to use the ND format for all input and output tensors. Other data formats are converted to the ND format by the framework by default for processing.

- Supported data types for each product model:

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    | x1 Data Type | x2 Data Type | Gamma Data Type | scales1 Data Type | scales2Optional Data Type | zeroPoints1Optional Data Type | zeroPoints2Optional Data Type | y1Out Data Type | y2Out Data Type | xOut Data Type |
    | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     INT8 | INT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | INT8 | INT8 | BFLOAT16 |
  <!-- end id10 -->

  <!-- npu="310p" id11 -->
  - <term>Atlas inference products</term>:

    | x1 Data Type | x2 Data Type | Gamma Data Type | scales1 Data Type | scales2Optional Data Type | zeroPoints1Optional Data Type | zeroPoints2Optional Data Type | y1Out Data Type | y2Out Data Type | xOut Data Type |
    | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     INT8 | INT8 | FLOAT16 |
  <!-- end id11 -->

  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>:

    | x1 Data Type | x2 Data Type | Gamma Data Type | scales1 Data Type | scales2Optional Data Type | zeroPoints1Optional Data Type | zeroPoints2Optional Data Type | y1Out Data Type | y2Out Data Type | xOut Data Type |
    | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     INT8 | INT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 |     INT32 | INT8 | INT8 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | INT8 | INT8 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |     FLOAT16 | INT8 | INT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | INT8 | INT8 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | INT8 | INT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | INT8 | INT8 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     HIFLOAT8 | HIFLOAT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 |     INT32 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | HIFLOAT8 | HIFLOAT8 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |     FLOAT16 | HIFLOAT8 | HIFLOAT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | HIFLOAT8 | HIFLOAT8 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 |     INT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |     FLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 |     FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 |     INT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |     FLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |     BFLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |     FLOAT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 |
  <!-- end id12 -->

- Deterministic computation:
  - aclnnAddRmsNormQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. The construction must be customized based on the API.
    std::vector<int64_t> xShape = {64, 2};
    std::vector<int64_t> gammaShape = {2};
    std::vector<int64_t> yShape = {64, 2};
    long long xShapeSize = GetShapeSize(xShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* scales1DeviceAddr = nullptr;
    void* zeroPoints1DeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* scales1 = nullptr;
    aclTensor* zeroPoints1 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* x = nullptr;
    std::vector<int16_t> x1HostData(xShapeSize, 0);
    std::vector<int16_t> x2HostData(xShapeSize, 0);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 0);
    std::vector<float> scales1HostData(gammaShapeSize, 1);
    std::vector<int32_t> zeroPoints1HostData(gammaShapeSize, 100);
    std::vector<int8_t> y1HostData(xShapeSize, 0);
    std::vector<int8_t> y2HostData(xShapeSize, 0);
    std::vector<int16_t> xHostData(xShapeSize, 0);
    float epsilon = 1e-6;
    int64_t axis = -1;
    bool divMode = true;
    // Create the x1 aclTensor.
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x2 aclTensor.
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create scales1 aclTensor.
    ret = CreateAclTensor(scales1HostData, gammaShape, &scales1DeviceAddr, aclDataType::ACL_FLOAT, &scales1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create zeroPoints1 aclTensor.
    ret =
        CreateAclTensor(zeroPoints1HostData, gammaShape, &zeroPoints1DeviceAddr, aclDataType::ACL_INT32, &zeroPoints1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create y1 aclTensor.
    ret = CreateAclTensor(y1HostData, yShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create y2 aclTensor.
    ret = CreateAclTensor(y2HostData, yShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAddRmsNormQuant.
    ret = aclnnAddRmsNormQuantGetWorkspaceSize(
        x1, x2, gamma, scales1, nullptr, zeroPoints1, nullptr, axis, epsilon, divMode, y1, y2, x, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddRmsNormQuant.
    ret = aclnnAddRmsNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait synchronously for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the result from device memory to host memory. Modify the code based on the actual API definition
    auto size = GetShapeSize(yShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), y1DeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the actual API definition
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(scales1);
    aclDestroyTensor(zeroPoints1);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(x);

    // 7. Release device resources. Modify the code based on the actual API definition
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(scales1DeviceAddr);
    aclrtFree(zeroPoints1DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(xDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
