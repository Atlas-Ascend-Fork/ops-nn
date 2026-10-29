# aclnnTransQuantParam

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:42:51.757Z -->

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

- Description: Converts the input scale data from FLOAT32 type to UINT64 type required by the hardware, and stores it in quantParam.
- Formula:
  1. `out` is in 64-bit format and is initialized to 0.

  2. `scale` is truncated to the upper 19 bits, stored at bit position 32 of `out`, and bit 46 is set to 1.

     $$
     out = out\ |\ (scale\ \&\ 0xFFFFE000)\ |\ (1\ll46)
     $$
  3. Perform subsequent calculations based on the `offset` value:
     - If `offset` does not exist, no further computation is performed.
     - If `offset` exists:
       1. Process the `offset` value as int, with a range of [-256, 255].

          $$
          offset = Max(Min(INT(Round(offset)),255),-256)
          $$

       2. Retain the lower 9 bits of `offset` and store them in bits 37 to 45 of `out`.

          $$
          out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0x1FF)\ll37)
          $$

## Function Prototype

```Cpp
aclnnStatus aclnnTransQuantParam(
  const float  *scaleArray,
  uint64_t      scaleSize,
  const float  *offsetArray,
  uint64_t      offsetSize,
  uint64_t    **quantParam,
  uint64_t     *quantParamSize)
```

## aclnnTransQuantParam

- **Parameters:**

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
      <td>scaleArray (float*)</td>
      <td>Input</td>
      <td>Indicates that it points to the memory that stores scale data, corresponding to `scale` in the formula.</td>
      <td>Ensure that NaN and inf do not exist in scale data.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleSize (uint64_t)</td>
      <td>Input</td>
      <td>Indicates the number of scale data.</td>
      <td>You need to manually ensure that `scaleSize` is the same as the number of elements contained in `scaleArray`.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetArray (float*)</td>
      <td>Input</td>
      <td>Indicates that it points to the memory within which offset data is stored, corresponding to `offset` in the formula.</td>
      <td>Ensure that no NaN or inf exists in the offset data.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetSize (uint64_t)</td>
      <td>Input</td>
      <td>Indicates the number of offset data.</td>
      <td>You need to manually ensure that `offsetSize` is the same as the number of elements in `offsetArray`.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantParam (uint64_t**)</td>
      <td>Output</td>
      <td>Indicates that it points to the address of the memory that stores the converted quantParam data, corresponding to `out` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantParamSize (uint64_t*)</td>
      <td>Output</td>
      <td>Indicates the number of quantParam data items stored.</td>
      <td>You need to manually ensure that `quantParamSize` is the same as the number of elements contained in `quantParam`.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The API performs input parameter validation and returns an error in the following scenarios:

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
      <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="3">161001</td>
      <td>Parameter quantParam is a null pointer.</td>
    </tr>
    <tr>
      <td>Parameter scaleArray is a null pointer.</td>
    </tr>
    <tr>
      <td>Parameter quantParamSize is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>When scaleArray is not a null pointer, parameter scaleSize < 1.</td>
    </tr>
    <tr>
      <td>When offsetArray is not a null pointer, parameter offsetSize < 1.</td>
    </tr>
    <tr>
      <td>offsetArray is a null pointer and parameter offsetSize is not 0.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>quantParam is a null pointer.</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic computation:
  - aclnnTransQuantParam defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param.h"

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

int main()
{
    float scaleArray[3] = {1.0, 1.0, 1.0};
    uint64_t scaleSize = 3;
    float offsetArray[3] = {1.0, 1.0, 1.0};
    uint64_t offsetSize = 3;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    auto ret = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParam failed. ERROR: %d\n", ret); return ret);
    for (auto i = 0; i < resultSize; i++) {
        LOG_PRINT("result[%d] is: %ld\n", i, result[i]);
    }
    free(result);
    return 0;
}
```
