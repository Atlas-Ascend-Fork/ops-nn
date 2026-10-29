# aclnnGroupedDynamicMxQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:34:49.844Z -->

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

- Description: Based on the start value of the passed-in group index (`groupIndex`), performs dynamic MX quantization with the target data type `float4/float8` on each group at the granularity of basic blocks.

- Formula:
  - Use case 1: when `scaleAlg` is 0:
    - The input `x` is first grouped along dimension 0 according to `groupIndex`, and within each group, numbers are grouped by k = blocksize numbers. For a group of k numbers {{x<sub>i</sub>}<sub>i=1</sub><sup>k</sup>}, the quantization scale `mxscale_pre`, {mxscale_pre, {P<sub>i</sub>}<sub>i=1</sub><sup>k</sup>}, corresponding to this group of numbers is calculated. The formula is shown in Formula (1) and (2) below.

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax  \tag{1}
    $$

    $$
    mxscale\_pre = 2^{shared\_exp}  \tag{2}
    $$

    - Each number in this group is divided by `mxscale_pre`, and converted to the corresponding `dst_type` according to `round_mode`, to obtain the quantized result `y`. The calculation formula is formula (3) below.

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize \tag{3}
    $$

    - The quantized $P_{i}$ are combined as output `y` according to the positions of the corresponding $V_{i}$. `mxscale_pre` is grouped by the corresponding `groupIndex`, the first dimension within each group is padded to an even number, and the result is combined as output `mxscale`.

    - `emax`: Exponent bits of the maximum normal number for the corresponding data type.

        | DataType | Emax |
        | :-----------: | :--: |
        | FLOAT4_E2M1   |  2   |
        | FLOAT4_E1M2   |  0   |
        | FLOAT8_E4M3FN |  8   |
        | FLOAT8_E5M2   |  15  |

  - Use case 2, when `scaleAlg` is 1, only FP8 types are involved:
    - Divide the long vector into blocks, each with a length of k. Compute a block scale factor $S_{fp32}^b$ for each block individually, and then map all elements within the block to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block has fewer than k elements, treat the missing values as 0 and process it as a complete block.
    - Find the maximum absolute value of the values in this block:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - Map FP32 to the range representable by the target data type FP8, where $Amax(DType)$ is the maximum value representable by the target precision.

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

    - Convert the block scale factor $S_{fp32}^b$ to the representable scale value $S_{ue8m0}^b$ in FP8 format.
    - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block floating-point scale factor $S_{fp32}^b$.
    - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:

      $
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, and } E_{int}^b < 254 \text{ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number, and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $

    - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
    - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - Apply to the final quantization step. For each element within a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ represents the block scale factor, i.e., $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ represents the quantized data within the block.

  - Scenario 3, when `scaleAlg` is 2, only the FP4 type is involved:
    - When `dstTypeMax = 0.0/6.0/7.0(FP4_E2M1)` or `dstTypeMax = 1.875(FP4_E1M2)`:
      - Group the input `x` along dimension 0 into groups of k = blocksize elements, and each group of k elements  $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = blocksize:
      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{ if} the leading one/two/three high bits of the mantissa \text{ are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{ otherwise} \end{cases} \\
      $$
      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
      $$
      - The quantized $P_{i}$ values are arranged as output `y` at the positions of the corresponding $V_{i}$, and the groupings along the corresponding dimension are arranged as output `mxscale`.
    - When `dstTypeMax != 0.0/6.0/7.0(FP4_E2M1)` or `dstTypeMax != 1.875(FP4_E1M2)`:
      - Divide the long vector into blocks, each of length k, and compute a block scaling factor $S_{fp32}^b$ separately for each block. Then map all elements in the block to the target low-precision type using the same $S_{fp32}^b$. If the last block contains fewer than k elements, treat the missing values as 0 and process the block as a full block.
      - Find the maximum absolute value in the block:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
      - Map FP32 to the range representable by the target data type, where $Amax(DType)$ is the maximum value representable by the target precision when `dst_type_max` is 0; $Amax(DType)$ is the value passed in through `dst_type_max` when `dst_type_max` is not 0.
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$
      - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
      - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point scaling factor $S_{fp32}^b$ of the block.
      - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:
        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{ otherwise} \end{cases}
        $$
      - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
      - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - Apply to the final quantization step. For each element within a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ represents the block scale factor, i.e., $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ represents the quantized data within the block.
      - The quantized $P_{i}$ at the positions of the corresponding $V_{i}$ form the output `y`, and the grouping along the corresponding dimensions forms the output `mxscale`.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGroupedDynamicMxQuantV2GetWorkspaceSize" API to obtain the workspace size required for calculation and the executor that contains calculation flows of the operator, and then call the "aclnnGroupedDynamicMxQuantV2" API to perform calculation.

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupIndex,
  const char      *roundMode,
  int64_t          dstType,
  int64_t          blocksize,
  int64_t          scaleAlg,
  double           dstTypeMax,
  const aclTensor *y,
  const aclTensor *mxscale,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedDynamicMxQuantV2GetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Tensor of the operator input. Input x in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Start index of the quantization group.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Indices must be greater than or equal to 0 and non-decreasing, and the last number must be equal to the first dimension size of x.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>Input</td>
      <td>round_mode in the formula, indicating the data conversion mode.</td>
      <td><ul><li>The "rint"/"round"/"floor" modes are supported.</li><li>For FLOAT8, only "rint" is supported. For FLOAT4, all three modes "rint"/"round"/"floor" are supported.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>dst_type in the formula, specifying the data type of y after conversion.</td>
      <td>Value range: {35, 36, 40, 41}, corresponding respectively to the data types of output y: {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN, 40: FLOAT4_E2M1, 41: FLOAT4_E1M2}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blocksize (int64_t)</td>
      <td>Input</td>
      <td>blocksize in the formula. Specifies the number of elements quantized each time.</td>
      <td>Currently, only the value 32 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Specifies the algorithm used for mxscale calculation.</td>
      <td><ul><li>The current value supports 0, 1, and 2. A value of 0 represents representative scenario 1, a value of 1 represents representative scenario 2, and a value of 2 represents representative scenario 3.</li><li>When dstType is FLOAT4, the supported values are 0 and 2; when dstType is FLOAT8, the supported values are 0 and 1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (double)</td>
      <td>Input</td>
      <td>Represents the value of maxType, corresponding to Amax(Dtype) in the formula.</td>
      <td><ul><li>The current value supports 0.0 and [1.75, 3.5) (FLOAT4_E1M2), as well as 0.0 and [6.0, 12.0](FLOAT4_E2M1). This value can be set only for FP4.</li><li>A value of 0.0 indicates that Amax(Dtype) is the maximum value of the quantized result data type; a value of [1.75, 3.5) or [6.0, 12.0] indicates that Amax(Dtype) is the input value.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantized output tensor. Corresponds to y in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The dimensions of shape are the same as those of x.</li><li>Data types: FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, and FLOAT8_E5M2, which must correspond to dstType.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>FLOAT8 supports non-contiguous, but non-contiguous is not supported for FLOAT4.</td>
    </tr>
    <tr>
      <td>mxscale (aclTensor*)</td>
      <td>Output</td>
      <td>Output mxscale composed of mxscale_pre in the formula, which is the quantization scale corresponding to each group.</td>
      <td><ul><li>Supports empty Tensor.</li><li>If the shape of x is [m,n] and the shape of groupIndex is [g], the shape of mxscale is [(m/(blocksize∗2)+g),n,2].</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
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
      <td>Returns the op executor, which contains the calculation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
  <tr>
      <th>Error Code</th>
      <th>Error code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>If a passed-in parameter is a mandatory input, output, or mandatory attribute and is a null pointer, 161001 is returned.</td>
  </tr>
  <tr>
    <td>The input roundMode is a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of x, groupIndex, y, and mxscale are not within the supported range.</td>
  </tr>
  <tr>
    <td>The shapes of x, y, and mxscale do not meet the check conditions.</td>
  </tr>
  <tr>
    <td>The dimensions of x, groupIndex, y, and mxscale are not within the supported range.</td>
  </tr>
  <tr>
    <td>roundMode, dstType, blocksize, scaleAlg, or dstTypeMax is not a currently supported value.</td>
  </tr>
  <tr>
    <td>Non-contiguous tensors are not supported when y is FLOAT4_E2M1 or FLOAT4_E1M2.</td>
  </tr>
  <tr>
    <td>Non-contiguous tensors are not supported by mxscale.</td>
  </tr>
  <tr>
    <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
    <td rowspan="1">361001</td>
    <td>The current platform is not within the range of supported platforms.</td>
  </tr>
  </tbody></table>

## aclnnGroupedDynamicMxQuantV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1033px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 748px">
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
      <td>Memory address of the workspace applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied by the Device side, obtained by calling the first-phase API aclnnGroupedDynamicMxQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains calculation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedDynamicMxQuantV2 defaults to a deterministic implementation.
- The shape constraints on x, groupIndex, y, and mxscale are as follows:
  - rank(mxscale) = rank(x) + 1.
  - mxscale.shape[0] = x.shape[0] / (blocksize * 2) + g, where g is the number of groups.
  - mxscale.shape[-1] = 2. Other dimensions are the same as those of input x.
  - Output y has the same shape as x.
  - If the data type of output y is FLOAT4_E2M1 or FLOAT4_E1M2, the length of the tail axis (dimension 1) of x must be an even number.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_dynamic_mx_quant_v2.h"

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
    // Boilerplate for resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create aclTensor
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

int aclnnGroupedDynamicMxQuantV2Test(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Custom construction is required based on the API.
    std::vector<int64_t> xShape = {8, 1};
    std::vector<int64_t> groupIndexShape = {2};
    std::vector<int64_t> yOutShape = {8, 1};
    std::vector<int64_t> mxscaleOutShape = {2, 1, 2};
    void* xDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* mxscaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* mxscaleOut = nullptr;
    // Values corresponding to BF16: (0, 8, 64, 512).
    std::vector<uint16_t> xHostData = {{0}, {16640}, {17024}, {17408}, {0}, {16640}, {17024}, {17408}};
    std::vector<uint32_t> groupIndexHostData = {4, 8};
    // Values corresponding to float8_e4m3: (0, 4, 32, 256).
    std::vector<uint8_t> yOutHostData = {{0}, {72}, {96}, {120}, {0}, {72}, {96}, {120}};
    // Value corresponding to float8_e8m0: (2).
    std::vector<std::vector<uint8_t>> mxscaleOutHostData = {{{128, 0}}, {{128, 0}}};
    const char* roundModeOptional = "rint";
    int64_t dstType = 36;
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the groupIndex aclTensor.
    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT32, &groupIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupIndexTensorPtr(groupIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupIndexDeviceAddrPtr(groupIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the yOut aclTensor.
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mxscaleOut aclTensor.
    ret = CreateAclTensor(mxscaleOutHostData, mxscaleOutShape, &mxscaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscaleOutTensorPtr(mxscaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscaleOutDeviceAddrPtr(mxscaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnGroupedDynamicMxQuantV2
    ret = aclnnGroupedDynamicMxQuantV2GetWorkspaceSize(x, groupIndex, roundModeOptional, dstType, blocksize, scaleAlg,
                                                       dstTypeMax, yOut, mxscaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnGroupedDynamicMxQuantV2
    ret = aclnnGroupedDynamicMxQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantV2 failed. ERROR: %d\n", ret); return ret);

    // (boilerplate) Synchronize and wait for the task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output value and copy the result on device memory to the host. Modify it according to the definition of the specific API
    auto size = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(size, 0); // fp8 data cannot be printed directly in C. Read the data as uint8 and manually convert the binary data to fp8
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                      size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yOutData[i]);
    }
    size = GetShapeSize(mxscaleOutShape);
    std::vector<uint8_t> mxscaleOutData(size, 0); // fp8 data cannot be printed directly in C. Read the data as uint8 and manually convert the binary data to fp8
    ret = aclrtMemcpy(mxscaleOutData.data(), mxscaleOutData.size() * sizeof(mxscaleOutData[0]), mxscaleOutDeviceAddr,
                      size * sizeof(mxscaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscaleOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mxscaleOut[%ld] is: %d\n", i, mxscaleOutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedDynamicMxQuantV2Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantV2Test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
