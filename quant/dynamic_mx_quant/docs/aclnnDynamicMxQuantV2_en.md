# aclnnDynamicMxQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:18:03.949Z -->

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

- Description: MX quantization for the FLOAT4 and FLOAT8 destination data types. Along a given axis, the quantization scale mxscale is computed for every group of blocksize elements and used as the corresponding part of the output mxscaleOut. Each element in the group is divided by mxscale and converted to the corresponding dstType according to round_mode, and the quantization result y is used as the corresponding part of the output yOut. When dstType is FLOAT8_E4M3FN or FLOAT8_E5M2, the value of scaleAlg determines the algorithm for computing mxscale. Compared with aclnnDynamicMxQuant, this API adds the dstTypeMax parameter, which sets the maximum value of the destination data type.

- Formula:
  - Scenario 1, when scaleAlg is 0:
    - The input x is grouped along the axis dimension in groups of k = blocksize elements, and each group of k elements $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized as $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$

    - ​The quantized $P_{i}$ values are placed at the positions corresponding to $V_{i}$ to form the output yOut, and the mxscale values of the groups along the corresponding axis dimension form the output mxscaleOut.

    - emax: The exponent of the maximum normal number of the corresponding data type.

        | DataType | Emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

  - Scenario 2: When scaleAlg is 1, only FP8 types are involved:
    - Divide the long vector into blocks, each with a length of k. Compute a block scale factor $S_{fp32}^b$ for each block individually, and then map all elements within the block to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block has fewer than k elements, treat the missing values as 0 and process it as a complete block.
    - Find the maximum absolute value in the block:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
    - Map FP32 to the representable range of the target data type FP8, where $Amax(DType)$ is the maximum value that the target precision can represent.
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
  - Use case 3: when scaleAlg is 2, only the FP4_E2M1 type is involved:
    - When dstTypeMax = 0.0/6.0/7.0:
      - Group the input x along the axis dimension into groups of k = blocksize elements. Each group of k elements is  $\{\{V_i\}_{i=1}^{k}\}$ dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize:
      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{ if} the one or two bits preceding the high bits of the mantissa \text{ are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{ otherwise} \end{cases} \\
      $$
      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
      $$
      - ​The quantized $P_{i}$ values are placed at the positions corresponding to $V_{i}$ to form the output yOut, and the mxscale values of the groups along the corresponding axis dimension form the output mxscaleOut.
    - When dstTypeMax != 0.0/6.0/7.0:
      - Divide the long vector into blocks, each of length k, and compute a block scaling factor $S_{fp32}^b$ for each block. Then map all elements in the block to the target low-precision type using the same $S_{fp32}^b$. If the last block contains fewer than k elements, treat the missing values as 0 and process it as a complete block.
      - Find the maximum absolute value in the block:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
      - Map FP32 to within the range representable by the target data type. When dst_max_value=0, $Amax(DType)$ is the maximum value representable by the target precision; when dst_max_value!=0, $Amax(DType)$ is the value passed in for dst_max_value.
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$
      - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in the FP8 format.
      - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
      - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:
        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{ otherwise} \end{cases}
        $$
      - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
      - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - Apply to the final quantization step. For each element within a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ represents the block scale factor, i.e., $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ represents the quantized data within the block.
      - Place the quantized $P_{i}$ values at the positions corresponding to $V_{i}$ to form the output yOut, and organize mxscale by the groups along the axis dimension to form the output mxscaleOut.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnDynamicMxQuantV2GetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnDynamicMxQuantV2" API to perform the computation.

```cpp
aclnnStatus aclnnDynamicMxQuantV2GetWorkspaceSize(
  const aclTensor *x,
  int64_t          axis,
  char            *roundModeOptional,
  int64_t          dstType,
  int64_t          blocksize,
  int64_t          scaleAlg,
  double           dstTypeMax,
  const aclTensor *yOut,
  const aclTensor *mxscaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicMxQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicMxQuantV2GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input x, corresponding to Vi and di in the formula.</td>
      <td><ul><li>If the destination type is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of x must be an even number.</li><li>If the type of x is FLOAT, only blocksize=32 is supported, and the dimension of the quantization axis must not be less than 32.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Represents the axis on which quantization is performed, corresponding to axis in the formula.</td>
      <td><ul><li>The value range is [-D, D-1], where D is the number of dimensions in the shape of x.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Represents the data conversion mode, corresponding to round_mode in the formula.</td>
      <td><ul><li>If dstType is 40/41, {"rint", "floor", "round"} are supported.</li><li>If dstType is 36/35, only {"rint"} is supported.</li><li>If a null pointer is passed in, the "rint" mode is used.</li></ul></td>
      <td>STRING</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the type of yOut after data conversion, corresponding to DType in the formula.</td>
      <td><ul><li>The value range is {35, 36, 40, 41}, corresponding to the data types of output yOut {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}, respectively.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blocksize (int64_t)</td>
      <td>Input</td>
      <td>Specifies the number of elements to quantize at a time, corresponding to blocksize in the formula.</td>
      <td><ul><li>Only multiples of 32 are supported. The value cannot be 0 or exceed 1024.</li><li>If scaleAlg is 2, blocksize must be 32.</li><li>If x is of the FLOAT type, blocksize must be 32.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Specifies the calculation method of mxscaleOut, corresponding to scaleAlg in the formula.</td>
      <td><ul><li>The value can be 0, 1, or 2. Value 0 indicates use case 1, value 1 indicates use case 2, and value 2 indicates use case 3.</li><li>If dstType is FLOAT4_E1M2, only value 0 is supported; if dstType is FLOAT4_E2M1, only values 0 and 2 are supported; if dstType is FLOAT8, only values 0 and 1 are supported.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (double)</td>
      <td>Input</td>
      <td>Specifies the value of maxType, corresponding to Amax(DType) in the formula.</td>
      <td><ul><li>Supports values 0.0 and 6.0-12.0. The value 0.0 indicates that Amax(DType) is the maximum value of the quantization result data type; a value in the range 6.0-12.0 indicates that Amax(DType) is the input value. This value can be set only when FP4E2M1 is used and blocksize is 32.</li></ul></td>
      <td>DOUBLE</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization result of input x, corresponding to Pi and di in the formula.</td>
      <td><ul><li>The shape is the same as that of input x.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>mxscaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale of each group, corresponding to mxscale and Sb in the formula.</td>
      <td><ul><li>The shape of the axis dimension is the value of the corresponding axis of x divided by blocksize and rounded up, and is padded to an even number with the padding value 0.</li><li>When axis is a non-tail axis, every two rows of the mxscaleOut output need to be interleaved.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
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
      <td>x is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td> The data types and data formats of x, axis, roundModeOptional, dstType, blocksize, scaleAlg, dstTypeMax, yOut, and mxscaleOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of x, yOut, or mxscaleOut do not meet the check conditions.</td>
    </tr>
    <tr>
      <td>The values of axis, roundModeOptional, dstType, blocksize, scaleAlg, and dstTypeMax are not among the currently supported values.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not in the range of supported platforms.</td>
    </tr>
  </tbody></table>

## aclnnDynamicMxQuantV2

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 173px">
  <col style="width: 124px">
  <col style="width: 852px">
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
      <td>Size of the workspace requested on the device, obtained by calling the first-phase API aclnnDynamicMxQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnDynamicMxQuantV2 defaults to a deterministic implementation.
- The shape constraints for x and mxscaleOut are as follows:
  - rank(mxscaleOut) = rank(x) + 1.
  - axis_change = axis if axis >= 0 else axis + rank(x).
  - mxscaleOut.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2.
  - mxscaleOut.shape[-1] = 2.
  - Other dimensions are consistent with input x.
- If the data type of x is FLOAT:
  - blocksize must be 32.
  - The dimension of the quantization axis cannot be less than 32.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_mx_quant_v2.h"

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

    int64_t
    GetShapeSize(const std::vector<int64_t>& shape)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnDynamicMxQuantV2Test(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs, which need to be custom-constructed as required by the API.
    std::vector<int64_t> xShape = {1, 4};
    std::vector<int64_t> yOutShape = {1, 4};
    std::vector<int64_t> mxscaleOutShape = {1, 1, 2};
    void* xDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* mxscaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* mxscaleOut = nullptr;
    // Corresponding BF16 values (0, 8, 64, 512)
    std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
    // Corresponding float8_e4m3 values (0, 4, 32, 256)
    std::vector<uint8_t> yOutHostData = {0, 72, 96, 120};
    // Corresponding float8_e8m0 values (2)
    std::vector<uint8_t> mxscaleOutHostData = {128, 0};
    int64_t axis = -1;
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t dstType = 36;
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
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

    // Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnDynamicMxQuantV2.
    ret = aclnnDynamicMxQuantV2GetWorkspaceSize(x, axis, roundModeOptional, dstType, blocksize, scaleAlg, dstTypeMax, yOut, mxscaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnDynamicMxQuantV2.
    ret = aclnnDynamicMxQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantV2 failed. ERROR: %d\n", ret); return ret);

    //(Boilerplate) Wait synchronously for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output value and copy the result from device memory to host memory. Modify it according to the definition of the specific API.
    auto size = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(
        size, 0);  // In C, fp8 data cannot be printed directly. Read it as uint8 and manually convert the binary data to fp8.
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                      size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
    }
    size = GetShapeSize(mxscaleOutShape);
    std::vector<uint8_t> mxscaleOutData(
        size, 0);  // In C, fp8 data cannot be printed directly. Read it as uint8 and manually convert the binary data to fp8.
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
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnDynamicMxQuantV2Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantV2Test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
