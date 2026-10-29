# aclnnSwigluMxQuantWithDualAxis

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:44:14.653Z -->

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

- Description: Fused operator that combines the SwiGLU activation function with dual-axis dynamic block quantization. It first computes the SwiGLU activation function on the input, and then dynamically quantizes the result block-wise along both the -1 axis and the -2 axis, outputting low-precision FP4/FP8 tensors and the corresponding scale factors.

- Formula:

  **Stage 1: SwiGLU activation function**

  - When `activateLeft=true`:

    $$
    A_i = x_i \text{ first half}, B_i = x_i \text{ second half}
    $$

  - When `activateLeft=false`:

    $$
    A_i = x_i \text{ second half}, B_i = x_i \text{ first half}
    $$

  - SwiGLU computation:

    $$
    swigluOut_i = Swish(A_i) * B_i = \frac{A_i}{1 + e^{-A_i}} * B_i
    $$

  **Stage 2: dual-axis dynamic block quantization**

  - **-1-axis quantization (column direction)**: The SwiGLU result is grouped along axis -1 in units of 32 numbers. Each group of 32 numbers $\{\{V_i\}_{i=1}^{32}\}$ is quantized to $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale1 = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - **-2-axis quantization (row direction)**: Groups the SwiGLU result along the -2 axis into groups of 32 numbers, and each group of 32 numbers $\{\{V_j\}_{j=1}^{32}\}$ is quantized into $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$

    $$
    mxscale2 = 2^{shared\_exp}
    $$

    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - $P_{i}$ obtained after -1-axis quantization forms the output y1Out according to the positions of the corresponding $V_{i}$, and mxscale1 forms the output mxscale1Out according to the groups along the -1-axis dimension. $P_{j}$ obtained after -2-axis quantization forms the output y2Out according to the positions of the corresponding $V_{j}$, and mxscale2 forms the output mxscale2Out according to the groups along the -2-axis dimension.

  - emax: The exponent of the maximum normal number of the corresponding data type.

    | DataType | Emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnSwigluMxQuantWithDualAxis" API to execute the computation.

```cpp
aclnnStatus aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *roundModeOptional,
  int64_t          scaleAlg,
  int64_t          dstType,
  double           maxDtypeValue,
  const aclTensor *y1Out,
  const aclTensor *mxscale1Out,
  const aclTensor *y2Out,
  const aclTensor *mxscale2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnSwigluMxQuantWithDualAxis(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluMxQuantWithDualAxis

- **Parameters:**

  <table><colgroup>
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
      <td>Input tensor, corresponding to x in the formula.</td>
      <td><ul><li>The shape is [M, 2N], and the last dimension must be an even number.</li><li>If dstType is FLOAT4_E2M1/FLOAT4_E1M2, the last dimension must be divisible by 4.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Group index, used to control group quantization boundaries.</td>
      <td><ul><li>The shape is [G]. In cumsum mode, the value indicates the cumulative number of rows in each group.</li><li>Currently, passing an empty tensor is not supported.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>SwiGLU activation side selection.</td>
      <td><ul><li>True indicates that the left half is hidden and the right half is gate.</li><li>False indicates that the right half is hidden and the left half is gate.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Represents the data conversion mode, corresponding to round_mode in the formula.</td>
      <td><ul><li>When dstType is 35/36, corresponding to output y1Out and y2Out data types of FLOAT8_E5M2/FLOAT8_E4M3FN, only {"rint"} is supported.</li><li>When dstType is 40/41, corresponding to output y1Out and y2Out data types of FLOAT4_E2M1/FLOAT4_E1M2, {"rint", "floor", "round"} are supported.</li><li>If a null pointer is passed, the "rint" mode is used.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Specifies the calculation method of mxscale1Out and mxscale2Out.</td>
      <td><ul><li>Value range: {1, 0}, where 1 indicates the cuBLAS implementation and 0 indicates the OCP implementation.</li><li>Only 0 is supported for scaleAlg when dstType is 40/41.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the data types of y1Out and y2Out after the specified data conversion.</td>
      <td><ul><li>Value range: {35, 36, 40, 41}, which respectively correspond to the data types of outputs y1Out and y2Out: {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN, 40: FLOAT4_E2M1, 41: FLOAT4_E1M2}.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxDtypeValue (double)</td>
      <td>Input</td>
      <td>Reserved parameter.</td>
      <td><ul><li>maxDtypeValue supports only the value 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of the SwiGLU output after quantization along axis -1, corresponding to <i>P<sub>i</sub></i> in the formula.</td>
      <td><ul><li>The shape is [M, N], which is half of the SwiGLU output.</li></ul></td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale corresponding to each group along the -1 axis, which corresponds to mxscale1 in the corresponding formula.</td>
      <td><ul><li>The shape is [M, ceil(N/64), 2]. Even padding is required, with a padding value of 0.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result obtained by quantizing the SwiGLU result on axis -2, corresponding to <i>P<sub>j</sub></i> in the formula.</td>
      <td><ul><li>Shape: [M, N].</li></ul></td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale corresponding to each group along the -2 axis, corresponding to mxscale2 in the formula.</td>
      <td><ul><li>If groupIndexOptional exists, the shape is [floor(M/64)+G, N, 2].</li><li>If groupIndexOptional does not exist, the shape is [ceil(M/64), N, 2].</li><li>Even padding is required, with the padding value set to 0.</li><li>The mxscale2Out output needs to be interleaved for every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
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
      <td>Returns the op executor that contains the operator computation flow.</td>
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

  <table><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
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
      <td>x, y1Out, mxscale1Out, y2Out, or mxscale2Out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type or data format of x, roundModeOptional, dstType, scaleAlg, y1Out, mxscale1Out, y2Out, or mxscale2Out is out of the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x, y1Out, y2Out, mxscale1Out, or mxscale2Out does not meet the check conditions.</td>
    </tr>
    <tr>
      <td>roundModeOptional, dstType, or scaleAlg is not one of the currently supported values.</td>
    </tr>
    <tr>
      <td>The last dimension of x is not divisible by 2.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is out of the supported platform range.</td>
    </tr>
  </tbody></table>

## aclnnSwigluMxQuantWithDualAxis

- **Parameters:**

  <table><colgroup>
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side, obtained by the first-phase API aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- x must be a 2D tensor, and the last dimension must be divisible by 2 (shape [M, 2N]).
- If dstType is FLOAT4_E2M1/FLOAT4_E1M2, the last dimension of x must be divisible by 4.
- FP8 output type (FLOAT8_E4M3FN) supports only the "rint" rounding mode.
- groupIndexOptional adopts the cumsum mode, where each value indicates the cumulative row count of the corresponding group. Each element of groupIndexOptional must be greater than 0, and the last element must be equal to M.
- Shape constraints for mxscale1Out and mxscale2Out:
  - mxscale1Out.shape[-2] = (ceil(N/32) + 2 - 1) / 2.
  - mxscale1Out.shape[-1] = 2.
  - mxscale2Out.shape[-1] = 2.
  - If groupIndexOptional is specified, mxscale2Out.shape[0] = floor(M/64) + G; otherwise, mxscale2Out.shape[0] = ceil(M/64).
  - Other dimensions are the same as those of the SwiGLU output.
- Saturation mode is supported by default.
- Deterministic computation: aclnnSwigluMxQuantWithDualAxis defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_mx_quant_with_dual_axis.h"

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
    // Boilerplate for resource initialization.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnSwigluMxQuantWithDualAxisTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction according to the API.
    // Input x shape: [M, 2N] = [256, 1024].
    std::vector<int64_t> xShape = {256, 1024};
    // group_index shape: [G] = [1]. In cumsum mode, a value of 256 indicates that one group contains all 256 rows.
    std::vector<int64_t> groupIndexShape = {1};
    // SwiGLU output shape: [M, N] = [256, 512].
    std::vector<int64_t> y1OutShape = {256, 512};
    std::vector<int64_t> y2OutShape = {256, 512};
    // mxscale1 shape: [M, ceil(N/32)/2 padded to even, 2] = [256, 8, 2].
    std::vector<int64_t> mxscale1OutShape = {256, 8, 2};
    // mxscale2 shape: [floor(M/64) + G, N, 2] = [4 + 1, 512, 2] = [5, 512, 2]
    std::vector<int64_t> mxscale2OutShape = {5, 512, 2};

    void* xDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* y1OutDeviceAddr = nullptr;
    void* mxscale1OutDeviceAddr = nullptr;
    void* y2OutDeviceAddr = nullptr;
    void* mxscale2OutDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* y1Out = nullptr;
    aclTensor* mxscale1Out = nullptr;
    aclTensor* y2Out = nullptr;
    aclTensor* mxscale2Out = nullptr;

    // Initialize the input data (BF16)
    std::vector<uint16_t> xHostData(256 * 1024, 0);
    for (int64_t i = 0; i < 256 * 1024; i++) {
        xHostData[i] = static_cast<uint16_t>(i % 100);
    }
    // group_index data: cumsum mode, a value of 256 indicates a single group containing all rows
    std::vector<int64_t> groupIndexHostData = {256};
    std::vector<uint8_t> y1OutHostData(256 * 512, 0);
    std::vector<uint8_t> y2OutHostData(256 * 512, 0);
    std::vector<uint8_t> mxscale1OutHostData(256 * 8 * 2, 0);
    std::vector<uint8_t> mxscale2OutHostData(5 * 512 * 2, 0);

    // Set parameters
    bool activateLeft = true;
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t dstType = 36;  // FLOAT8_E4M3FN
    int64_t scaleAlg = 1;  // cuBLAS
    double maxDtypeValue = 0.0;

    // Create the x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the groupIndex aclTensor
    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64, &groupIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupIndexTensorPtr(groupIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupIndexDeviceAddrPtr(groupIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the y1Out aclTensor
    ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the mxscale1Out aclTensor
    ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale1OutTensorPtr(mxscale1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale1OutDeviceAddrPtr(mxscale1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the y2Out aclTensor
    ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the mxscale2Out aclTensor
    ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale2OutTensorPtr(mxscale2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale2OutDeviceAddrPtr(mxscale2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call the CANN operator library API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnSwigluMxQuantWithDualAxis (with groupIndex input).
    ret = aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize(x, groupIndex, activateLeft, roundModeOptional,
        scaleAlg, dstType, maxDtypeValue, y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // Call the second-phase API of aclnnSwigluMxQuantWithDualAxis.
    ret = aclnnSwigluMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantWithDualAxis failed. ERROR: %d\n", ret); return ret);

    // (Boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output values and copy the results from device to host.
    auto size1 = GetShapeSize(y1OutShape);
    auto size2 = GetShapeSize(y2OutShape);
    std::vector<uint8_t> y1OutData(size1, 0);
    std::vector<uint8_t> y2OutData(size2, 0);

    ret = aclrtMemcpy(y1OutData.data(), y1OutData.size() * sizeof(uint8_t), y1OutDeviceAddr,
                      size1 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(y2OutData.data(), y2OutData.size() * sizeof(uint8_t), y2OutDeviceAddr,
                      size2 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2Out from device to host failed. ERROR: %d\n", ret);
              return ret);

    // Print part of the output results.
    LOG_PRINT("y1Out first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size1; i++) {
        LOG_PRINT("y1Out[%ld] = %d\n", i, y1OutData[i]);
    }
    LOG_PRINT("y2Out first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size2; i++) {
        LOG_PRINT("y2Out[%ld] = %d\n", i, y2OutData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnSwigluMxQuantWithDualAxisTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantWithDualAxisTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
