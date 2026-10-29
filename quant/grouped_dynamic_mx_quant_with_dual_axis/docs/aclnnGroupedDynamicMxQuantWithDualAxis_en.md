# aclnnGroupedDynamicMxQuantWithDualAxis

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:34:59.194Z -->

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

- Description: Performs grouped dynamic MX dual-axis quantization on the 2D input x. In a single call, the operator completes quantization along the -1 and -2 axes based on groupIndex and outputs y1Out, y1ScaleOut, y2Out, and y2ScaleOut. Both quantization axes determine the group ranges based on groupIndex: within each group, MX quantization is performed along the -1 axis in the column direction, and MX quantization is performed along the -2 axis in the row direction.
- Computation: Quantization uses 32 elements as one MX block. When scaleAlg = 1, each block generates a scale of the FLOAT8_E8M0 type, and the scale is used to convert elements in the block to the data type specified by dstDtype.

- Formula:

  For the input data in an MX block:

  $$
  D^{b}=[d^{i}]_{i=1}^{k},\quad k=32
  $$

  Compute the maximum absolute value of the elements in the block:

  $$
  Amax(D^{b})=\max_i(|d^{i}|)
  $$

  When scaleAlg=1, compute the FP32 scale based on the maximum representable value of the target FP8 type:

  $$
  S_{fp32}^{b}=\frac{Amax(D^{b})}{Amax(dstDtype)}
  $$

  Extract the unbiased exponent $E_{int}^{b}$ and mantissa $M_{fixp}^{b}$ from the block scale factor $S_{fp32}^{b}$.

  To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:

  $$
  E_{int}^{b} = \begin{cases}
    E_{int}^{b} + 1, & \text{ if} S_{fp32}^{b} \text{ is a normal number, and} E_{int}^{b} < 254 \text{ and} M_{fixp}^{b} > 0 \\
    E_{int}^{b} + 1, & \text{ if} S_{fp32}^{b} \text{ is a subnormal number, and} M_{fixp}^{b} > 0.5 \\
    E_{int}^{b}, & \text{ otherwise}
  \end{cases}
  $$

  Compute the block scale factor:

  $$
  S_{ue8m0}^{b}=2^{E_{int}^{b}}
  $$

  Compute the block conversion factor:

  $$
  R_{fp32}^{b}=\frac{1}{fp32(S_{ue8m0}^{b})}
  $$

  Perform quantization on each element in the block:

  $$
  p_i=cast\_to\_dst\_type(d^{i}\times R_{fp32}^{b},roundMode)
  $$

  Output the scale and quantization results corresponding to the block:

  $$
  (S_{ue8m0}^{b},[p_i]_{i=1}^{k})
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation flow, and then call the "aclnnGroupedDynamicMxQuantWithDualAxis" API to execute the computation.

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupIndex,
  char            *roundModeOptional,
  int64_t          scaleAlg,
  int64_t          dstDtype,
  double           maxDtypeValue,
  const aclTensor *y1Out,
  const aclTensor *y1ScaleOut,
  const aclTensor *y2Out,
  const aclTensor *y2ScaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantWithDualAxis(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize

- **Parameters**

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x (const aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor to be quantized, corresponding to input data D in the formula.</td>
      <td><ul><li>The shape is [M,N].</li><li>N must be greater than 0 and divisible by 64.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndex (const aclTensor*)</td>
      <td>Input</td>
      <td>Group quantization index, describing the boundaries of each group in cumsum form.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is [G].</li><li>Each element indicates the end row index of a group. The value must be non-negative and non-decreasing; the last element must be equal to M.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Optional input</td>
      <td>Rounding mode used to convert the quantization result to the target type of output y.</td>
      <td><ul><li>Optional input. Passing a null pointer indicates that "rint" is used.</li><li>If not empty, this parameter only supports taking the value "rint".</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Scale computation algorithm for y1ScaleOut and y2ScaleOut.</td>
      <td><ul><li>Only supports taking the value 1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstDtype (int64_t)</td>
      <td>Input</td>
      <td>Target output data type of y1Out and y2Out.</td>
      <td><ul><li>Only supports taking values 35 and 36, where 35 indicates FLOAT8_E5M2 and 36 indicates FLOAT8_E4M3FN.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxDtypeValue (double)</td>
      <td>Input</td>
      <td>Reserved parameter.</td>
      <td><ul><li>Only supports taking the value 0.0.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (const aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the corresponding result after quantizing x along the -1 axis.</td>
      <td><ul><li>The shape needs to be consistent with x, which is [M,N].</li><li>The data type is specified by the caller when pre-allocating the tensor, and needs to be consistent with dstDtype.</li></ul></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y1ScaleOut (const aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale corresponding to each group along the -1 axis.</td>
      <td><ul><li>The shape is [M,ceil(N/64),2].</li><li>Every 2 scales are stored in pairs in the last dimension; one scale is generated for every 32 elements in the N direction, with even padding and a pad value of 0.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2Out (const aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the result of quantizing x along axis -2.</td>
      <td><ul><li>The shape needs to be consistent with x, which is [M,N].</li><li>The data type is specified by the caller when pre-allocating the tensor, and needs to be consistent with dstDtype.</li></ul></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2ScaleOut (const aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale corresponding to each group along the -2 axis.</td>
      <td><ul><li>The shape is [floor(M/64)+G,N,2], where G is the number of elements in groupIndex.</li><li>One scale is generated per 32 elements along the second-to-last dimension, and even padding is applied.</li><li>The y2ScaleOut output needs to be interleaved for every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
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
      <td>Returns the op executor containing the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1056px"><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, groupIndex, y1Out, y1ScaleOut, y2Out, y2ScaleOut, workspaceSize, or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type, data format, or shape of x, groupIndex, y1Out, y1ScaleOut, y2Out, or y2ScaleOut is out of the supported range.</td>
    </tr>
    <tr>
      <td>x is not a 2D ND tensor, or x.shape[1] is not greater than 0 or is not divisible by 64.</td>
    </tr>
    <tr>
      <td>groupIndex is an empty tensor, is not a 1D ND tensor, has negative element values, does not meet the non-decreasing requirement, or its last element is not equal to x.shape[0].</td>
    </tr>
    <tr>
      <td>roundModeOptional, scaleAlg, dstDtype, or maxDtypeValue is out of the supported value range.</td>
    </tr>
    <tr>
      <td>The shape of the output tensor is inconsistent with the API requirements.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not within the supported platform range.</td>
    </tr>
  </tbody></table>

## aclnnGroupedDynamicMxQuantWithDualAxis

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Device-side workspace memory address.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Device-side workspace size, obtained by calling the first-phase API aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, including the computation flow of the contained operators.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- aclnnGroupedDynamicMxQuantWithDualAxis defaults to a deterministic implementation.
- N needs to be aligned to 64.
- The shapes of y1Out and y2Out need to be consistent with that of x, both [M,N]; the data types of y1Out and y2Out need to be consistent with the target type specified by dstDtype.
- The shape of y1ScaleOut is determined by x and needs to be [M,ceil(N/64),2]; the shape of y2ScaleOut is determined by both x and groupIndex and needs to be [floor(M/64)+G,N,2], where G is the number of elements in groupIndex.
- groupIndex uses the cumsum mode, where each element indicates the ending row index of the corresponding group. Each element value needs to be non-negative and non-decreasing, and the last element needs to be equal to M (that is, x.shape[0]). If M is 0, the last element is 0.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_dynamic_mx_quant_with_dual_axis.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        if (i != 0 && shapeSize > INT64_MAX / i) {
            return -1;
        }
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    *stream = nullptr;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}

class AclEnvironmentGuard {
public:
    AclEnvironmentGuard(int32_t deviceId, aclrtStream stream) : deviceId_(deviceId), stream_(stream) {}

    ~AclEnvironmentGuard()
    {
        aclrtDestroyStream(stream_);
        aclrtResetDevice(deviceId_);
        aclFinalize();
    }

private:
    int32_t deviceId_;
    aclrtStream stream_;
};

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    const int64_t shapeSize = GetShapeSize(shape);
    if (shapeSize <= 0 || hostData.size() != static_cast<size_t>(shapeSize) ||
        static_cast<uint64_t>(shapeSize) > std::numeric_limits<size_t>::max() / sizeof(T)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const size_t size = static_cast<size_t>(shapeSize) * sizeof(T);
    *deviceAddr = nullptr;
    *tensor = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

int CopyAndPrintOutput(const char* name, std::vector<uint8_t>& hostData, void* deviceAddr)
{
    auto ret = aclrtMemcpy(hostData.data(), hostData.size() * sizeof(uint8_t), deviceAddr,
                           hostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    constexpr size_t PRINT_COUNT = 10;
    std::printf("%s first 10 elements:\n", name);
    for (size_t i = 0; i < PRINT_COUNT && i < hostData.size(); i++) {
        std::printf("%s[%zu] = %u\n", name, i, static_cast<unsigned int>(hostData[i]));
    }
    return ACL_SUCCESS;
}

int aclnnGroupedDynamicMxQuantWithDualAxisTest(int32_t deviceId)
{
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    AclEnvironmentGuard environmentGuard(deviceId, stream);

    // x shape: [M, N] = [256, 1024], N must be 64-aligned
    constexpr int64_t M = 256;
    constexpr int64_t N = 1024;
    constexpr int64_t BLOCK_SIZE = 32;
    constexpr int64_t SCALE_PAIR = 2;
    constexpr int64_t SCALE_AXIS_UNIT = BLOCK_SIZE * SCALE_PAIR;
    constexpr int64_t G = 4; // 4 groups, each has 64 rows

    std::vector<int64_t> xShape = {M, N};
    std::vector<int64_t> groupIndexShape = {G};
    std::vector<int64_t> y1ScaleOutShape = {M, N / SCALE_AXIS_UNIT, SCALE_PAIR};
    std::vector<int64_t> y2ScaleOutShape = {M / SCALE_AXIS_UNIT + G, N, SCALE_PAIR};

    // groupIndex: cumsum, last element must equal M
    std::vector<int64_t> groupIndexHostData = {64, 128, 192, 256};

    // x data: BF16 (uint16_t storage), fill with deterministic pattern
    std::vector<uint16_t> xHostData(M * N, 0);
    for (int64_t i = 0; i < M * N; i++) {
        xHostData[i] = static_cast<uint16_t>((i % 200) - 100);
    }

    // output buffers
    std::vector<uint8_t> y1OutHostData(M * N, 0);
    std::vector<uint8_t> y2OutHostData(M * N, 0);
    std::vector<uint8_t> y1ScaleOutHostData(GetShapeSize(y1ScaleOutShape), 0);
    std::vector<uint8_t> y2ScaleOutHostData(GetShapeSize(y2ScaleOutShape), 0);

    // attributes
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t scaleAlg = 1;  // Ceil rounding algorithm
    int64_t dstDtype = 36; // FLOAT8_E4M3FN
    double maxDtypeValue = 0.0;

    void* xDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* y1OutDeviceAddr = nullptr;
    void* y1ScaleOutDeviceAddr = nullptr;
    void* y2OutDeviceAddr = nullptr;
    void* y2ScaleOutDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* y1Out = nullptr;
    aclTensor* y1ScaleOut = nullptr;
    aclTensor* y2Out = nullptr;
    aclTensor* y2ScaleOut = nullptr;

    // create x (BF16)
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create groupIndex (INT64)
    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64,
                          &groupIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupIndexTensorPtr(groupIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupIndexDeviceAddrPtr(groupIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y1Out (FLOAT8_E4M3FN)
    ret = CreateAclTensor(y1OutHostData, xShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y1ScaleOut (FLOAT8_E8M0)
    ret = CreateAclTensor(y1ScaleOutHostData, y1ScaleOutShape, &y1ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &y1ScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1ScaleOutTensorPtr(y1ScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1ScaleOutDeviceAddrPtr(y1ScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y2Out (FLOAT8_E4M3FN)
    ret = CreateAclTensor(y2OutHostData, xShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y2ScaleOut (FLOAT8_E8M0)
    ret = CreateAclTensor(y2ScaleOutHostData, y2ScaleOutShape, &y2ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &y2ScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2ScaleOutTensorPtr(y2ScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2ScaleOutDeviceAddrPtr(y2ScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // first-stage API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize(x, groupIndex, roundModeOptional, scaleAlg, dstDtype,
                                                                 maxDtypeValue, y1Out, y1ScaleOut, y2Out, y2ScaleOut,
                                                                 &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // allocate workspace
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // second-stage API
    ret = aclnnGroupedDynamicMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // sync
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CopyAndPrintOutput("y1Out", y1OutHostData, y1OutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y1ScaleOut", y1ScaleOutHostData, y1ScaleOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y2Out", y2OutHostData, y2OutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y2ScaleOut", y2ScaleOutHostData, y2ScaleOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

int main() { return aclnnGroupedDynamicMxQuantWithDualAxisTest(0); }

```
