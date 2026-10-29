# aclnnGroupedDynamicMxQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:32:20.316Z -->

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

- Description: Performs grouped float8 dynamic quantization on the input data based on the start values of the group indices.

- Formula:
  - First, group the input x along dimension 0 by groupIndex. Within each group, divide the elements into subgroups of k = blocksize elements. For each subgroup of k elements $\{\{x_i\}_{i=1}^{k}\}$ compute the quantization scale mxscale_pre of these elements, $\{mxscale\_pre, \{P_i\}_{i=1}^{k}\}$, as defined in Formulas (1) and (2) below.
  $$
  shared\_exp = floor(log_2(max_i(|V_i|))) - emax  \tag{1}
  $$
  $$
  mxscale\_pre = 2^{shared\_exp}  \tag{2}
  $$
  - Divide each element of the subgroup by mxscale_pre, convert the result to the corresponding dst_type according to round_mode, and obtain the quantization result y, as defined in Formula (3) below.
  $$
  P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize \tag{3}
  $$

   The quantized $P_i$ are placed at the positions of the corresponding $x_i$ to form the output y. mxscale_pre is grouped by the corresponding groupIndex. Within each group, the first dimension is padded to an even number to form the output mxscale.

  - emax: The exponent of the maximum normal number of the corresponding data type.

    | DataType | Emax |
    | :-----------: | :--: |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGroupedDynamicMxQuantGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation flow. Then call the "aclnnGroupedDynamicMxQuant" API to perform the computation.

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupIndex,
  const char      *roundMode,
  int64_t          dstType,
  int64_t          blocksize,
  const aclTensor *y,
  const aclTensor *mxscale,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedDynamicMxQuantGetWorkspaceSize

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
      <td>Input tensor of the operator. Corresponds to input x in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Start index of the quantization grouping.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Indices must be greater than or equal to 0 and non-decreasing. The last index must equal the first dimension size of x.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>Input</td>
      <td>round_mode in the Formula. Mode of data conversion.</td>
      <td>Only "rint" mode is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>dst_type in the formula. It specifies the type of y after data conversion.</td>
      <td>The input range is {35, 36}, corresponding to the data types of output y: {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blocksize (int64_t)</td>
      <td>Input</td>
      <td>blocksize in the formula. Specifies the number of elements quantized each time.</td>
      <td>Currently, only 32 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output tensor after quantization. Corresponds to y in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape dimensions are the same as those of x.</li><li>The supported data types are FLOAT8_E4M3FN and FLOAT8_E5M2, which must correspond to dstType.</li></ul></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale (aclTensor*)</td>
      <td>Output</td>
      <td>Output mxscale composed of mxscale_pre in the formula, that is, the quantization scale corresponding to each group.</td>
      <td><ul><li>Supports empty tensors.</li><li>If the shape of x is [m,n] and the shape of groupIndex is [g], the shape of mxscale is [(m/(blocksize∗2)+g),n,2].</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
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
      <th>Error Code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>If a passed-in parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
  </tr>
  <tr>
    <td>roundMode is a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of x, groupIndex, y, and mxscale are not in the supported range.</td>
  </tr>
  <tr>
    <td>The shapes of x, y, and mxscale do not meet the check conditions.</td>
  </tr>
  <tr>
    <td>The dimensions of x, groupIndex, y, and mxscale are not in the supported range.</td>
  </tr>
  <tr>
    <td>roundMode, dstType, and blocksize do not match the currently supported values.</td>
  </tr>
  <tr>
    <td>mxscale does not support non-contiguous tensors.</td>
  </tr>
  </tbody>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>The current platform is not in the range of supported platforms.</td>
  </tr>
  </table>

## aclnnGroupedDynamicMxQuant

- **Parameters:**

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
      <td>Address of the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnGroupedDynamicMxQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
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

## Constraints

- Deterministic computation:
  - aclnnGroupedDynamicMxQuant defaults to a deterministic implementation.

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_dynamic_mx_quant.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
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

int aclnnGroupedDynamicMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Custom construction is required based on the API.
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
    // Corresponding BF16 values (0, 8, 64, 512)
    std::vector<uint16_t> xHostData = {{0}, {16640}, {17024}, {17408}, {0}, {16640}, {17024}, {17408}};

    std::vector<uint32_t> groupIndexHostData = {4,8};
    // Corresponding float8_e4m3 values (0, 4, 32, 256)
    std::vector<uint8_t> yOutHostData = {{0}, {72}, {96}, {120}, {0}, {72}, {96}, {120}};
    // Corresponding float8_e8m0 values (2)
    std::vector<std::vector<uint8_t>> mxscaleOutHostData = {{{128, 0}}, {{128, 0}}};
    const char* roundModeOptional = "rint";
    int64_t dstType = 36;
    int64_t blocksize = 32;
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the groupedIndex aclTensor.
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

    // Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnGroupedDynamicMxQuant.
    ret = aclnnGroupedDynamicMxQuantGetWorkspaceSize(x, groupIndex, roundModeOptional, dstType, blocksize, yOut, mxscaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnGroupedDynamicMxQuant.
    ret = aclnnGroupedDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuant failed. ERROR: %d\n", ret); return ret);

    // (Boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
    auto size = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(
        size, 0);  // fp8 data cannot be printed directly in C. Read it as uint8 and manually convert the binary data to fp8.
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                    size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yOutData[i]);
    }
    size = GetShapeSize(mxscaleOutShape);
    std::vector<uint8_t> mxscaleOutData(
        size, 0);  // fp8 data cannot be printed directly in C. Read it as uint8 and manually convert the binary data to fp8.
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
    // 1. (boilerplate) Device/stream initialization. Refer to the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedDynamicMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
