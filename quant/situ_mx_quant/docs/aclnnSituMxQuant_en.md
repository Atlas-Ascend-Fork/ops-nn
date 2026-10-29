# aclnnSituMxQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:39:19.182Z -->

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

- Description: Fuses the Situ activation function and dynamic MX (Microscaling) quantization into a single operator. It applies Situ activation to the input data x, applies MX quantization to the activated result, and outputs the quantized result and scale.

- Formula:

1. Situ Activation

$$
situ_a = \beta \times \tanh(gate / \beta) \times sigmoid(gate)
$$

When linear_beta > 0:

$$
up = linear\_beta \times \tanh(up / linear\_beta)
$$

$$
situOut = situ_a \times up
$$

    where, when activate_left is true, gate takes the first half of x and up takes the second half; when activate_left is false, gate takes the second half of x and up takes the first half.

2. MX quantization (OCP algorithm)

$$
shared\_exp = floor(log2(max(|situOut_i|))) - emax
$$

$$
y\_scale = 2^{shared\_exp}  (E8M0)
$$

$$
y = cast\_to\_fp8(situOut / y\_scale)
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSituMxQuantGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation flow, and then call the "aclnnSituMxQuant" API to execute the computation.

```Cpp
aclnnStatus aclnnSituMxQuantGetWorkspaceSize(
    const aclTensor*  x,
    double            beta,
    double            linearBeta,
    bool              activateLeft,
    int64_t           axis,
    int64_t           dstType,
    char*             roundModeOptional,
    const aclTensor*  yOut,
    const aclTensor*  yScaleOut,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnSituMxQuant(
    void*            workspace,
    uint64_t         workspaceSize,
    aclOpExecutor*   executor,
    aclrtStream      stream)
```

## aclnnSituMxQuantGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 320px">
  <col style="width: 177px">
  <col style="width: 104px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input x in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape supports 1-8 dimensions.</li><li>The last dimension must be a multiple of 2.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (double)</td>
      <td>Input</td>
      <td>The beta parameter for Situ activation.</td>
      <td>Must be greater than 0. Defaults to 1.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linearBeta (double)</td>
      <td>Input</td>
      <td>The linear_beta parameter of Situ activation.</td>
      <td>Not enabled when the value is ≤ 0. Defaults to 0.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Whether to apply Situ activation to the left half of the input.</td>
      <td>If the value is false, activation is applied to the right half of the input. Default: false.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Quantization axis.</td>
      <td>Currently, only -1 is supported. Defaults to -1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Data type of output y.</td>
      <td>Input range: {35, 36}, corresponding to {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}. Defaults to 36.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Quantization rounding mode.</td>
      <td>Supports {"rint", "round", "floor"}. FP8 output supports only "rint". Default "rint".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output, which is y in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is x.shape[:-1]+[H], where H=x.shape[-1]/2.</li></ul></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yScaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>MX quantization scale (E8M0 format).</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is x.shape[:-1]+[ceil(H/64), 2], where H=x.shape[-1]/2.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-9</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
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

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

The first-phase API performs input parameter validation and returns an error in the following scenarios:

<table style="undefined;table-layout: fixed;width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
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
    <td>The inputs x, yOut, and yScaleOut are null pointers.</td>
  </tr>
  <tr>
    <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="10">161002</td>
    <td>x, yOut, or yScaleOut is an empty tensor.</td>
  </tr>
  <tr>
    <td>The data type of x is out of the supported range.</td>
  </tr>
  <tr>
    <td>axis is not -1.</td>
  </tr>
  <tr>
    <td>dstType is out of the supported range.</td>
  </tr>
  <tr>
    <td>roundModeOptional is out of the supported range.</td>
  </tr>
  <tr>
    <td>The last dimension of x is not a multiple of 2.</td>
  </tr>
  <tr>
    <td>beta is not greater than 0.</td>
  </tr>
  <tr>
    <td>The data type of yOut does not match dstType, or the data type of yScaleOut is not FLOAT8_E8M0.</td>
  </tr>
  <tr>
    <td>The shape of yOut or yScaleOut is inconsistent with the deduction result.</td>
  </tr>
  <tr>
    <td>The data format of x, yOut, or yScaleOut is not ND.</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>The current platform is out of the supported platform range.</td>
  </tr>
</tbody></table>

## aclnnSituMxQuant

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
      <td>Size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnSituMxQuantGetWorkspaceSize.</td>
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

- **Return value**

  aclnnStatus: returns a status code. For details, see [aclnn return codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSituMxQuant defaults to a deterministic implementation.

- The last dimension of x must be a multiple of 2.
- The number of dimensions of x must be greater than or equal to 1.
- axis currently supports only -1 (quantization along the last axis).
- dstType supports 36 (FLOAT8_E4M3FN) or 35 (FLOAT8_E5M2).
- roundModeOptional must be "rint".
- The data type of yOut must match dstType, and the data type of yScaleOut must be FLOAT8_E8M0.
- The shapes of yOut and yScaleOut must be consistent with the deduction results (see Parameters).
- The shape constraints of yScaleOut are as follows:
  - H = x.shape[-1] / 2
  - scaleNum = ceil(H / 64)
  - yScaleOut.shape = x.shape[:-1] + [scaleNum, 2]

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

#include "acl/acl.h"
#include "aclnnop/aclnn_situ_mx_quant.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

uint16_t FloatToBf16Bits(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

int main()
{
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Customize the construction based on the API.
    std::vector<int64_t> xShape = {4, 128};
    std::vector<int64_t> yShape = {4, 64};
    std::vector<int64_t> yScaleShape = {4, 1, 2};

    // Generate BF16 input data.
    std::vector<uint16_t> xHostData(GetShapeSize(xShape));
    for (int64_t i = 0; i < GetShapeSize(xShape); i++) {
        float val = ((i * 17 + 3) % 200 - 100) / 50.0f;
        xHostData[i] = FloatToBf16Bits(val);
    }
    std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<uint8_t> yScaleHostData(GetShapeSize(yScaleShape), 0);

    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScale = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &yScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = false;
    int64_t axis = -1;
    int64_t dstType = 36;

    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnSituMxQuant.
    ret = aclnnSituMxQuantGetWorkspaceSize(x, beta, linearBeta, activateLeft, axis, dstType,
                                           const_cast<char*>("rint"),
                                           y, yScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnSituMxQuant.
    ret = aclnnSituMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituMxQuant failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, copy the result from device memory to host memory, and modify them based on the specific API definition.
    auto yResultSize = GetShapeSize(yShape);
    std::vector<uint8_t> yResult(yResultSize, 0);
    ret = aclrtMemcpy(yResult.data(), yResult.size(), yDeviceAddr,
                      yResultSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yResult from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < yResultSize; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yResult[i]);
    }

    // 6. Release aclTensor and aclScalar, and modify them based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(yScale);

    // 7. Release device resources, and modify them based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(yScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
