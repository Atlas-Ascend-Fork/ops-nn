# aclnnTransQuantParamV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:44:48.898Z -->

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

- Description: Converts the data type of the quantization parameter scale from FLOAT32 to UINT64 or INT64 as required by the hardware.
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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnTransQuantParamV2GetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation flow. Then call the "aclnnTransQuantParamV2" API to execute the computation.

```Cpp
aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(
  const aclTensor* scale,
  const aclTensor* offset,
  const aclTensor* out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnTransQuantParamV2(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnTransQuantParamV2GetWorkspaceSize

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>The scale value in quantization. Corresponds to `scale` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Ensure that the scale data does not contain NaN or Inf.</li><li>For shape constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>offset (aclTensor*)</td>
      <td>Input</td>
      <td>Optional parameter. The offset value in quantization. Corresponds to `offset` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Ensure that the offset data contains no NaN or Inf.</li><li>For shape constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Calculation output of quantization. Corresponding to `out` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>For shape constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>UINT64, INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√ </td>
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
      <td>Returns the op executor, which contains the computation process of the operator.</td>
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>scale or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type and data format of scale, offset, or out are not within the supported range.</td>
    </tr>
    <tr>
      <td>An error is reported if the shape of offset is not (t,) or (1, n). t is 1 or n, and n is the same as n of x2 in matmul computation (or the same as shape n of the right matrix (Corresponding Parameter weight) in grouped matmul computation).</td>
    </tr>
    <tr>
      <td>If the shape of scale is not 1-D (t,) or (g,) or 2-D (1, n), (g, 1), or (g, n), an error is reported. Specifically: t = 1 or n; n is the same as the n of x2 in the matmul computation (or the same as the shape n of the right matrix (corresponding parameter: weight) in the grouped matmul computation); g is the same as the number of groups in the grouped matmul computation (the size of the shape of the corresponding parameter groupListOptional).</td>
    </tr>
  </tbody></table>

## aclnnTransQuantParamV2

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
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnTransQuantParamV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes the tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="950,A3,910b,310p" id7 -->
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: This API can be used together with matmul operators (such as [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md)).
<!-- end id7 -->
<!-- npu="A3,910b,310p" id8 -->
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This API does not support use with grouped matmul class operators (such as aclnnGroupedMatmulV4).
<!-- end id8 -->
- The shapes of scale, offset, and out are described as follows:
  - When there is no offset, the out shape is the same as the scale shape.
    - If out is used as the input of a matmul operator (such as [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md)), its shape can be 1D (1,), (n,), or 2D (1, n). n must be the same as the shape n of the right matrix in the matmul computation (corresponding parameter x2).
    - If out is used as a grouped matmul class operator input (such as aclnnGroupedMatmulV4), it is only used when the grouping mode is m-axis grouping (corresponding parameter groupType is 0). Its shape supports 1D (g,) or 2D (g, 1), (g, n), where n is consistent with the shape n of the right matrix (corresponding parameter x2) in the grouped matmul computation, and g is consistent with the number of groups (corresponding parameter groupListOptional's shape size) in the grouped matmul computation.
  - If offset exists, it can only be used as the input of a matmul operator (such as [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md)):
    - The shapes of offset, scale, and out support 1D (1,), (n,) or 2D (1, n), where n is consistent with the shape n of the right matrix (corresponding parameter x2) in the matmul computation.
    - When the input scale's shape is 1D, the out shape is also 1D, and its shape size is the maximum of the single-dimension shape sizes of scale and offset.
    - When the shape of the input scale is 2D, the shape of out has exactly the same dimensions and size as the shape of the input scale.
- Deterministic computation:
  - aclnnTransQuantParamV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <memory>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnTransQuantParamV2Test(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs as required by the API.
    std::vector<int64_t> offsetShape = {3};
    std::vector<int64_t> scaleShape = {3};
    std::vector<int64_t> outShape = {3};
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> scaleHostData = {1, 1, 1};
    std::vector<float> offsetHostData = {1, 1, 1};
    std::vector<uint64_t> outHostData = {1, 1, 1};
    // Create the scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT64, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // Call the first-phase API of aclnnTransQuantParamV2.
    ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory for the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnTransQuantParamV2.
    ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device-side memory to the host side. Modify it based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint64_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %lu\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnTransQuantParamV2Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2Test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
