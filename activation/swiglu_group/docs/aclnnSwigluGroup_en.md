# aclnnSwigluGroup

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T15:11:33.498Z -->

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

- Description: Executes the SwiGLU activation and outputs the activation result with the same data type as input x.

- Formula:

  Let the size of the last dimension of input x be D. The left half is A, and the right half is B (that is, `h = D // 2`, `A = x[..., :h]`, and `B = x[..., h:]`).

  When `clampLimit` is greater than 0, truncate A and B first:

  $$
  A = min(A, clampLimit)
  $$

  $$
  B = min(max(B, -clampLimit), clampLimit)
  $$

  Calculate SwiGLU:

  $$
  y = silu(A) \times B = \frac{A}{1 + e^{-A}} \times B
  $$

  When `weightOptional` is not empty, multiply y by the weight token by token:

  $$
  y = y \times weightOptional
  $$

  When `groupIndexOptional` is not empty, the count mode controls the number of tokens actually processed: the number of tokens actually processed is the smaller value between the sum of all elements in `groupIndexOptional` and `bs`, where `bs` is the product of the dimensions of x except the last dimension.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSwigluGroupGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call the "aclnnSwigluGroup" API to perform the computation.

```Cpp
aclnnStatus aclnnSwigluGroupGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *groupIndexOptional,
  double           clampLimit,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor);
```

```Cpp
aclnnStatus aclnnSwigluGroup(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream);
```

## aclnnSwigluGroupGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 300px">
  <col style="width: 360px">
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 120px">
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
      <td>The input x in the formula. The last dimension is evenly split into two parts, A and B.</td>
      <td>Null pointers are not supported, but empty tensors are supported. The dimension must be greater than 0, and the last dimension must be an even number.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>The input weightOptional in the formula. Per-token weight.</td>
      <td>Supports null pointers. Supports empty tensors. If the pointer is not null, the number of elements must equal the product of the dimensions of x excluding the last dimension (bs).</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Input groupIndexOptional in the formula. Number of tokens per group in count mode.</td>
      <td>Supports null pointers, but does not support a standalone empty tensor. If the pointer is not null, the tensor must be 1-dimensional, and its elements must be greater than or equal to 0. The i-th element indicates the number of tokens of x to be processed for the i-th group. The sum of all elements, after truncation, is the actual number of tokens processed.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>clampLimit (double)</td>
      <td>Input</td>
      <td>Input clampLimit in the formula, which specifies the upper bound for clamping A and B before activation.</td>
      <td>Clamps A and B before activation. Defaults to -1.0, which indicates no clamping. If clamping is enabled, a value greater than 0 must be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output y in the formula, the SwiGLU computation result.</td>
      <td>The data type is consistent with x. The last dimension is half of the last dimension of x, and other dimensions are consistent with x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, containing the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>N/A</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>The passed-in x or out is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of x or out is not supported, or the data type of out is not consistent with that of x.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="5">561002</td>
      <td>x is a 0-dimensional tensor (scalar), or the last dimension of x is not an even number.</td>
    </tr>
    <tr>
      <td>clampLimit is not -1.0 and not greater than 0.</td>
    </tr>
    <tr>
      <td>The data type of weightOptional is not FLOAT32, or the number of elements in weightOptional is not consistent with the product of the dimensions of x excluding the last dimension.</td>
    </tr>
    <tr>
      <td>The data type of groupIndexOptional is not INT64, or groupIndexOptional alone is an empty tensor.</td>
    </tr>
    <tr>
      <td>The shape and dtype of other input tensors or the input attributes are not within the valid value ranges (see Parameters and Constraints for details).</td>
    </tr>
  </tbody>
  </table>

## aclnnSwigluGroup

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
      <td>Memory address of the workspace allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnSwigluGroupGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator calculation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Non-contiguous tensors are not supported: inputs x, weightOptional, groupIndexOptional, and output out must all be contiguous tensors.
- The last dimension of input x must be an even number (evenly split into two parts, A and B, and the last dimension of the output is half of the input).
- The data type of input x supports only FLOAT16, BFLOAT16, and FLOAT32, and the data type of out must be consistent with that of x.
- Deterministic computation: aclnnSwigluGroup defaults to a deterministic implementation. Non-deterministic implementation is not supported currently.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group.h"

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
    // Boilerplate, acl initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the ACL external API list.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs as required by the API.
    std::vector<int64_t> xShape = {2, 256};
    std::vector<int64_t> yShape = {2, 128};

    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = static_cast<uint16_t>(i % 23);
    }
    std::vector<uint16_t> yHostData(GetShapeSize(yShape), 0);

    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // weightOptional and groupIndexOptional are optional inputs. Passing nullptr here indicates that they are not used.
    double clampLimit = -1.0;

    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // Call the first-phase API of aclnnSwigluGroup.
    ret = aclnnSwigluGroupGetWorkspaceSize(x, nullptr, nullptr, clampLimit, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate Device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnSwigluGroup.
    ret = aclnnSwigluGroup(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroup failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from Device memory to the Host side.
    std::vector<uint16_t> resultData(GetShapeSize(yShape), 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return ret);
    LOG_PRINT("result[0] is: %u\n", resultData[0]);

    // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(y);

    // 7. Release Device resources. Modify them based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
