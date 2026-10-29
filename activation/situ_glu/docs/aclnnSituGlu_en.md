# aclnnSituGlu

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T13:56:50.008Z -->

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

- Description: SiTU Gated Linear Unit activation function. Splits the input tensor x along the specified dimension into two halves (gate and up) and computes the output according to the SiTU formula.
- Formula:

  For a given input tensor x with dimensions [a, b, c, d, e, f, g, ...], aclnnSituGlu performs the following computation:

  1. Flatten x based on the input parameter dim. The dimensions after flattening are [pre, cut], where cut is the axis to be split into two tensors, and cut must be an even number. Let h = cut // 2.

  2. Split x into the front and back parts based on the input parameter activateLeft:

     - If activateLeft is true:

     $$
     gate = x[:, :h], \quad up = x[:, h:]
     $$

     - If activateLeft is false:

     $$
     gate = x[:, h:], \quad up = x[:, :h]
     $$

  3. Perform the SiTU computation based on the input parameters beta and linearBeta:

     $$
     situ\_a = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \text{sigmoid}(gate)
     $$

     When linearBeta > 0:

     $$
     up = linearBeta \cdot \tanh\left(\frac{up}{linearBeta}\right)
     $$

  4. Output:

     $$
     y = situ\_a \cdot up
     $$

  5. Reshape the output tensor y to have the same number of dimensions as x before concatenation, with the size on the dim axis being half of that of x, and other dimensions identical to x.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSituGluGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnSituGlu" API to perform the computation.

```cpp
aclnnStatus aclnnSituGluGetWorkspaceSize(
    const aclTensor *x,
    int64_t dim,
    double beta,
    double linearBeta,
    bool activateLeft,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnSituGlu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnSituGluGetWorkspaceSize

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
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input x in the formula.</td>
      <td><ul><li>Supports empty tensors (the output is the corresponding empty tensor).</li><li>shape must be even along the dimension corresponding to input parameter dim.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>Input dim in the formula, which indicates the dimension index for merging and splitting x.</td>
      <td>The value range is [-x.dim(), x.dim()-1].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta (double)</td>
      <td>Input</td>
      <td>The beta input in the formula, the scaling factor of the SiTU gated part.</td>
      <td>A non-zero value is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linearBeta (double)</td>
      <td>Input</td>
      <td>linearBeta in the formula, the scaling coefficient of linear tanh in the up path. If the value is less than or equal to 0, no transformation is performed on the up path.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Input activateLeft in the formula. Indicates whether gate is the first half when x is split.</td>
      <td>true indicates that gate is the first half and up is the second half; false indicates that gate is the second half and up is the first half.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output y in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is half of that of x on the dimension corresponding to input parameter dim, and the same as x on other dimensions.</li><li>The data type is the same as that of x.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be applied on the Device side.</td>
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
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
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
      <td>x or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of x or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape dimension of x or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>dim is not within the specified value range, or x is not divisible by 2 on the dimension corresponding to dim.</td>
    </tr>
    <tr>
      <td>The number of dimensions of out is inconsistent with that of x, or the size of out on the dimension corresponding to dim is not equal to half of x.</td>
    </tr>
    <tr>
      <td>The data type of out does not match that of x.</td>
    </tr>
  </tbody></table>

## aclnnSituGlu

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>Workspace memory address applied on the side of the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the side of the Device, obtained by the first-phase API aclnnSituGluGetWorkspaceSize.</td>
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
  </tbody></table>

- **Return Value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnSituGlu defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_situ_glu.h"

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
    // Boilerplate, acl initialization
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

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (boilerplate) Device/stream initialization. For details, see the ACL API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output.
    std::vector<int64_t> xShape = {2, 32};
    std::vector<int64_t> outShape = {2, 16};
    void* xDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> xHostData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int64_t dim = -1;
    double beta = 1.0;
    double linear_beta = 0.0;
    bool activate_left = true;
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnSituGlu.
    ret = aclnnSituGluGetWorkspaceSize(x, dim, beta, linear_beta, activate_left, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory according to the workspaceSize calculated by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnSituGlu
    ret = aclnnSituGlu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGlu failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait synchronously for the task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // 6. Release the aclTensor
    aclDestroyTensor(x);
    aclDestroyTensor(out);
    // 7. Release device resources
    aclrtFree(xDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
