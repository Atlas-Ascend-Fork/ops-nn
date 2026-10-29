# aclnnIndexCopy&aclnnInplaceIndexCopy

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:58:54.330Z -->

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

Uses the element values in the index tensor as indices to copy elements from source to the corresponding positions in selfRef or out along the specified axis dim.

## Function Prototype

- aclnnIndexCopy and aclnnInplaceIndexCopy implement the same function. The differences in usage are as follows. Select the appropriate API based on your actual scenario.
  - aclnnIndexCopy: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceIndexCopy: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnIndexCopyGetWorkspaceSize" or "aclnnInplaceIndexCopyGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnIndexCopy" or "aclnnInplaceIndexCopy" API to perform the computation.

```Cpp
aclnnStatus aclnnIndexCopyGetWorkspaceSize(
  aclTensor*        selfRef,
  int64_t           dim,
  const aclTensor*  index,
  const aclTensor*  source,
  aclTensor*        outRef,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnIndexCopy(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnInplaceIndexCopyGetWorkspaceSize(
 aclTensor*       selfRef,
 int64_t          dim,
 const aclTensor* index,
 const aclTensor* source,
 uint64_t*        workspaceSize,
 aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceIndexCopy(
 void             *workspace,
 uint64_t          workspaceSize,
 aclOpExecutor    *executor,
 aclrtStream       stream)
```

## aclnnIndexCopyGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1441px"><colgroup>
    <col style="width: 145px">
    <col style="width: 120px">
    <col style="width: 230px">
    <col style="width: 226px">
    <col style="width: 294px">
    <col style="width: 119px">
    <col style="width: 162px">
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
        <td>selfRef</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>-</td>
        <td>FLOAT, BFLOAT16, FLOAT16, INT32, UINT32, INT64, UINT64, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX128, COMPLEX64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Axis along which the index copy is performed.</td>
        <td>In the range [-selfRef.dim(), selfRef.dim() - 1].</td>
        <td>int64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td><ul><li>The number of dimensions cannot be greater than 1. The number of elements must be the same as the size of the axis dim in tensor source.</li><li>If index contains duplicate index values, the order of the result is not preserved.</li><li>Out-of-bounds index data in index is not supported.</li></ul></td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>source</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>The size of the dim axis is the same as the number of elements in index, and other dimensions are consistent with the shape of selfRef.</td>
        <td>Consistent with selfRef</td>
        <td>ND</td>
        <td>Consistent with selfRef</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outRef</td>
        <td>Output</td>
        <td>Output aclTensor.</td>
        <td>-</td>
        <td>Consistent with selfRef</td>
        <td>-</td>
        <td>Consistent with selfRef</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: UINT32 and UINT64 are not supported.
    <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input selfRef, index, source, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type of selfRef or index is out of the supported range.</td>
      </tr>
      <tr>
      <td>The data types of selfRef and source are different.</td>
      </tr>
      <tr>
      <td>The size of source on the dim axis is not equal to the number of elements of index.</td>
      </tr>
      <tr>
      <td>The shapes of selfRef and source have different sizes on axes other than dim.</td>
      </tr>
      <tr>
      <td>The number of dimensions of index is greater than 1.</td>
      </tr>
      <tr>
      <td>When source is a scalar, the number of elements of index is not 1.</td>
      </tr>
      <tr>
      <td>source and selfRef are not scalars and have different dimensions.</td>
      </tr>
      <tr>
      <td>selfRef and out differ in shape or data type.</td>
      </tr>
      <tr>
      <td>dim is out of range.</td>
      </tr>
    </tbody>
    </table>

## aclnnIndexCopy

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnIndexCopyGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceIndexCopyGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1441px"><colgroup>
    <col style="width: 145px">
    <col style="width: 120px">
    <col style="width: 230px">
    <col style="width: 226px">
    <col style="width: 294px">
    <col style="width: 119px">
    <col style="width: 162px">
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
        <td>selfRef</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>-</td>
        <td>FLOAT, BFLOAT16, FLOAT16, INT32, UINT32, INT64, UINT64, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX128, COMPLEX64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>Within the range [-selfRef.dim(), selfRef.dim() - 1].</td>
        <td>int64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td><ul><li>The dimension cannot be greater than 1. The number of elements must be the same as the size of the dim axis in tensor source.</li><li>When index contains duplicate index values, the order of the result is not guaranteed.</li></ul></td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>source</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>The size of the dim axis is the same as the number of elements in index, and other dimensions are the same as the shape of selfRef.</td>
        <td>consistent with selfRef</td>
        <td>ND</td>
        <td>consistent with selfRef</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be applied for on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: UINT32 and UINT64 are not supported.
    <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed selfRef, index, source, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of selfRef or index is not within the supported range.</td>
      </tr>
      <tr>
      <td>selfRef and source have different data types.</td>
      </tr>
      <tr>
      <td>The size of the dim axis of source is not equal to the number of elements of index.</td>
      </tr>
      <tr>
      <td>The sizes of the shapes of selfRef and source on axes other than dim are different.</td>
      </tr>
      <tr>
      <td>When the dimension of index is greater than 1.</td>
      </tr>
      <tr>
      <td>When source is a scalar, the number of elements in index is not 1.</td>
      </tr>
      <tr>
      <td>When source and selfRef are not scalars, their dimensions are different.</td>
      </tr>
      <tr>
      <td>When dim is out of range.</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceIndexCopy

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Workspace memory address allocated on the Device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device, obtained via the first-phase API aclnnInplaceIndexCopyGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: default to a nondeterministic implementation. You can enable determinism through `aclrtSetSysParamOpt`.
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: default to a deterministic implementation.
  <!-- end id10 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_copy.h"

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
    // Call aclrtMalloc to apply for memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> indexShape = {4};
    std::vector<int64_t> sourceShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* sourceDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* index = nullptr;
    aclTensor* source = nullptr;
    aclTensor* out = nullptr;
    int64_t dim = 0;
    std::vector<float> selfHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> indexHostData = {3, 2, 1, 0};
    std::vector<float> sourceHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the index aclTensor.
    ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the source aclTensor.
    ret = CreateAclTensor(sourceHostData, sourceShape, &sourceDeviceAddr, aclDataType::ACL_FLOAT, &source);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnIndexCopy.
    ret = aclnnIndexCopyGetWorkspaceSize(self, dim, index, source, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnIndexCopy.
    ret = aclnnIndexCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexCopy failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait synchronously until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value. Copy the result from device memory to host memory. Modify according to the actual API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("aclnnIndexCopy result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify according to the actual API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(index);
    aclDestroyTensor(source);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify it according to the definition of the specific API.
    aclrtFree(selfDeviceAddr);
    aclrtFree(indexDeviceAddr);
    aclrtFree(sourceDeviceAddr);
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
