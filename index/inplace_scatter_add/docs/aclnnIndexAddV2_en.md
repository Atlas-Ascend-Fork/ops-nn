# aclnnIndexAddV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:26:28.480Z -->

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

Adds values from the source tensor to the values at the corresponding positions in the input tensor along a specified dimension based on the given indices.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnIndexAddV2GetWorkspaceSize" to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnIndexAddV2" to perform computation.

```Cpp
aclnnStatus aclnnIndexAddV2GetWorkspaceSize(
 const aclTensor*  self,
 const int64_t     dim,
 const aclTensor*  index,
 const aclTensor*  source,
 const aclScalar*  alpha,
 int64_t           mode,
 aclTensor*        out,
 uint64_t*         workspaceSize,
 aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnIndexAddV2(
 void*          workspace,
 uint64_t       workspaceSize,
 aclOpExecutor* executor,
 aclrtStream    stream)
```

## aclnnIndexAddV2GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1451px"><colgroup>
    <col style="width: 143px">
    <col style="width: 120px">
    <col style="width: 227px">
    <col style="width: 265px">
    <col style="width: 274px">
    <col style="width: 117px">
    <col style="width: 160px">
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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, INT32, INT16, INT8, UINT8, DOUBLE, INT64, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>The specified dimension.</td>
        <td>Value range: [-self.dim(), self.dim()-1].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index (aclTensor*)</td>
        <td>Input</td>
        <td>Index.</td>
        <td>The shape size of index must be equal to the shape value of source on the dim dimension.</td>
        <td>INT64, INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>source (aclTensor*)</td>
        <td>Input</td>
        <td>Source tensor.</td>
        <td>Except for the dim dimension, other dimension values of the source shape must be equal to those of the self shape.</td>
        <td>Consistent with self</td>
        <td>ND</td>
        <td>Consistent with self</td>
        <td>√</td>
      </tr>
      <tr>
        <td>alpha (aclScalar*)</td>
        <td>Input</td>
        <td>Scaling factor.</td>
        <td>The data type can be converted to the data type after <a href="../../../docs/en/context/deduction_relationship.md">deduction</a> between self and source.<br>In a non-deterministic computation use case where mode is 0, alpha does not participate in the computation.</td>
        <td>-</td>
        <td>Consistent with self</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>mode (int64_t)</td>
        <td>Input</td>
        <td>Computation mode.</td>
        <td>When the value is 0, the high-performance mode is used. Other values indicate the non-high-performance mode. For limitations of the high-performance mode, see Constraints.<br>In deterministic computation use cases, the value of mode is ignored and the deterministic computation mode is used.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Output aclTensor.</td>
        <td>-</td>
        <td>Consistent with self</td>
        <td>ND</td>
        <td>Consistent with self</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation flow.</td>
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
      <td>self, index, source, alpha, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data types of self, index, source, and out are not within the supported range.</td>
      </tr>
      <tr>
      <td>The data types of self, source, and out are not consistent.</td>
      </tr>
      <tr>
      <td>The value of dim is greater than the shape size of self.</td>
      </tr>
      <tr>
      <td>The computed data type cannot be converted to the data type of the specified output out.</td>
      </tr>
      <tr>
      <td>Except for the dim dimension, the shape values of self and source are not equal on other dimensions.</td>
      </tr>
      <tr>
      <td>index is not 1-D.</td>
      </tr>
      <tr>
      <td>The shape size of index is not equal to the shape size of source on dimension dim.</td>
      </tr>
      <tr>
      <td>The shape of out is not equal to the shape of self.</td>
      </tr>
      <tr>
      <td>The value range of index is not within [0,self.shape[dim]).</td>
      </tr>
      <tr>
      <td>When mode=0, the high-performance mode restrictions in Constraints are not met.</td>
      </tr>
    </tbody>
    </table>

## aclnnIndexAddV2

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
      <td>Workspace memory address requested on the device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained by the first-phase API aclnnIndexAddV2GetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. It contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnIndexAddV2 defaults to a non-deterministic implementation. You can enable deterministic computation through aclrtCtxSetSysParamOpt.
- Index value range
  - The value range of the index input is [0,self.shape[dim]), that is, the index value range is the size of self along dim. Negative indices and out-of-bounds indices are not supported.
- High-performance mode (mode=0) has the following additional constraints:
  - dim must be 0 or -2.
  - The data type of self must be FLOAT, FLOAT16, INT32, INT16, or BFLOAT16.
  - self must be a two-dimensional tensor.
  - If mode=0 and the constraints in the parameter description are met but the above three constraints are not, the API throws an error.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_add_v2.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization.
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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
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

int main() {
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
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
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> indexHostData = {0, 1, 2, 3};
  std::vector<float> sourceHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  int64_t dim = 0;
  int64_t mode = 0;
  float alphaValue = 1.0f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the source aclTensor.
  ret = CreateAclTensor(sourceHostData, sourceShape, &sourceDeviceAddr, aclDataType::ACL_FLOAT, &source);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnIndexAddV2.
  ret = aclnnIndexAddV2GetWorkspaceSize(self, dim, index, source, alpha, mode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexAddV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnIndexAddV2.
  ret = aclnnIndexAddV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexAddV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Change it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Change it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(source);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the specific API definition.
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
