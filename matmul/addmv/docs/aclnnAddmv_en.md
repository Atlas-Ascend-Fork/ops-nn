# aclnnAddmv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:41:57.414Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Performs matrix-vector multiplication and then adds the result to a vector.
- Formula:

  $$
  out = β  self + α  (mat @ vec)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAddmvGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the computation process, and then call the "aclnnAddmv" API to perform the computation.

```Cpp
aclnnStatus aclnnAddmvGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* mat,
  const aclTensor* vec,
  const aclScalar* alpha,
  const aclScalar* beta,
  aclTensor*       out,
  int8_t           cubeMathType,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAddmv(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnAddmvGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>self</td>
      <td>Input</td>
      <td>1-D vector to be added to the subsequent multiplication result.</td>
      <td><ul><li>The data type must have a <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> with mat@vec.</li>
      <li>If alpha is not 0, the shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with mat@vec.</li>
      <li>If alpha is 0, the shape must be the same as mat@vec.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT64, INT16, INT8, UINT8, DOUBLE, BOOL</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat</td>
      <td>Input</td>
      <td>2-D matrix to be multiplied with vec.</td>
      <td><ul><li>Data type must form a <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> with self.</li>
      <li>shape must satisfy a multiplication relationship with vec.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT64, INT16, INT8, UINT8, DOUBLE, BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vec</td>
      <td>Input</td>
      <td>A 1-dimensional vector multiplied with mat.</td>
      <td><ul><li>Data type must form a <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> with self.</li>
      <li>shape must satisfy a multiplication relationship with mat.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT64, INT16, INT8, UINT8, DOUBLE, BOOL</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>α in the formula, coefficient of the product of mat and vec.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>β in the formula, the coefficient of self.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Specified 1-D output vector.</td>
      <td><ul><li>Data type must be the <a href="../../../docs/en/context/deduction_relationship.md">deduced data type</a> of self, mat, vec, alpha, and beta.</li>
      <li>Shape is the same as that of the product of mat and vec.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32, INT64, INT16, INT8, UINT8, DOUBLE</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter defaults to processing the deduced data type. The supported enumeration values are as follows:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, allows downgrading the input data precision for computation.</li>
        <li>2: USE_FP16, allows downgrading the input precision to FLOAT16 for computation.</li>
        <li>3: USE_HF32, allows downgrading the input precision to the data type HFLOAT32 for computation.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
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

  <!-- npu="950,A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.

  <!-- end id7 -->
  <!-- npu="910,310p" id8 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 is not supported.

  <!-- end id8 -->

- **Return Value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
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
      <td>self, mat, vec, alpha, beta, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types and data formats of self, mat, and vec are not in the supported range.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed on self, mat, and vec.</td>
    </tr>
    <tr>
      <td>The deduced data type/data format cannot be converted to the type/format of the specified output out.</td>
    </tr>
    <tr>
      <td>The shapes of mat and vec do not meet the multiplication requirements, or self and the multiplication result do not meet the addition requirements.</td>
    </tr>
  </tbody>
  </table>

## aclnnAddmv

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Workspace memory address applied for on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory applied for on the Device, obtained by the first-phase API aclnnAddmvGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism description:

  <!-- npu="910,310p" id9 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: aclnnAddmv defaults to a non-deterministic implementation. Deterministic computation can be enabled by calling aclrtCtxSetSysParamOpt.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnAddmv defaults to a deterministic implementation.

  <!-- end id10 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addmv.h"

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
  // Boilerplate, resource initialization
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device.
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

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {2};
  std::vector<int64_t> matShape = {2, 2};
  std::vector<int64_t> vecShape = {2};
  std::vector<int64_t> outShape = {2};
  void* selfDeviceAddr = nullptr;
  void* matDeviceAddr = nullptr;
  void* vecDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat = nullptr;
  aclTensor* vec = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* beta = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 1};
  std::vector<float> matHostData = {1, 1, 1, 1};
  std::vector<float> vecHostData = {1, 1};
  std::vector<float> outHostData(2, 0);
  int8_t cubeMathType = 1;
  float alphaValue = 1.0f;
  float betaValue = 1.0f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mat aclTensor.
  ret = CreateAclTensor(matHostData, matShape, &matDeviceAddr, aclDataType::ACL_FLOAT, &mat);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the vec aclTensor.
  ret = CreateAclTensor(vecHostData, vecShape, &vecDeviceAddr, aclDataType::ACL_FLOAT, &vec);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create the beta aclScalar.
  beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAddmv.
  ret = aclnnAddmvGetWorkspaceSize(self, mat, vec, alpha, beta, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmvGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAddmv.
  ret = aclnnAddmv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmv failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from the device to the host. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy the aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(mat);
  aclDestroyTensor(vec);
  aclDestroyScalar(alpha);
  aclDestroyScalar(beta);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(matDeviceAddr);
  aclrtFree(vecDeviceAddr);
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
