# aclnnBatchMatMulWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:51:57.602Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description: Performs matrix multiplication between tensor self and tensor mat2. mat2 only supports the NZ format. Only supports self with 3 dimensions and mat2 with 5 dimensions.

- Formula:

  $$
  out = self@mat2
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnBatchMatMulWeightNzGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size based on the process, and then call "aclnnBatchMatMulWeightNz" to perform the computation.

```cpp
aclnnStatus aclnnBatchMatMulWeightNzGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *mat2,
  aclTensor       *out,
  int8_t           cubeMathType,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnBatchMatMulWeightNz(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnBatchMatMulWeightNzGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 200px">
  <col style="width: 480px">
  <col style="width: 200px">
  <col style="width: 111px">
  <col style="width: 111px">
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
      <td>self</td>
      <td>Input</td>
      <td>Represents the first matrix in matrix multiplication, self in the formula.</td>
      <td><ul><li>The data type of self needs to satisfy the data type inference rules with mat2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>If self is not transposed, the dimension representation is (b, m, k).</li>
      <li>If self is transposed, the dimension representation is (b, k, m).</li>
      <li>The first dimension b of self needs to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the first dimension b of mat2.</li>
      </ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat2</td>
      <td>Input</td>
      <td>Represents the second matrix in matrix multiplication, mat2 in the formula.</td>
      <td><ul><li>The data type needs to satisfy the data type deduction rule with self (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The Reduce dimension of mat2 needs to be the same size as the Reduce dimension of self.</li>
      <li>When matrix B is not transposed, the dimensions of the Nz format are represented as (b, n1, k1, k0, n0), where k0 = 16 and n0 = 16. The k in the self shape and the k1 in the mat2 shape need to satisfy the following relationship: ceil(k, k0) = k1. The n1 in the mat2 shape and the n in out need to satisfy the following relationship: ceil(n, n0) = n1.</li>
      <li>When matrix B is transposed, the dimensions of the Nz format are represented as (b, k1, n1, n0, k0), where n0 = 16 and k0 = 16. The k in the self shape and the k1 in the mat2 shape need to satisfy the following relationship: ceil(k, k0) = k1. The n1 in the mat2 shape and the n in out need to satisfy the following relationship: ceil(n, n0) = n1.</li>
      <li>The first dimension b of mat2 needs to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the first dimension b of self.</li>
      </ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>Nz</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the output matrix of the matrix multiplication, which is out in the formula.</td>
      <td><ul><li>The data type needs to satisfy the deduction rule with the data type of self and mat2 after deduction (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>The dimensions are represented as (b, m, n), where m is consistent with m of self, n satisfies the relationship ceil(n / n0) = n1 with n1 and n0 of mat2, and b needs to be consistent with the b obtained by broadcasting b of self and b of mat2.</li>
      </ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter processes the deduced data type by default. Supported enumeration values:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports converting the input data to a lower precision for computation. Not supported.</li>
        <li>2: USE_FP16, supports converting the input to FLOAT16 for computation. Not supported when the input data type is BFLOAT16.</li>
        <li>3: USE_HF32, supports converting the input to the data type HFLOAT32 for computation. Not supported.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output parameter</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
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
  <table style="undefined;table-layout: fixed; width: 809px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 650px">
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
      <td>The input self, mat2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and data formats of self and mat2 are not supported.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed for self and mat2.</td>
    </tr>
    <tr>
      <td>The deduced data type cannot be converted to the data type of the specified output out.</td>
    </tr>
  </tbody>
  </table>

## aclnnBatchMatMulWeightNz

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 250px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnBatchMatMulWeightNzGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>The op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBatchMatMulWeightNz defaults to a deterministic implementation.

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
  - If self and mat2 are both of data type FLOAT16 or both of data type BFLOAT16, and out is of data type FLOAT32, the multiply-add results are accumulated in FLOAT32, and the final result is written to out in FLOAT32.

<!-- end id7 -->
- Data type deduction between BFLOAT16 and FLOAT16 inputs is not supported.
- self only supports 3 dimensions, and mat2 only supports the Ascend private format. Before calling this API, you must complete the conversion of mat2 from ND to the Ascend private format.
- The case where one of the last two axes of mat2 is 1, i.e., k=1 or n=1, is not supported.

## Example

<!-- npu="A3,910b" id8 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
<!-- end id8 -->

The following is the sample code for the use case where the data types of self and mat2 are float16 and mat2 is in Nz format. It is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_batch_matmul.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_cast.h"

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

  // Convert the uint16_t representation of FP16 to the float representation.
  float Fp16ToFloat(uint16_t h) {
    int s = (h >> 15) & 0x1;              // sign
    int e = (h >> 10) & 0x1F;             // exponent
    int f =  h        & 0x3FF;            // fraction
    if (e == 0) {
      // Zero or Denormal
      if (f == 0) {
        return s ? -0.0f : 0.0f;
      }
      // Denormals
      float sig = f / 1024.0f;
      float result = sig * pow(2, -24);
      return s ? -result : result;
    } else if (e == 31) {
        // Infinity or NaN
        return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
    }
    // Normalized
    float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
    return s ? -result : result;
  }

  int Init(int32_t deviceId, aclrtStream* stream) {
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
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the memory on the device side.
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

  template <typename T>
  int CreateAclTensorWeight(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));

    const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSize(mat2Size, &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSize failed. ERROR: %d\n", ret); return ret);
    size *= sizeof(T);

    // Call aclrtMalloc to apply for memory on the device side.
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the memory on the device side.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));

    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. (boilerplate) Initialize device/stream. See the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> selfShape = {2, 16, 32};
    std::vector<int64_t> mat2Shape = {2, 32, 16};
    std::vector<int64_t> outShape = {2, 16, 16};
    void* selfDeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    std::vector<uint16_t> selfHostData(1024, 0x3C00); // In float16_t, 0x3C00 represents 1 in int_16.
    std::vector<uint16_t> mat2HostData(1024, 0x3C00); // In float16_t, 0x3C00 represents 1 in int_16.
    std::vector<uint16_t> outHostData(512, 0);
    // Create the self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the other aclTensor
    ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    int8_t cubeMathType = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call TransWeight.
    ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnTransMatmulWeight.
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API aclnnBatchMatMulWeightNz.
    uint64_t workspaceSizeMm = 0;
    ret = aclnnBatchMatMulWeightNzGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddrMm = nullptr;
    if (workspaceSizeMm > 0) {
      ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnBatchMatMulWeightNz.
    ret = aclnnBatchMatMulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code according to the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // The C language cannot print fp16 data directly. Read the data as uint16 and manually convert the binary to the float representation of fp16.
    for (int64_t i = 0; i < size; i++) {
      float fp16Float = Fp16ToFloat(resultData[i]);
      LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
    }

    // 6. Release aclTensor and aclScalar. Modify the code according to the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(mat2);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify based on the specific API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(mat2DeviceAddr);
    aclrtFree(outDeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    if (workspaceSizeMm > 0) {
      aclrtFree(workspaceAddrMm);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```

<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT products</term>:
  The following sample code is for the use case where the data types of self and mat2 are float16 and mat2 is in NZ format, for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_batch_matmul.h"
  #include "aclnnop/aclnn_npu_format_cast.h"
  #include "aclnnop/aclnn_cast.h"

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

  // Convert the uint16_t representation of FP16 to the float representation.
  float Fp16ToFloat(uint16_t h) {
    int s = (h >> 15) & 0x1;              // sign
    int e = (h >> 10) & 0x1F;             // exponent
    int f =  h        & 0x3FF;            // fraction
    if (e == 0) {
      // Zero or Denormal
      if (f == 0) {
        return s ? -0.0f : 0.0f;
      }
      // Denormals
      float sig = f / 1024.0f;
      float result = sig * pow(2, -24);
      return s ? -result : result;
    } else if (e == 31) {
        // Infinity or NaN
        return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
    }
    // Normalized
    float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
    return s ? -result : result;
  }

  int Init(int32_t deviceId, aclrtStream* stream) {
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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape,
      uint64_t* storageShapeSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
      // Call aclrtMalloc to apply for device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data from the host to the device memory
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int main() {
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API documentation
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Customize the construction based on the API
    std::vector<int64_t> selfShape = {2, 16, 32};
    std::vector<int64_t> mat2Shape = {2, 32, 16};
    std::vector<int64_t> outShape = {2, 16, 16};

    std::vector<uint16_t> selfHostData(1024, 0x3C00); // For float16_t, 0x3C00 represents 1 in int_16
    std::vector<uint16_t> mat2HostData(1024, 0x3C00); // For float16_t, 0x3C00 represents 1 in int_16
    std::vector<uint16_t> outHostData(512, 0);

    aclTensor* self = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mat2NZ = nullptr;

    void* selfDeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* dstDeviceAddr = nullptr;

    // Create the self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the other aclTensor
    ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Convert the weight tensor from ND to NZ, and call the npu_format_cast API.
    aclDataType additionalDtype = aclDataType::ACL_FLOAT16;
    aclDataType srcDtype = aclDataType::ACL_FLOAT16;
    int64_t* dstShape = nullptr;
    uint64_t dstShapeSize = 0;
    int actualFormat;
    int8_t cubeMathType = 0;
    aclOpExecutor* executor = nullptr;

    uint64_t workspaceSize = 0;
    uint64_t workspaceSizeMm = 0;

    void* workspaceAddr = nullptr;
    void* workspaceAddrMm = nullptr;
    // Calculate the shape and format of the target tensor.
    ret = aclnnNpuFormatCastCalculateSizeAndFormat(mat2, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

    ret = CreateAclTensorWithFormat(
        mat2HostData, mat2Shape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &mat2NZ,
        static_cast<aclFormat>(actualFormat));
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCastGetWorkspaceSize(mat2, mat2NZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

    // 4. Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API aclnnBatchMatMulWeightNz.
    ret = aclnnBatchMatMulWeightNzGetWorkspaceSize(self, mat2NZ, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSizeMm > 0) {
        ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnBatchMatMulWeightNz.
    ret = aclnnBatchMatMulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 5. (boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. Obtain the output value, copy the result on the device memory to the host, and modify the code as required by the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // fp16 data cannot be printed directly in C. Read it as uint16 and manually convert the binary to the fp16 value represented by float.
    for (int64_t i = 0; i < size; i++) {
        float fp16Float = Fp16ToFloat(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
    }

    // 7. Destroy aclTensor and aclScalar, and modify the code as required by the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(mat2);
    aclDestroyTensor(out);

    // 8. Release device resources, and modify the code as required by the specific API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(mat2DeviceAddr);
    aclrtFree(outDeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    if (workspaceSizeMm > 0) {
      aclrtFree(workspaceAddrMm);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```

<!-- end id9 -->
