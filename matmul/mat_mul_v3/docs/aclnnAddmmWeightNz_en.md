# aclnnAddmmWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:14:34.798Z -->

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

- Description: Computes α multiplied by the product of mat1 and mat2, then adds the product of β and self. Compared with the original addmm API, the new API supports the NZ format for mat2.
- Formula:

  $$
  out = β  self + α  (mat1 @ mat2)
  $$

- Example:
  * For the aclnnAddmmWeightNz API, the shape of self is [n,], the shape of mat1 is [m, k], the shape of mat2 is [k, n], the shape of the Matrix Multiplication result of mat1 and mat2 is [m, n], and the shape of self can be broadcast to [m, n].
  * For the aclnnAddmmWeightNz API, the shape of self is [1, n], the shape of mat1 is [m, k], the shape of mat2 is [k, n], the shape of the matrix multiplication result of mat1 and mat2 is [m, n], and the shape of self can be broadcast to [m, n].
  * For the aclnnAddmmWeightNz API, self is of shape [m, 1], mat1 is of shape [m, k], mat2 is of shape [k, n], the result of the matrix multiplication of mat1 and mat2 is of shape [m, n], and self can be broadcast to [m, n].
  * For the aclnnAddmmWeightNz API, self is of shape [1, 1], mat1 is of shape [m, k], mat2 is of shape [k, n], the result of the matrix multiplication of mat1 and mat2 is of shape [m, n], and self can be broadcast to [m, n].
  * For the aclnnAddmmWeightNz API, the shape of self is [m, n], the shape of mat1 is [m, k], the shape of mat2 is [k, n], and the shape of the matrix multiplication result of mat1 and mat2 is [m, n].

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnAddmmWeightNzGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation process. Then call the "aclnnAddmmWeightNz" API to perform the computation.

```cpp
aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *mat1,
  const aclTensor *mat2,
  const aclScalar *beta,
  const aclScalar *alpha,
  aclTensor       *out,
  int8_t          cubeMathType,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAddmmWeightNz(
  void           *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream    stream)
```

## aclnnAddmmWeightNzGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1563px"><colgroup>
  <col style="width: 153px">
  <col style="width: 123px">
  <col style="width: 232px">
  <col style="width: 437px">
  <col style="width: 203px">
  <col style="width: 120px">
  <col style="width: 149px">
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
      <td>Represents the bias matrix, self in the formula.</td>
      <td><ul><li>The data type needs to satisfy the data type deduction rule with mat1@mat2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>Need to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with mat1@mat2.</li> <li>self supports shape (n), (1,n), (m,1), [1,1], or (m,n).</li> </ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat1</td>
      <td>Input</td>
      <td>Represents the first matrix of the Matrix Multiplication, mat1 in the formula.</td>
      <td><ul><li>The data type needs to satisfy the data type inference rule with mat2 (See Also <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>Needs to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with self and mat2.</li> <li> Dimension representation when mat1 is not transposed: (m, k).</li><li>Dimension representation when mat1 is transposed: (k, m).</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat2</td>
      <td>Input</td>
      <td>Indicates the second matrix of Matrix Multiplication, that is, mat2 in the formula.</td>
      <td><ul><li>The data type needs to satisfy the data type inference rule with mat1 (See Also <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The Reduce dimension of mat2 needs to be of the same size as the Reduce dimension of mat1.</li><li>Needs to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with self and mat1.</li>
      <li>When the mat2 matrix is not transposed, the dimension representation of the NZ format is [n1, k1, k0, n0], where k0 = 16 and n0 = 16. k in the mat1 shape and k1 in the mat2 shape need to satisfy the following relationship: ceil(k, k0) = k1, and n1 in the mat2 shape and n in out need to satisfy the following relationship: ceil(n, n0) = n1.</li>
     <li>When the mat2 matrix is transposed, the dimension representation of the NZ format is [k1, n1, n0, k0], where n0 = 16 and k0 = 16. k in the mat1 shape and k1 in the mat2 shape need to satisfy the following relationship: ceil(k, k0) = k1, and n1 in the mat2 shape and n in out need to satisfy the following relationship: ceil(n, n0) = n1.</li> </ul>
      </td>
      <td>BFLOAT16, FLOAT16</td>
      <td>NZ</td>
      <td>4</td>
      <td>-</td>
    </tr>
    <tr>
    <tr>
      <td>beta(β)</td>
      <td>Input</td>
      <td>Represents β in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha(α)</td>
      <td>Input</td>
      <td>Represents α in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the output matrix of Matrix Multiplication, that is, out in the formula.</td>
      <td>The data type needs to be the same as the data type deduced from self and mat2 (See Also <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter processes the data type after deduction by default. The supported enumerated values are as follows:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports downgrading the input data to a lower precision for computation.</li>
        <li>2: USE_FP16, supports downgrading the input to FLOAT16 for computation.</li>
        <li>3: USE_HF32, supports downgrading the input to the data type HFLOAT32 for computation.</li>
        <li>4: USE_FP32_ADD, supports computing in high-precision mode.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output parameter</td>
      <td>Returns the workspace size that needs to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>output</td>
      <td>Returns the op executor, including the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
      - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
      - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
      - cubeMathType=3, when the input data type is FLOAT32, it is converted to HFLOAT32 for computation; when the input is of other data types, this option is not supported;
      - cubeMathType=4, when self, mat1, and mat2 are all FLOAT16 or all BFLOAT16, the Matrix Multiplication and bias addition process uses the FLOAT32 Intermediate Result for computation. Broadcasting self relative to the Matrix Multiplication result is supported, and the output data type is specified by out.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
      - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
      - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
      - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
      - cubeMathType=4, when the input data type is FLOAT16/BFLOAT16, the addmm process performs computation with higher precision. In this case, broadcasting the input self with the matmul Computation Result matrix is currently not supported.
  <!-- end id8 -->
  <!-- npu="910,310p" id9 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 is not supported.

  <!-- end id9 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 283px">
      <col style="width: 120px">
      <col style="width: 747px">
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
          <td>The input self, mat1, mat2, or out is a null pointer.</td>
        </tr>
        <tr>
          <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
          <td rowspan="3">161002</td>
          <td>The data type and data format of self and mat2 are not supported.</td>
        </tr>
        <tr>
          <td>mat1 and mat2 do not satisfy the multiplication condition.</td>
        </tr>
        <tr>
          <td>The shape of out is inconsistent with that of mat1@mat2.</td>
        </tr>
      </tbody>
      </table>

## aclnnAddmmWeightNz

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnAddmmWeightNzGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation description: aclnnAddmmWeightNz defaults to a deterministic implementation.

<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>:
  - When mat1 and mat2 are both of the data type FLOAT16 or both of BFLOAT16, the computation precision is as follows:
    - self is a 1D tensor:
      - When the data type of out is FLOAT32, MatMul accumulates in FLOAT32, and beta×self, alpha×(mat1@mat2), and the intermediate result of adding the two parts are all computed in FLOAT32. The final result is written to out in FLOAT32.
      - When self is of shape [n] and alpha=1, beta=1, self is directly used as the bias input of MatMul, and the addition is performed inside MatMul.
      - When the data type of out is FLOAT16 or BFLOAT16, setting cubeMathType=4 does not make the intermediate computation result use FLOAT32.
    - self is a 2D Tensor:
      - When the data type of out is FLOAT32, MatMul accumulates in FLOAT32, and beta×self, alpha×(mat1@mat2), and the intermediate result of adding the two parts are all computed in FLOAT32. The final result is written to out in FLOAT32.
      - When the data type of out is FLOAT16 or BFLOAT16 and cubeMathType=4, the preceding intermediate results are computed in FLOAT32, and the final result is converted to the data type specified by out.
      - When the data type of out is FLOAT16 or BFLOAT16 and cubeMathType is not 4, the intermediate results are not converted to FLOAT32.
      - When self is of shape [1, n] and alpha=1, beta=1, self is directly used as the bias input of MatMul, and the addition is performed inside MatMul.

<!-- end id10 -->
- Computation consistency description

  <!-- npu="910,310p" id11 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>:
    - When the strong consistency computation feature is enabled, the computation result is deterministic, and multiple executions produce the same output. In addition, the computation result is independent of the data location.
    - aclnnAddmmWeightNz defaults to a non-deterministic implementation and supports enabling deterministic computation via aclrtCtxSetSysParamOpt.
    - For example, during Matrix Multiplication, the accumulation order of different basic blocks may vary, which can cause slight differences in the computation results of the same data across different rows. However, when deterministic computation is enabled, the computation results will be identical as long as the inputs are the same, even across different rows.

  <!-- end id11 -->

- Data type deduction is not supported when one of the two inputs, mat1 and mat2, is BFLOAT16 and the other is FLOAT or FLOAT16.
- The case where one of the last two axes of mat2 is 1, i.e., k=1 or n=1, is not supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_addmm.h"
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

  // Convert the uint16_t representation of FP16 to a float representation.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
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

    // Call aclrtMalloc to allocate device-side memory.
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Customize the construction based on the API.
    std::vector<int64_t> selfShape = {16};
    std::vector<int64_t> mat1Shape = {16, 32};
    std::vector<int64_t> mat2Shape = {32, 16};
    std::vector<int64_t> outShape = {16, 16};
    void* selfDeviceAddr = nullptr;
    void* mat1DeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mat1 = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    aclScalar* alpha = nullptr;
    aclScalar* beta = nullptr;

    std::vector<uint16_t> selfHostData(16, 0x3C00); // float16_t uses 0x3C00 to represent 1 of int_16.
    std::vector<uint16_t> mat1HostData(512, 0x3C00); // float16_t uses 0x3C00 to represent 1 of int_16.
    std::vector<uint16_t> mat2HostData(512, 0x3C00); // float16_t uses 0x3C00 to represent 1 of int_16.
    std::vector<uint16_t> outHostData(256, 0);
    float alphaValue = 1.0f;
    float betaValue = 1.0f;

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mat1 aclTensor.
    ret = CreateAclTensor(mat1HostData, mat1Shape, &mat1DeviceAddr, aclDataType::ACL_FLOAT16, &mat1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mat2 aclTensor.
    ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create alpha aclScalar.
    alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return ret);
    // Create beta aclScalar.
    beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
    CHECK_RET(beta != nullptr, return ret);


    // 3. Call CANN operator library APIs. Replace with the specific API name.
    int8_t cubeMathType = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call TransWeight.
    ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnTransMatmulWeight.
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnAddmmWeightNz.
    uint64_t workspaceSizeMm = 0;
    ret = aclnnAddmmWeightNzGetWorkspaceSize(self, mat1, mat2, beta, alpha, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddrMm = nullptr;
    if (workspaceSizeMm > 0) {
      ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddmmWeightNz.
    ret = aclnnAddmmWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to end.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, copy the result on the device memory to the host, and modify it based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // fp16 data cannot be directly printed in C. Read it as uint16 and manually convert the binary to fp16 represented as float.
    for (int64_t i = 0; i < size; i++) {
      float fp16Float = Fp16ToFloat(resultData[i]);
      LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
    }

    // 6. Release the aclTensor and aclScalar, and modify it based on the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(mat1);
    aclDestroyTensor(mat2);
    aclDestroyScalar(beta);
    aclDestroyScalar(alpha);
    aclDestroyTensor(out);

    // 7. Release the device resources, and modify it based on the specific API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(mat1DeviceAddr);
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
