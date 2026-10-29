# aclnnRenorm&aclnnInplaceRenorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:32:43.706Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Returns a tensor where each sub-tensor of the input tensor self along the dimension dim is normalized so that the p-norm of the sub-tensor is below the maxNorm value.

- Formula:

  $$
  output_i=\left\{
  \begin{aligned}
  input_i,\quad ||input_i||_p ≤ maxNorm \\
  \frac {input_i} {max(||input_i||_p, eps)} \cdot maxNorm,\quad ||input_i||_p>maxNorm
  \end{aligned}
  \right.
  $$

  Where:
  $i$ is a tensor slice along a dimension determined by dim:

  $$
  ||input_i||_p =\left\{
  \begin{aligned}
  \sqrt[p]{reduce\_sum(|input_i|^p, dim)},\quad (p > 0) \& (p ≠ inf) \\
  max(|input_i|),\quad p = inf
  \end{aligned}
  \right.
  $$

- Example:

  ```text
  x = tensor([[1.,1.,1.],
              [2.,2.,2.],
              [3.,3.,3.]])
  Here, p=1, dim=0, and maxNorm=5. These values are passed in the aclnn API call.
  Since dim=0, the determination and computation are performed in units of rows (dimension 0);
  - The norm of the first row sub-tensor is 1+1+1=3, which is less than 5. Therefore, the sub-tensor remains unchanged.
  - The norm of the second row sub-tensor is 2+2+2=6, which is greater than 5. Therefore, the computation is applied to the sub-tensor: (2/6)*5=1.6667.
  - The norm of the third row sub-tensor is 3+3+3=9, which is greater than 5. Therefore, the computation is applied to the sub-tensor: (3/9)*5=1.6667.
    tensor([[1.0000,1.0000,1.0000],
           [1.6667,1.6667,1.6667],
           [1.6667,1.6667,1.6667]])
  If p=2, the norm computation of the first row sub-tensor changes to √1+1+1=1.73. Similarly, the second and third rows become:
  √2 * 2+2 * 2+2 * 2=3.46, √3 * 3+3 * 3+3 * 3=5.19
  ```

## Function Prototype

- aclnnRenorm and aclnnInplaceRenorm implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnRenorm: Create a new output tensor object to store the computation result.
  - aclnnInplaceRenorm: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnRenormGetWorkspaceSize" or "aclnnInplaceRenormGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnRenorm" or "aclnnInplaceRenorm" API to execute the computation.

  ```Cpp
  aclnnStatus aclnnRenormGetWorkspaceSize(
    const aclTensor* self,
    const aclScalar* p,
    int64_t          dim,
    const aclScalar* maxNorm,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnRenorm(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceRenormGetWorkspaceSize(
    aclTensor*       selfRef,
    const aclScalar* p,
    int64_t          dim,
    const aclScalar* maxNorm,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceRenorm(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
  ```

## aclnnRenormGetWorkspaceSize

- **Parameters**

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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input for renormalization computation, corresponding to `input` in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>p (aclScalar*)</td>
      <td>Input</td>
      <td>Norm, corresponding to `p` in the formula.</td>
      <td>Greater than or equal to 0</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>Specifies the dimension along which the norm is computed. Corresponds to `i` in the corresponding formula.</td>
      <td>Value range: [-number of dimensions of self, number of dimensions of self - 1]</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxNorm (aclScalar*)</td>
      <td>Input</td>
      <td>Represents the maximum allowed normalized value. Corresponds to `maxNorm` in the formula.</td>
      <td><ul><li>The value must be greater than or equal to 0.</li><li>If the `p` norm (determined by `p`) of the corresponding dimension is greater than `maxNorm` during the operation, the values along that dimension are normalized with respect to the `p` norm and then multiplied by `maxNorm`.</li><li>If the `p` norm (determined by `p`) of the corresponding dimension is less than `maxNorm` during the operation, the tensor along that dimension is output unchanged.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the final output, corresponding to `output` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Data type and shape must be consistent with input parameter `self`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for on the Device.</td>
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
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: The data type of the `self` and `out` parameters does not support BFLOAT16.
  <!-- end id7 -->

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
      <td>The passed-in self, p, maxNorm, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of self or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The dtypes of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The value of p does not meet the requirement.</td>
    </tr>
    <tr>
      <td>The value of dim is not in the range [-number of dimensions of self, number of dimensions of self - 1].</td>
    </tr>
    <tr>
      <td>maxNorm < 0.</td>
    </tr>
    <tr>
      <td>The dimension of input self is not within [2,8].</td>
    </tr>
  </tbody></table>

## aclnnRenorm

- **Parameters**

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
      <td>Pointer to the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device, obtained by the first-phase API aclnnRenormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceRenormGetWorkspaceSize

- **Parameters**

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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input and final output for renormalization calculation, corresponding to `input` and `output` in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>p (aclScalar*)</td>
      <td>Input</td>
      <td>Norm, corresponding to `p` in the formula.</td>
      <td>Must be greater than or equal to 0.</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>Indicates the dimension along which the norm is calculated. Corresponds to `i` in the formula.</td>
      <td>Value range: [-number of dimensions of selfRef, number of dimensions of selfRef - 1].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxNorm(aclScalar*)</td>
      <td>Input</td>
      <td>Represents the maximum allowed normalization value. It corresponds to `maxNorm` in the corresponding formula.</td>
      <td><ul><li>The value must be greater than or equal to 0.</li><li>If the `p`-norm (determined by the `p` value) of the corresponding dimension is greater than `maxNorm` during computation, the values of this dimension are normalized with respect to the `p`-norm and multiplied by `maxNorm`.</li><li>If the `p`-norm (determined by the `p` value) of the corresponding dimension is less than `maxNorm` during computation, the tensor of this dimension is output unchanged.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to allocate on the Device.</td>
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

  <!-- npu="910" id8 -->
  - <term>Atlas training products</term>: The data type of the parameter `selfRef` does not support BFLOAT16.
  <!-- end id8 -->

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
      <td>The passed-in selfRef, p, or maxNorm is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of selfRef is out of the supported range.</td>
    </tr>
    <tr>
      <td>p < 0.</td>
    </tr>
    <tr>
      <td>The value of dim is not within the range [-number of dims of selfRef, number of dims of selfRef - 1].</td>
    </tr>
    <tr>
      <td>maxNorm < 0.</td>
    </tr>
    <tr>
      <td>The dimension of input selfRef is not in the range [2,8].</td>
    </tr>
  </tbody></table>

## aclnnInplaceRenorm

- **Parameters**

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnInplaceRenormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the computation process of the operator.</td>
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
  - aclnnRenorm defaults to a deterministic implementation.
  - aclnnInplaceRenorm defaults to a deterministic implementation.
- If input parameter `p` is 0, see "Formula" in [aclnnEmbeddingRenorm](aclnnEmbeddingRenorm_en.md#function) for the formula.

## Example

The following example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

- **aclnnRenorm sample code:**

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_renorm.h"

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
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. See the acl API manual
      // Set deviceId based on the actual device
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output as required by the API
      std::vector<int64_t> selfShape = {3, 3};
      std::vector<int64_t> outShape = {3, 3};
      void* selfDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* self = nullptr;
      aclScalar* p = nullptr;
      aclScalar* maxNorm = nullptr;
      aclTensor* out = nullptr;
      std::vector<float> selfHostData = {1, 1, 1, 2, 2, 2, 3, 3, 3};
      std::vector<float> outHostData(9, 0);
      int64_t dim = -1;
      float pValue = 1.0f;
      float maxNormValue = 5.0f;
      // Create the self aclTensor
      ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the p aclScalar
      p = aclCreateScalar(&pValue, aclDataType::ACL_FLOAT);
      CHECK_RET(p != nullptr, return ret);
      // Create the maxNorm aclScalar.
      maxNorm = aclCreateScalar(&maxNormValue, aclDataType::ACL_FLOAT);
      CHECK_RET(maxNorm != nullptr, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call the first-phase API of aclnnRenorm.
      ret = aclnnRenormGetWorkspaceSize(self, p, dim, maxNorm, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRenormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnRenorm.
      ret = aclnnRenorm(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRenorm failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait synchronously for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result on the device memory to the host side. Modify it according to the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }

      // 6. Release aclTensor and aclScalar. Modify it according to the definition of the specific API.
      aclDestroyTensor(self);
      aclDestroyScalar(p);
      aclDestroyScalar(maxNorm);
      aclDestroyTensor(out);
      return 0;
  }
  ```

- **Sample code for aclnnInplaceRenorm:**

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_renorm.h"

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
      // Boilerplate, resource initialization.
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
      // Call aclrtMalloc to allocate device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (Boilerplate) Device/stream initialization. Refer to the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Custom construction is required based on the API.
      std::vector<int64_t> selfRefShape = {3, 3};
      void* selfRefDeviceAddr = nullptr;
      aclTensor* selfRef = nullptr;
      aclScalar* p = nullptr;
      aclScalar* maxNorm = nullptr;
      aclTensor* out = nullptr;
      std::vector<float> selfRefHostData = {1, 1, 1, 2, 2, 2, 3, 3, 3};
      int64_t dim = -1;
      float pValue = 1.0f;
      float maxNormValue = 5.0f;
      // Create the selfRef aclTensor.
      ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the p aclScalar.
      p = aclCreateScalar(&pValue, aclDataType::ACL_FLOAT);
      CHECK_RET(p != nullptr, return ret);
      // Create the maxNorm aclScalar.
      maxNorm = aclCreateScalar(&maxNormValue, aclDataType::ACL_FLOAT);
      CHECK_RET(maxNorm != nullptr, return ret);

      // 3. Call the CANN operator library API, and modify it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call the first-phase API of aclnnInplaceRenorm.
      ret = aclnnInplaceRenormGetWorkspaceSize(selfRef, p, dim, maxNorm, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRenormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory using the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnInplaceRenorm.
      ret = aclnnInplaceRenorm(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRenorm failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronously wait for the task to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, and copy the result from the device memory to the host memory. Modify it based on the specific API definition.
      auto size = GetShapeSize(selfRefShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }

      // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
      aclDestroyTensor(selfRef);
      aclDestroyScalar(p);
      aclDestroyScalar(maxNorm);

      // 7. Release device resources. Modify it based on the specific API definition.
      aclrtFree(selfRefDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
