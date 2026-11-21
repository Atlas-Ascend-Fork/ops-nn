# aclnnSituGluGroupGrad

<!-- aclnnSituGluGroupGrad: aclnn interface of the SituGluGroupGrad operator, including the optional inputs weight/group_index -->

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

- Interface function: the backward gradient computation of the SiTU Gated Linear Unit activation. Given the upstream gradient gradY and the forward input x, it computes the gradient gradX of x and the gradient gradWeight of weight. When the optional input weight is passed, gradX is weighted per row and gradWeight is computed; when weight is absent, gradWeight is an empty tensor (shape [0]). When the optional input groupIndex is passed, only the first realBs = min(groupRows, number of rows) rows are computed, where groupRows is groupIndex[-1] in cumsum mode (groupListType 0, the default) or sum(groupIndex) in count mode (groupListType 1). Supported only on Ascend 950PR&950DT products.

- Calculation formula:

  For a given input tensor x (dimensions [..., cut], where cut is even, let h = cut // 2) and upstream gradient gradY (dimensions [..., h]), aclnnSituGluGroupGrad performs the following computations:

  1. Splits x into gate and up based on activateLeft (each of shape [..., h]).

  2. Computes the forward intermediates:

     $$
     t = \tanh\left(\frac{gate}{\beta}\right), \quad s = \sigma(gate), \quad situ\_a = \beta \cdot t \cdot s
     $$

     When linearBeta > 0:

     $$
     up' = linearBeta \cdot \tanh\left(\frac{up}{linearBeta}\right)
     $$

     Otherwise up' = up.

  3. Product rule:

     $$
     grad\_situ\_a = gradY \cdot up', \quad grad\_up' = gradY \cdot situ\_a
     $$

  4. Gate gradient (w is the scalar of the corresponding row of the optional input weight; w = 1 when weight is absent):

     $$
     grad\_gate = grad\_situ\_a \cdot s \cdot \left[ (1 - t^2) + \beta \cdot t \cdot (1 - s) \right] \cdot w
     $$

  5. Up gradient:

     - When linearBeta > 0:

     $$
     grad\_up = grad\_up' \cdot \left(1 - \tanh^2\left(\frac{up}{linearBeta}\right)\right) \cdot w
     $$

     - When linearBeta ≤ 0:

     $$
     grad\_up = grad\_up' \cdot w
     $$

  6. Concatenates grad_gate and grad_up into grad_x based on activateLeft (shape identical to x).

  7. Weight gradient (output grad_weight; an empty tensor when weight is absent), summed over h for each row n (note: not multiplied by w):

     $$
     grad\_w_n = \sum_{h} gradY_{n,h} \cdot situ\_a_{n,h} \cdot up'_{n,h}
     $$

  8. Row masking (optional input groupIndex): realBs = min(groupRows, number of rows). Only the first realBs rows are computed and written back; rows whose index ≥ realBs are neither read nor written (the corresponding rows of the output buffer keep the caller's original values). groupRows is selected by groupListType: groupIndex[-1] in cumsum mode (0, the default) and sum(groupIndex) in count mode (1).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSituGluGroupGradGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnSituGluGroupGrad" API to perform the computation.

```cpp
aclnnStatus aclnnSituGluGroupGradGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *weightOptional,
    const aclTensor *groupIndexOptional,
    int64_t dim,
    double beta,
    double linearBeta,
    bool activateLeft,
    int64_t groupListType,
    const aclTensor *gradXOut,
    const aclTensor *gradWeightOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnSituGluGroupGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnSituGluGroupGradGetWorkspaceSize

- **Parameter description**

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
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage notes</th>
      <th>Data type</th>
      <th>Data format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradY (aclTensor*)</td>
      <td>Input</td>
      <td>Input gradY in the formula, the upstream gradient.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape on the dimension corresponding to dim is half of x; the other dimensions are the same as x.</li><li>The data type is the same as x.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input x in the formula, the forward input.</td>
      <td><ul><li>Empty tensors are supported (the output is the corresponding empty tensor).</li><li>The shape must be even on the dimension corresponding to dim.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input weight in the formula, the per-row routing weight of the forward.</td>
      <td><ul><li>Supported only on Ascend 950PR&950DT products; may be nullptr.</li><li>The number of elements must equal the product of the dimensions of x before the dim axis.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input groupIndex in the formula, the number of tokens of each group, whose meaning is selected by groupListType.</td>
      <td><ul><li>Supported only on Ascend 950PR&950DT products; may be nullptr.</li><li>It must be a 1-D non-empty tensor with non-negative elements.</li><li>Only the first min(groupRows, number of rows) rows are computed and written back, where groupRows is selected by groupListType.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (int64_t)</td>
      <td>Input</td>
      <td>Input dim in the formula, the index of the dimension along which x is folded and split.</td>
      <td>The value range is [-x.dim(), x.dim()-1].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta (double)</td>
      <td>Input</td>
      <td>Input beta in the formula, the scaling coefficient of the SiTU gate part.</td>
      <td>It is recommended to be a non-zero value.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linearBeta (double)</td>
      <td>Input</td>
      <td>Input linearBeta in the formula, the scaling coefficient of the linear tanh on the up path. When it is less than or equal to 0, the up path is not transformed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Input activateLeft in the formula, whether gate is the front half when x is split.</td>
      <td>true means gate is the front half and up is the back half; false means gate is the back half and up is the front half.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType (int64_t)</td>
      <td>Input</td>
      <td>Input groupListType in the formula, how groupIndex expresses the group sizes.</td>
      <td><ul><li>The value can only be 0 or 1; any other value returns a parameter validation failure.</li><li>0 is cumsum mode: groupIndex holds the cumulative row counts, so the total is groupIndex[-1].</li><li>1 is count mode: groupIndex holds the per-group row counts, so the total is sum(groupIndex).</li><li>The default value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output grad_x in the formula, the input gradient.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is the same as x.</li><li>The data type is the same as x.</li><li>When groupIndex is passed, rows whose index ≥ realBs are not written (the buffer keeps its original values).</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradWeightOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output grad_weight in the formula, the gradient of weight.</td>
      <td><ul><li>Supported only on Ascend 950PR&950DT products.</li><li>When weight is passed, the shape is the same as weight and the number of elements equals the product of the dimensions of x before the dim axis; when weight is absent, it must be an empty tensor (shape [0]).</li><li>When groupIndex is passed, rows whose index ≥ realBs are not written (the buffer keeps its original values).</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that must be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return values**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API completes input parameter validation. An error is returned in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>One of gradY, x, gradXOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type of gradY, x or gradXOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape dimension of gradY, x or gradXOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>dim is not within the specified value range, or x cannot be divided by 2 on the dimension corresponding to dim.</td>
    </tr>
    <tr>
      <td>The number of dimensions of gradY is inconsistent with x, or the size of gradY on the dimension corresponding to dim is not equal to half of x.</td>
    </tr>
    <tr>
      <td>The data type of gradY does not match x.</td>
    </tr>
    <tr>
      <td>The data type of weightOptional is not FLOAT, or its number of elements does not equal the product of the dimensions of x before the dim axis.</td>
    </tr>
    <tr>
      <td>The data type of groupIndexOptional is not INT64, it is not 1-D, or it is an empty tensor.</td>
    </tr>
    <tr>
      <td>The value of groupListType is neither 0 nor 1.</td>
    </tr>
    <tr>
      <td>When weightOptional is passed, the number of elements of gradWeightOut does not equal the product of the dimensions of x before the dim axis; or when weightOptional is absent, gradWeightOut is not empty.</td>
    </tr>
  </tbody></table>

## aclnnSituGluGroupGrad

- **Parameter description**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>The address of the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The size of the workspace allocated on the device, obtained from the first-phase API aclnnSituGluGroupGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>The op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>The stream on which the task is executed.</td>
    </tr>
  </tbody></table>

- **Return values**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnSituGluGroupGrad defaults to a deterministic implementation.
- Supported only on Ascend 950PR&950DT products (weightOptional and groupIndexOptional are optional inputs, gradWeightOut is a required output).
- When weightOptional is absent, gradWeightOut must be an empty tensor (shape [0]); no weight gradient is produced in this case.
- When groupIndexOptional is passed, rows whose index ≥ realBs in gradXOut and gradWeightOut are not written; the caller must clear the buffer explicitly if strict zeros are required.
- groupListType must take a value from [0, 1].
- The values of groupIndexOptional must meet the following conditions, otherwise the output is not guaranteed to be as expected:
  - When groupListType is 0, groupIndexOptional must be a non-negative, monotonically non-decreasing sequence, representing the cumsum result (cumulative sum) of the size of each group after grouping. The last value must be less than or equal to the number of rows.
  - When groupListType is 1, groupIndexOptional must be a non-negative sequence, representing the size of each group after grouping. The sum of the values must be less than or equal to the number of rows.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_situ_glu_group_grad.h"

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
      // Fixed code: ACL initialization
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
      // grad_weight is an empty tensor (shape [0]) when weight is absent, so size is 0; allocate at least one block for portability
      auto allocSize = size > 0 ? size : static_cast<decltype(size)>(32);
      // Call aclrtMalloc to allocate device memory
      auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory
      if (size > 0) {
          ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      }
      // Compute the strides of a contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create the aclTensor
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (Fixed code) device/stream initialization, see the ACL API reference
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs
      std::vector<int64_t> xShape = {2, 32};
      std::vector<int64_t> gradYShape = {2, 16};
      std::vector<int64_t> gradXShape = {2, 32};
      std::vector<int64_t> gradWeightShape = {0}; // weight is absent, grad_weight is the required empty tensor
      void* gradYDeviceAddr = nullptr;
      void* xDeviceAddr = nullptr;
      void* gradXDeviceAddr = nullptr;
      void* gradWeightDeviceAddr = nullptr;
      aclTensor* gradY = nullptr;
      aclTensor* x = nullptr;
      aclTensor* gradX = nullptr;
      aclTensor* gradWeight = nullptr;
      std::vector<float> gradYHostData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
      std::vector<float> xHostData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
      std::vector<float> gradXHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      int64_t dim = -1;
      double beta = 1.0;
      double linearBeta = 0.0;
      bool activateLeft = true;
      int64_t groupListType = 0;
      // Create the gradY aclTensor
      ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the gradX aclTensor
      ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the gradWeight aclTensor (grad_weight is a required output; weight is absent here, so pass an empty tensor)
      std::vector<float> gradWeightHostData;
      ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT,
                            &gradWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call the first-phase API of aclnnSituGluGroupGrad (pass nullptr for weightOptional/groupIndexOptional to omit them;
      // gradWeightOut is a required output, so an empty tensor must be passed)
      ret = aclnnSituGluGroupGradGetWorkspaceSize(gradY, x, nullptr, nullptr, dim, beta, linearBeta, activateLeft,
                                              groupListType, gradX, gradWeight, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGroupGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory according to the workspaceSize computed by the first-phase API
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnSituGluGroupGrad
      ret = aclnnSituGluGroupGrad(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGroupGrad failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed code) synchronize and wait for the task to finish
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Get the output values: copy the result from device memory to host
      auto size = GetShapeSize(gradXShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), gradXDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }
      // 6. Destroy the aclTensors
      aclDestroyTensor(gradY);
      aclDestroyTensor(x);
      aclDestroyTensor(gradX);
      aclDestroyTensor(gradWeight);
      // 7. Release device resources
      aclrtFree(gradYDeviceAddr);
      aclrtFree(xDeviceAddr);
      aclrtFree(gradXDeviceAddr);
      aclrtFree(gradWeightDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
