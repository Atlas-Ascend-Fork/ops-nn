# aclnnQuantMatmulReduceSumWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:53:24.050Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: Performs quantized grouped matrix computation, and then sums the matrix computation results of all groups and outputs the result.

- Formula:

$$
out = \sum_{i=0}^{batch}(x1_i @ x2_i) * x1Scale * x2Scale
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the computation flow of the operator, and then call "aclnnQuantMatmulReduceSumWeightNz" to execute the computation.

```Cpp
aclnnStatus aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(
    const aclTensor   *x1,
    const aclTensor   *x2,
    const aclTensor   *x1Scale,
    const aclTensor   *x2Scale,
    const aclTensor   *yScale,
    const aclTensor   *x1Offset,
    const aclTensor   *x2Offset,
    const aclTensor   *yOffset,
    const aclTensor   *bias,
    bool               transposeX1,
    bool               transposeX2,
    int64_t            groupSize,
    const aclIntArray *dims,
    bool               keepDims,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnQuantMatmulReduceSumWeightNz(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize

- Mandatory

  <table style="undefined; table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 165px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>x1 in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>(batch, m, k)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>x2 in the formula.</td>
      <td>        <ul><li>Empty tensors are not supported.</li>
        <li>Dimensions: (batch, n1, k1, k0, n0), where k0 = 16 and n0 = 32. k in the x1 shape and k1 in the x2 shape must satisfy: ceil(k / 16) = k1. n1 in the x2 shape and n in out must satisfy: ceil(n / n0) = n1.</li>
        <li>The aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight APIs can be used to convert the input format from ND to the AI processor-affinitized data layout format. The shape of the original ND format is (batch, k, n).</li></ul>
      </td>
      <td>INT8</td>
      <td>NZ</td>
      <td>5D</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x1Scale (aclTensor*)</td>
      <td>Input</td>
      <td>x1Scale in the formula.</td>
      <td>        <ul><li>Empty tensors are not supported.</li>
        <li>During actual computation, x1Scale is broadcast to (batch, m, n).</li></ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(batch, m)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2Scale (aclTensor*)</td>
      <td>Input</td>
      <td>x2Scale in the formula.</td>
      <td>        <ul><li>Empty tensors are not supported.</li>
        <li>During actual computation, x2Scale is broadcast to (batch, m, n).</li></ul>
      </td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(n,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yScale (aclTensor*)</td>
      <td>Input</td>
      <td>Reserved parameter, not supported in the current version.</td>
      <td>Must be nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x1Offset (aclTensor*)</td>
      <td>Input</td>
      <td>Reserved parameter, not supported in the current version.</td>
      <td>Pass in nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2Offset (aclTensor*)</td>
      <td>Input</td>
      <td>Reserved parameter, not supported in the current version.</td>
      <td>Pass in nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOffset (aclTensor*)</td>
      <td>Input</td>
      <td>Reserved parameter, not supported in the current version.</td>
      <td>Pass in nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias (aclTensor*)</td>
      <td>Input</td>
      <td>Reserved parameter, not supported in the current version.</td>
      <td>Pass in nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1 (bool)</td>
      <td>Input</td>
      <td>Whether the input shape of x1 includes transpose.</td>
      <td>Only false is supported in the current version, indicating that the meaning of the input shape of x1 remains unchanged.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2 (bool)</td>
      <td>Input</td>
      <td>Whether the input shape of x2 contains transpose.</td>
      <td>Only false is supported in the current version, indicating that the meaning of the input shape of x2 remains unchanged.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupSize (int64_t)</td>
      <td>Input</td>
      <td>Reserved parameter, not supported in the current version.</td>
      <td>Pass in 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dims (aclIntArray *)</td>
      <td>Input</td>
      <td>Specifies the reduce dimension.</td>
      <td>The current version supports only [0], indicating that ReduceSum is performed along dimension 0 (the batch dimension).</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepDims (bool)</td>
      <td>Input</td>
      <td>Indicates whether to keep the dimensions of the input tensor in the output tensor.</td>
      <td>The current version supports only false.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>The out in the formula.</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(m, n)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
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

- Return Value

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
        <td>The input x1, x2, x1Scale, x2Scale, or out is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data types and data formats of x1, x2, x1Scale, x2Scale, or out are not in the supported range.</td>
    </tr>
    <tr>
        <td>The shape of x1, x2, x1Scale, x2Scale, or out does not meet the validation requirements.</td>
    </tr>
    <tr>
        <td>x1, x2, x1Scale, x2Scale, or out is an empty tensor.</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantMatmulReduceSumWeightNz

- Mandatory

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
        <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the computation flow of the operator.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

- Return Value

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnQuantMatmulReduceSumWeightNz defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
  #include <cstdint>
  #include <cstring>
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_reduce_sum.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  float Bfloat16ToFloat(uint16_t value)
  {
      // BF16 is the upper 16 bits of FP32. The corresponding FP32 bit representation can be obtained by padding the lower 16 bits with 0.
      const uint32_t bits = static_cast<uint32_t>(value) << 16U;
      float result = 0.0F;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // Boilerplate code for resource initialization.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate device memory.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output as required based on the API.
      int64_t b = 8;
      int64_t m = 2048;
      int64_t k = 1024;
      int64_t n = 7168;
      // Create the x1 aclTensor.
      std::vector<int64_t> x1Shape = {b, m, k};
      void *x1DeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      std::vector<int8_t> x1HostData(b * m * k, 1);
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor in the AI processor-affine data layout format.
      std::vector<int64_t> x2Shape = {b, k, n};
      void *x2DeviceAddr = nullptr;
      aclTensor *x2 = nullptr;
      std::vector<int8_t> x2HostData(b * k * n, 1);
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x1Scale aclTensor.
      std::vector<int64_t> x1ScaleShape = {b, m};
      void *x1ScaleDeviceAddr = nullptr;
      std::vector<float> x1ScaleHostData(b * m, 1);
      aclTensor *x1Scale = nullptr;
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Scale aclTensor.
      std::vector<int64_t> x2ScaleShape = {n};
      void *x2ScaleDeviceAddr = nullptr;
      aclTensor *x2Scale = nullptr;
      constexpr uint16_t bf16One = 0x3F80U;
      std::vector<uint16_t> x2ScaleHostData(n, bf16One);  // Bit representation of the BF16 value 1.0
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_BF16, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      std::vector<int64_t> outShape = {m, n};
      void *outDeviceAddr = nullptr;
      aclTensor *out = nullptr;
      std::vector<uint16_t> outHostData(m * n, 0);  // Bit representation of the BF16 value 0.0
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      // Create the dims aclIntArray.
      std::vector<int64_t> dimsData = {0};
      aclIntArray *dims = nullptr;
      dims = aclCreateIntArray(dimsData.data(), dimsData.size());
      CHECK_RET(dims != nullptr, return ret);

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulReduceSumWeightNz.
      ret = aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(
        x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, nullptr, transposeX1, transposeX2, 0,
        dims, false, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API aclnnQuantMatmulReduceSumWeightNz.
      ret = aclnnQuantMatmulReduceSumWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulReduceSumWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, and copy the result on device memory to host memory. Modify the code based on the specific API definition.
      auto size = GetShapeSize(outShape);
      // C++ cannot directly print BF16. Read its bit representation as uint16_t first, and then convert BF16 to FP32.
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < 5; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, Bfloat16ToFloat(resultData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Initialize device/stream. For details, see the acl API documentation.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }

```
