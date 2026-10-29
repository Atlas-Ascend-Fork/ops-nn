# aclnnWeightQuantBatchMatmulNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:19:08.348Z -->

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

- **Description**: Performs a matrix multiplication computation where the input is in a pseudo-quantization scenario. This API only supports the right input matrix of the matrix multiplication in FRACTAL_NZ format.
- **Formula**:
  - Basic formula:

    $$
    y = x @ ANTIQUANT(weight) + bias
    $$

    Where $weight$ is the input in the pseudo-quantization scenario, and its antiquantization formula $ANTIQUANT(weight)$ is:

    $$
    ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
    $$

  - Quantization formula when the output needs to be quantized:

    $$
    \begin{aligned}
    y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
    &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
    \end{aligned}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnWeightQuantBatchMatmulNzGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the computation process of the operator, and then call the "aclnnWeightQuantBatchMatmulNz" API to perform the computation.

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *antiquantScale,
  const aclTensor *antiquantOffsetOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *biasOptional,
  int              antiquantGroupSize,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulNz(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnWeightQuantBatchMatmulNzGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1550px">
    <colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
    </colgroup>
    <thread>
      <tr>
        <th>Parameter</th>
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th style="white-space: nowrap">Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Left input matrix of the matrix multiplication, corresponding to input <code>x</code> in the formula.</td>
        <td>Transpose is not supported.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Right input matrix of the matrix multiplication, corresponding to input <code>weight</code> in the formula.</td>
        <td>Supports transpose and non-transpose.</td>
        <td>INT4, FLOAT4_E2M1, INT32, FLOAT, INT8</td>
        <td>FRACTAL_NZ</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Scale parameter for dequantizing the input. It corresponds to the input <code>antiquantScale</code> in the formula.</td>
        <td>Has the same contiguity requirement as weight.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1, 2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>Optional input</td>
        <td>The offset parameter for input dequantization, which is <code>antiquantOffset</code> in the formula.</td>
        <td>The contiguity requirement is the same as that of weight.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1, 2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>Optional input</td>
        <td>Reserved parameter, not in use. A null pointer must be passed in.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffsetOptional</td>
        <td>Optional input</td>
        <td>Reserved parameter, currently not used. A null pointer must be passed in.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Optional input</td>
        <td>Bias input, corresponding to <code>bias</code> in the formula.</td>
        <td>Pass in a null pointer when not required.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1, 2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>antiquantGroupSize</td>
        <td>Input</td>
        <td>In pergroup and MX<a href="../../../docs/en/context/quant_mode_introduction.md" target="_blank">quantization mode</a>, this input is the groupSize for the dequantization computation on weight, indicating the size of each group when the reduce axis is grouped.</td>
        <td>        In pergroup<a href="../../../docs/en/context/quant_mode_introduction.md" target="_blank">quantization mode</a>, the value range is [32, k-1] and the value must be a multiple of 32;<br>
        in MX<a href="../../../docs/en/context/quant_mode_introduction.md" target="_blank">quantization mode</a>, the value must be 32;<br>
        in perchannel<a href="../../../docs/en/context/quant_mode_introduction.md" target="_blank">quantization mode</a>, the value must be 0.<br>
        </td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Computation output, <code>y</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device side.</td>
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
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
        <td>If a passed parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The shape, data format, or contiguity of the input tensor does not meet requirements.</td>
      </tr>
      <tr>
        <td>Empty tensor use cases are not supported.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>The product model is not supported.</td>
      </tr>
    </tbody>
  </table>

## aclnnWeightQuantBatchMatmulNz

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnWeightQuantBatchMatmulNzGetWorkspaceSize.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - Deterministic computation description: aclnnWeightQuantBatchMatmulNz defaults to a deterministic implementation.
  - Supported quantization modes: perchannel [quantization mode](../../../docs/en/context/quant_mode_introduction.md), pergroup [quantization mode](../../../docs/en/context/quant_mode_introduction.md), and MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md).
  - The input and output support the following data type and shape combinations:

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>:

      | [Quantization Mode](../../../docs/en/context/quant_mode_introduction.md) | x | Weight | antiquantScale | antiquantOffsetOptional | biasOptional | y | antiquantScale Shape | antiquantOffsetOptional Shape |
      |------------| ----     | ----------------- | ----------- | ------------- | --------------------- | -------- | ------------------------------- | ------------------------------------ |
      | perchannel | BFLOAT16 | INT32/INT4        | BFLOAT16    | null/BFLOAT16 | null/BFLOAT16/FLOAT32 | BFLOAT16 | (1, n)/(n,)                     | null/(1, n)/(n,)                     |
      | perchannel | FLOAT16  | INT32/INT4        | FLOAT16     | null/FLOAT16  | null/FLOAT16          | FLOAT16  | (1, n)/(n,)                     | null/(1, n)/(n,)                     |
      | perchannel | BFLOAT16 | INT8              | BFLOAT16    | null/BFLOAT16 | null/FLOAT32          | BFLOAT16 | (1, n)/(n,)                     | null/(1, n)/(n,)                     |
      | perchannel | FLOAT16  | INT8              | FLOAT16     | null/FLOAT16  | null/FLOAT16          | FLOAT16  | (1, n)/(n,)                     | null/(1, n)/(n,)                     |
      |  pergroup  | BFLOAT16 | INT32/INT4        | BFLOAT16    | null/BFLOAT16 | null/BFLOAT16/FLOAT32 | BFLOAT16 | (ceil(k/antiquantGroupSize), n) | null/(ceil(k/antiquantGroupSize), n) |
      |  pergroup  | FLOAT16  | INT32/INT4        | FLOAT16     | null/FLOAT16  | null/FLOAT16          | FLOAT16  | (ceil(k/antiquantGroupSize), n) | null/(ceil(k/antiquantGroupSize), n) |
      |  pergroup  | BFLOAT16 | FLOAT/FLOAT4_E2M1 | BFLOAT16    | null          | null/BFLOAT16         | BFLOAT16 | (ceil(k/antiquantGroupSize), n) | null                                 |
      |  pergroup  | FLOAT16  | FLOAT/FLOAT4_E2M1 | FLOAT16     | null          | null/FLOAT16          | FLOAT16  | (ceil(k/antiquantGroupSize), n) | null                                 |
      |     MX     | BFLOAT16 | FLOAT/FLOAT4_E2M1 | FLOAT8_E8M0 | null          | null/BFLOAT16         | BFLOAT16 | (ceil(k/32), n)                 | null                                 |
      |     MX     | FLOAT16  | FLOAT/FLOAT4_E2M1 | FLOAT8_E8M0 | null          | null/FLOAT16          | FLOAT16  | (ceil(k/32), n)                 | null                                 |

      - The shape of x is (m, k), the shape of y is (m, n), and the shape of biasOptional is null/(1, n)/(n,).
      - When the data type of weight is INT32 or FLOAT, it indicates tightly packed INT4 or FLOAT4_E2M1, and the following constraints must be met:
        - The last dimension of the original ND matrix is 8-aligned.
        - Before calling this API, use the `aclnnConvertWeightToINT4Pack` API to convert data from sparse-layout INT32/FLOAT to tightly packed INT4/FLOAT4_E2M1 and from ND to FRACTAL_NZ. For details, see [a sample](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md).
        - The shape of the FRACTAL_NZ matrix passed to this API is (ceil(n/16), ceil(k/16), 16, 2).
        - weight only supports non-transposed mode, and the shape of the original ND matrix is (k, n).
      - When the data type of weight is INT4 or FLOAT4_E2M1, the following constraints must be met:
        - The last dimension of the original ND matrix must be 2-aligned.
        - Before calling this API, use the `aclnnConvertWeightToINT4Pack` API to convert data from ND to FRACTAL_NZ. For details, see [a sample](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md).
        - The shape of the FRACTAL_NZ matrix passed to this API is (ceil(n/16), ceil(k/16), 16, 16).
        - weight only supports non-transposed mode, and the shape of the original ND matrix is (k, n).
      - When the data type of weight is INT8, the following constraints must be met:
        - The shape of the FRACTAL_NZ matrix passed to this API is (ceil(n/32), ceil(k/16), 16, 32).
        - k is in the range [1, 65535], and n is in the range [2, 65535].
        - weight supports transposed and non-transposed formats, with the original ND matrix shape being (n, k) or (k, n).
      - m is in the range [1, 2^31-1].

    <!-- end id7 -->

## Example

  For reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

- x is FLOAT16 and weight is FLOAT32. Example: call the `aclnnConvertWeightToINT4Pack` API to assist the call:

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"

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

  template <typename T1>
  inline T1 CeilDiv(T1 a, T1 b)
  {
      return b == 0 ? a : (a + b - 1) / b;
  };
  template <typename T1>
  inline T1 CeilAlign(T1 a, T1 b)
  {
      return (a + b - 1) / b * b;
  };

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
      const aclTensor* weight, const aclTensor* weightInt4Pack, uint64_t* workspaceSize, aclOpExecutor** executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(
      void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

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
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  int GetSize(std::vector<int64_t>& shape)
  {
      int64_t size = 1;
      for (auto i : shape) {
          size *= i;
      }
      return size;
  }

  template <typename T>
  int CreateAclTensorB4(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor, aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
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
      if (format == aclFormat::ACL_FORMAT_ND) {
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
              shape.size(), *deviceAddr);
      } else {
          std::vector<int64_t> nzShape;
          if (dataType == aclDataType::ACL_INT4 || dataType == aclDataType::ACL_FLOAT4_E2M1) {
              nzShape = {CeilDiv(shape[1], (int64_t)16), CeilDiv(shape[0], (int64_t)16), 16, 16};
          } else {
              nzShape = {CeilDiv(shape[1], (int64_t)2), CeilDiv(shape[0], (int64_t)16), 16, 2};
          }
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(),
              nzShape.size(), *deviceAddr);
      }

      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
  {
      int64_t m = resultShape[0];
      int64_t n = resultShape[1];
      for (size_t i = 0; i < m; i++) {
          printf(i == 0 ? "[[" : " [");
          for (size_t j = 0; j < n; j++) {
              printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
              if (j == 2 && j + 3 < n) {
                  printf("..., ");
                  j = n - 4;
              }
          }
          printf(i < m - 1 ? "],\n" : "]]\n");
          if (i == 2 && i + 3 < m) {
              printf(" ... \n");
              i = m - 4;
          }
      }
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnWeightQuantBatchMatmulNzTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      aclDataType weightPackedDtype = aclDataType::ACL_FLOAT; // Optional: ACL_FLOAT type
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs as required by the API.
      int64_t m = 16;
      int64_t k = 64;
      int64_t n = 64;
      int64_t groupSize = 32;
      int64_t weightDim0 = k;
      int64_t weightDim1 = n;
      bool isWeightTransposed = false;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {k / groupSize, n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* weightB4PackDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* y = nullptr;
      aclTensor* antiquantScale = nullptr;
      std::vector<int64_t> weightPackedShape;
      weightPackedShape = {weightDim0, weightDim1 / 8};
      std::vector<uint16_t> xHostData(GetSize(xShape), 0b0011110000000000); // 1.0 in float16
      std::vector<float> weightHostData(GetSize(weightShape), 1.0); // 1.0 in fp32, converted to 1.0 in fp4_e2m1 after int4pack
      std::vector<float> yHostData(GetSize(yShape), 0);

      std::vector<uint16_t> antiquantScaleHostData(GetSize(antiquantScaleShape), 0b0011110000000000); // 1.0 in float16

      // Create the x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the other aclTensor.
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create y aclTensor.
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create antiquantScale aclTensor.
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create yFp16 aclTensor.
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ; // Optional: ACL_FORMAT_ND
      aclTensor* weightPacked = nullptr;

      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); // Each B8 value stores two B4 values, so divide by 2 here.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
          weightB4PackHostData.resize(CeilAlign(weightDim1 / 2, (int64_t)8) * CeilAlign(weightDim0, (int64_t)16), 0);
      }
      // Create weightInt4Pack aclTensor.
      ret = CreateAclTensorB4(
          weightB4PackHostData, weightPackedShape, &weightB4PackDeviceAddr, weightPackedDtype, &weightPacked,
          weightFormat);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPackedTensorPtr(weightPacked, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightPackedDeviceAddrPtr(weightB4PackDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Convert weight from int32 to int4pack.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      void* workspacePackAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspacePackAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspacePackAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspacePackAddrPtr.reset(workspacePackAddr);
      }
      ret = aclnnConvertWeightToINT4Pack(workspacePackAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnWeightQuantBatchMatmulNz.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(
          x, weightPacked, antiquantScale, nullptr, nullptr, nullptr, nullptr, groupSize, yFp16, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnWeightQuantBatchMatmulNz.
      ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Convert the output to FP32.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Get the output value and copy the result from the device side to the host side. Modify it according to the specific API definition.
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      PrintMat(resultData, yShape);
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnWeightQuantBatchMatmulNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- Example where x is FLOAT16 and weight is INT8. The `aclnnTransMatmulWeight` API must be called to assist the call:

  ```Cpp
  #include <cmath>
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

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
      // Boilerplate: initialize resources.
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
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWeightNz(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size =static_cast<uint64_t>(GetShapeSize(shape) * sizeof(T));
      const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      aclDestroyIntArray(mat2Size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMalloc to allocate device-side memory.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  // Convert the uint16_t representation of FP16 to float.
  float Fp16ToFloat(uint16_t h)
  {
      int s = (h >> 15) & 0x1;  // sign
      int e = (h >> 10) & 0x1F; // exponent
      int f = h & 0x3FF;        // fraction
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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnWeightQuantBatchMatmulNzA16W8Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs. Custom construction is required based on the API.
      int64_t m = 2;
      int64_t k = 16;
      int64_t n = 8;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> yShape = {m, n};
      std::vector<int64_t> antiquantScaleShape = {n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* antiquantScale = nullptr;
      aclTensor* y = nullptr;
      std::vector<uint16_t> xHostData(m * k, 0b0011110000000000); // 0b0011110000000000 is fp16 1.0.
      std::vector<int8_t> weightHostData(k * n, 1);
      std::vector<float> yHostData(m * n, 0);

      std::vector<uint16_t> antiquantScaleHostData(n, 0b0011110000000000); // 0b0011110000000000 is 1.0 in fp16.

      // Create x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create weight aclTensor.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      ret = CreateAclTensorWeightNz(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Call the first-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(weight, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      void* workspaceTransMatmulAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceTransMatmulAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceTransMatmulAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceTransMatmulAddrPtr.reset(workspaceTransMatmulAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceTransMatmulAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret);
                return ret);

      // Create antiquantScale aclTensor.
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create y aclTensor.
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, and replace it with the specific API name.
      // Call the first-phase API of aclnnWeightQuantBatchMatmulNz.
      ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(
          x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API aclnnWeightQuantBatchMatmulNz
      ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait for the task execution to complete
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the definition of the specific API
      auto size = GetShapeSize(yShape);
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual
      // Set deviceId based on the actual device
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnWeightQuantBatchMatmulNzA16W8Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulNzA16W8Test failed. ERROR: %d\n", ret);
                     return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- npu="950" id8 -->
- Example where x is FLOAT16 and weight is INT4 with data format FRACTAL_NZ_C0_16 (supported only on <term>Ascend 950PR&950DT products</term>. The weight in FRACTAL_NZ_C0_16 format is generated by the non-transpose path of the `aclnnWeightQuantPreprocess` API, so the `aclnnConvertWeightToINT4Pack` API is not required for conversion, and the storage shape is (ceil(n/16), ceil(k/16), 16, 16)):

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"

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

  template <typename T1>
  inline T1 CeilDiv(T1 a, T1 b)
  {
      return b == 0 ? a : (a + b - 1) / b;
  };

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
      // (boilerplate) Initialize resources
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
      // Call aclrtMalloc to allocate device memory
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from host memory to device memory
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

  int GetSize(std::vector<int64_t>& shape)
  {
      int64_t size = 1;
      for (auto i : shape) {
          size *= i;
      }
      return size;
  }

  // Create an aclTensor with tightly packed INT4 weight and data format FRACTAL_NZ_C0_16,
  // where the logical shape is passed in as the original ND matrix (k, n), and the storageShape is (ceil(n/16), ceil(k/16), 16, 16).
  int CreateAclTensorNzC016(const std::vector<int8_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor)
  {
      // 4bit tightly packed, one B8 element stores 2 B4 elements, and the memory size is half the number of elements.
      auto size = hostData.size() * sizeof(int8_t);
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
      std::vector<int64_t> storageShape = {CeilDiv(shape[1], (int64_t)16), CeilDiv(shape[0], (int64_t)16), 16, 16};
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ_C0_16, storageShape.data(), storageShape.size(),
                                *deviceAddr);
      return 0;
  }

  void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
  {
      int64_t m = resultShape[0];
      int64_t n = resultShape[1];
      for (size_t i = 0; i < m; i++) {
          printf(i == 0 ? "[[" : " [");
          for (size_t j = 0; j < n; j++) {
              printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
              if (j == 2 && j + 3 < n) {
                  printf("..., ");
                  j = n - 4;
              }
          }
          printf(i < m - 1 ? "],\n" : "]]\n");
          if (i == 2 && i + 3 < m) {
              printf(" ... \n");
              i = m - 4;
          }
      }
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnWeightQuantBatchMatmulNzTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output, which need to be custom-built based on the API.
      int64_t m = 16;
      int64_t k = 64;
      int64_t n = 64;
      int64_t groupSize = 32;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {k / groupSize, n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* y = nullptr;
      aclTensor* antiquantScale = nullptr;
      std::vector<uint16_t> xHostData(GetSize(xShape), 0b0011110000000000); // 1.0 in fp16
      // The weight is packed in 4 bits. One int8 stores two int4 values, and 0x11 means both int4 values are 1.
      std::vector<int8_t> weightHostData(k * n / 2, 0x11);
      std::vector<float> yHostData(GetSize(yShape), 0);

      std::vector<uint16_t> antiquantScaleHostData(GetSize(antiquantScaleShape), 0b0011110000000000); // 1.0 in fp16

      // Create x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weight aclTensor. The weight is densely packed INT4 with the data format FRACTAL_NZ_C0_16,
      // In actual use cases, the weight is generated by the non-transpose path of aclnnWeightQuantPreprocess. Equivalent data is constructed directly here.
      ret = CreateAclTensorNzC016(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT4, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create y aclTensor.
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create antiquantScale aclTensor.
      ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                            aclDataType::ACL_FLOAT16, &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(antiquantScale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create yFp16 aclTensor.
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, and change it to the specific API name.
      // Call the first-phase API of aclnnWeightQuantBatchMatmulNz.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr,
                                                          groupSize, yFp16, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed to allocate workspace. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnWeightQuantBatchMatmulNz.
      ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait for the task execution to finish.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Convert the output to FP32.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed to allocate workspace. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the definition of the specific API.
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      PrintMat(resultData, yShape);
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnWeightQuantBatchMatmulNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulNzTest failed. ERROR: %d\n", ret);
                     return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
<!-- end id8 -->
