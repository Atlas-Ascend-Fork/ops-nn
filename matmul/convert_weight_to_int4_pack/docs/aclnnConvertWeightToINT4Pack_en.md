# aclnnConvertWeightToINT4Pack

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:09:47.333Z -->

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

Preprocesses the input weight data to convert the layout of low-bit data from sparse storage to tightly packed storage. When the [data format](../../../docs/en/context/data_format.md) of the output weightInt4Pack is declared as FRACTAL_NZ, this operator converts the [data format](../../../docs/en/context/data_format.md) from ND to FRACTAL_NZ.

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Weight input data of the INT32 data type is packed into tightly packed INT4 data.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: Packs the weight of the INT32 type into tightly packed INT4, and packs the weight of the FLOAT type into tightly packed FLOAT4_E2M1.

<!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnConvertWeightToINT4PackGetWorkspaceSize" to obtain the workspace size required for the computation and an executor that contains the operator computation process, and then call "aclnnConvertWeightToINT4Pack" to execute the computation.

```cpp
aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
  const aclTensor *weight,
  aclTensor       *weightInt4Pack,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnConvertWeightToINT4Pack(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnConvertWeightToINT4PackGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1078px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 320px">
  <col style="width: 183px">
  <col style="width: 183px">
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
      <td>weight</td>
      <td>Input</td>
      <td>Low-bit quantized weight of Matmul-like operators, where the 32-bit type carries 4-bit weight values.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightInt4Pack</td>
      <td>Output</td>
      <td>Low-bit quantized weight of Matmul-like operators, where the weight values are 4-bit and tightly packed.</td>
      <td>-</td>
      <td>INT4,INT32</td>
      <td>ND,FRACTAL_NZ</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the device.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
    <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If any required input, output, or attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The shape dimensions of the passed-in weight and weightInt4Pack do not meet the requirements.</td>
    </tr>
    <tr>
      <td>The data types of the passed-in weight and weightInt4Pack are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape sizes of the passed-in weight and weightInt4Pack do not meet the constraint requirements.</td>
    </tr>
    <tr>
      <td>An empty tensor is passed in.</td>
    </tr>
    <tr>
      <td>The format of the input tensor is not ND.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_RUNTIME_ERROR</td>
      <td rowspan="2">361001</td>
      <td>An exception occurs when data is copied from the host to the device.</td>
    </tr>
    <tr>
      <td>An exception occurs when data is copied from the device to the host.</td>
    </tr>
  </tbody>
  </table>

## aclnnConvertWeightToINT4Pack

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnConvertWeightToINT4PackGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="950,A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: aclnnConvertWeightToINT4Pack defaults to a deterministic implementation.

  <!-- end id9 -->

- The relationships between data types and data formats of parameters are as follows:

    <!-- npu="A3,910b" id10 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
      <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
      <col style="width: 200px">
      <col style="width: 121px">
      <col style="width: 175px">
      <col style="width: 170px">
      <col style="width: 160px">
      <col style="width: 208px">
      <col style="width: 185px">
      </colgroup>
      <thead>
        <tr>
          <th>weight data type</th>
          <th>weight data format</th>
          <th>weightInt4Pack data type</th>
          <th>weightInt4Pack data format</th>
          <th>weight shape</th>
          <th>weightInt4Pack view shape</th>
          <th>weightInt4Pack storage shape</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>INT32 (carries data of the INT4 type, with a data representation range of [-8, 7])</td>
          <td>ND</td>
          <td>INT4</td>
          <td>ND</td>
          <td>The last dimension is aligned to 2.</td>
          <td>Remains consistent with the input weight, i.e., (dim0, dim1).</td>
          <td>Same as view shape</td>
        </tr>
        <tr>
          <td>INT32 (carries INT4 data, data representation range [-8, 7])</td>
          <td>ND</td>
          <td>INT4</td>
          <td>FRACTAL_NZ</td>
          <td>The last dimension is aligned to 2.</td>
          <td>Remains consistent with the input weight, i.e., (dim0, dim1).</td>
          <td>(⌈dim1/64⌉, ⌈dim0/16⌉, 16, 64)</td>
        </tr>
        <tr>
          <td>INT32 (carrying INT4 type data, data representation range [-8, 7])</td>
          <td>ND</td>
          <td>INT32 (1 INT32 data stores 8 INT4 data)</td>
          <td>ND</td>
          <td>The last dimension is aligned to 8.</td>
          <td>The last dimension is 1/8 of the last dimension of weight, i.e., (dim0, dim1/8).</td>
          <td>Same as view shape.</td>
        </tr>
        <tr>
          <td>INT32 (carrying INT4 type data, data representation range [-8, 7])</td>
          <td>ND</td>
          <td>INT32 (1 INT32 data stores 8 INT4 data)</td>
          <td>FRACTAL_NZ</td>
          <td>The last dimension is aligned to 8.</td>
          <td>The last dimension is 1/8 of the last dimension of weight, i.e., (dim0, dim1/8).</td>
          <td>(⌈dim1/64⌉, ⌈dim0/16⌉, 16, 8)</td>
        </tr>
      </tbody></table>

    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>:
      <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
      <col style="width: 200px">
      <col style="width: 121px">
      <col style="width: 175px">
      <col style="width: 170px">
      <col style="width: 160px">
      <col style="width: 208px">
      <col style="width: 185px">
      </colgroup>
      <thead>
        <tr>
          <th>weight data type</th>
          <th>weight data format</th>
          <th>weightInt4Pack data type</th>
          <th>weightInt4Pack data format</th>
          <th>weight shape</th>
          <th>weightInt4Pack view shape</th>
          <th>weightInt4Pack storage shape</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>INT32 (carrying INT4 data, data representation range [-8, 7]) or FLOAT (carrying FLOAT4_E2M1 data, data representation range [-6.0, 6.0])</td>
          <td>ND</td>
          <td>INT4 or FLOAT4_E2M1</td>
          <td>ND</td>
          <td>The last dimension is aligned to 2.</td>
          <td>Remains consistent with the input weight, i.e., (dim0, dim1).</td>
          <td>Same as the view shape.</td>
        </tr>
        <tr>
          <td>INT32 (carrying INT4 data, data representation range [-8, 7]) or FLOAT (carrying FLOAT4_E2M1 data, data representation range [-6.0, 6.0])</td>
          <td>ND</td>
          <td>INT4 or FLOAT4_E2M1</td>
          <td>FRACTAL_NZ</td>
          <td>The last dimension is aligned to 2.</td>
          <td>Remains consistent with input weight, that is, (dim0, dim1)</td>
          <td>(⌈dim1/16⌉, ⌈dim0/16⌉, 16, 16)</td>
        </tr>
        <tr>
          <td>INT32 (carrying INT4 data, data representation range [-8, 7]) or FLOAT (carrying FLOAT4_E2M1 data, data representation range [-6.0, 6.0])</td>
          <td>ND</td>
          <td>INT32 (1 INT32 stores 8 INT4 data items) or FLOAT (1 FLOAT stores 8 FLOAT4_E2M1 data items).</td>
          <td>ND</td>
          <td>The last dimension is aligned to 8.</td>
          <td>The last dimension is 1/8 of the last dimension of weight, i.e., (dim0, dim1/8).</td>
          <td>Same as view shape.</td>
        </tr>
        <tr>
          <td>INT32 (carries INT4 type data, data representation range [-8, 7]) or FLOAT (carries FLOAT4_E2M1 type data, data representation range [-6.0, 6.0])</td>
          <td>ND</td>
          <td>INT32 (one INT32 stores 8 INT4 data) or FLOAT (one FLOAT stores 8 FLOAT4_E2M1 data)</td>
          <td>FRACTAL_NZ</td>
          <td>The last dimension is aligned to 8.</td>
          <td>The last dimension is 1/8 of the last dimension of weight, i.e., (dim0, dim1/8).</td>
          <td>(⌈dim1/16⌉, ⌈dim0/16⌉, 16, 2)</td>
        </tr>
        <tr>
          <td>INT32 (carries INT4-type data, with data representation range [-8, 7])</td>
          <td>ND</td>
          <td>INT4 (1 INT32 data value stores 8 INT4 data values)</td>
          <td>FRACTAL_NZ</td>
          <td>The last dimension is aligned to 8.</td>
          <td>The last dimension is 1/8 of the last dimension of weight, that is, (dim0, dim1/8).</td>
          <td>(⌈dim1/32⌉, ⌈dim0/16⌉, 16, 32)</td>
        </tr>
      </tbody></table>

    <!-- end id11 -->

## Example

<!-- npu="A3,910b" id12 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).
  Two APIs are available for pseudo-quantization: aclnnWeightQuantBatchMatmulV2 and aclnnWeightQuantBatchMatmulV3. aclnnWeightQuantBatchMatmulV2 is used as an example here.

  ```cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
      uint64_t *workspaceSize, aclOpExecutor **executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
      aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
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
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute contiguous tensor strides.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  template <typename T>
  int CreateAclTensorInt4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute contiguous tensor strides.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create aclTensor.
    if (format == aclFormat::ACL_FORMAT_ND) {
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    } else {
      std::vector<int64_t> nzShape;
      if (dataType == aclDataType::ACL_INT4) {
          nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 64};
      } else {
          nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 8};
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
  }

  int main() {
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightInt4PackDtype = aclDataType::ACL_INT4;
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    bool isWeightTransposed = true;

    // 2. Construct the inputs and outputs, which need to be custom-built based on the API.
    int64_t m = 16;
    int64_t k = 72;
    int64_t n = 17;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
      weightDim0 = n;
      weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightInt4PackShape;
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      weightInt4PackShape = {weightDim0, weightDim1};
    } else {
      weightInt4PackShape = {weightDim0, weightDim1/8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightInt4PackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {n};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(n, 1);

    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      std::vector<int8_t> weightInt4PackHostData(n * k / 2, 0); // One int8 data value stores 2 int4 data values, so divide by 2 here.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/2, 32) * CEIL_ALIGN(weightDim0, 16), 0);
      }
      // Create the weightInt4Pack aclTensor.
      ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                weightInt4PackDtype, &weightInt4Pack, weightFormat);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
      std::vector<int32_t> weightInt4PackHostData(n * k / 8, 1); // One int32 data value stores 8 int4 data values, so divide by 8 here.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/8, 8) * CEIL_ALIGN(weightDim0, 16), 0);
        ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                  weightInt4PackDtype, &weightInt4Pack, weightFormat);
      } else {
          // Create the weightInt4Pack aclTensor.
          ret = CreateAclTensor(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                weightInt4PackDtype, &weightInt4Pack);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // Create the y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the antiquantScale aclTensor.
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the xFp16 aclTensor.
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the antiquantScale aclTensor.
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the yFp16 aclTensor.
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change the name to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // Convert weight from int32 to int4pack.
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // If weight is transposed and the shape of weightInt4Pack is NZ, call aclInitTensor to convert it to a non-contiguous tensor.
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
      std::vector<int64_t> strides(weightInt4PackShape.size(), 1);
      for (int64_t i = weightInt4PackShape.size() - 2; i >= 0; i--) {
          strides[i] = weightInt4PackShape[i + 1] * strides[i + 1];
      }
      std::swap(strides[0], strides[1]);
      std::swap(weightInt4PackShape[0], weightInt4PackShape[1]);
      std::vector<int64_t> nzShape = {CEIL_DIV(k, 64), CEIL_DIV(n, 16), 16, 8};
      if (weightInt4PackDtype == aclDataType::ACL_INT4) {
          nzShape[3] = 64;
      }
      aclInitTensor(weightInt4Pack, weightInt4PackShape.data(), weightInt4PackShape.size(), weightInt4PackDtype, strides.data(), 0,
                    weightFormat, nzShape.data(), nzShape.size(), weightInt4PackDeviceAddr);
    }

    // Call cast to generate the FP16 input.
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Convert the output to FP32.
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code as required by the specific API definition.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code as required by the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightInt4Pack);
    aclDestroyTensor(antiquantScale);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp16);
    aclDestroyTensor(yFp16);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightInt4PackDeviceAddr);
    aclrtFree(antiquantScaleDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp16DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }

<!-- end id12 -->
<!-- npu="950" id13 -->
- <term>Ascend 950PR&950DT products</term>:

  The sample code is as follows (INT32 input). It is for reference only. For details on the compilation and execution process, refer to [Compilation and Execution Sample](../../../docs/en/context/compile_and_run_sample.md).
  For the FRACTAL_NZ use case on Ascend 950, you need to call the aclnnWeightQuantBatchMatmulNz API. The following uses aclnnWeightQuantBatchMatmulNz as an example.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_convert_weight_to_int4_pack.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  // Boilerplate for resource initialization
  int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  // Create an aclTensor with a contiguous layout in ND format, and copy hostData to the device.
  // The device-side memory is allocated based on the actual number of bytes of hostData (each element occupies sizeof(T) bytes).
  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create the aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  // Create an aclTensor with tightly packed 4-bit data (INT4/FLOAT4_E2M1) in FRACTAL_NZ storage format.
  // Pass in the logical shape based on the original ND matrix (k, n); on Ascend 950, the NZ storageShape is (ceil(n/16), ceil(k/16), 16, 16),
  // each block contains 16 * 16 4-bit data values (occupying 128 bytes), and the device-side memory is allocated based on the storageShape size.
  // hostData is the packed byte data (two 4-bit values stored in one int8); only hostData.size() bytes are copied,
  // the trailing aligned portion does not need to be initialized, and the entire memory block is rewritten after aclnnConvertWeightToINT4Pack is called.
  template <typename T>
  int CreateAclTensorNz(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = CEIL_DIV(shape[1], 16) * CEIL_DIV(shape[0], 16) * 128;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, hostData.size() * sizeof(T), hostData.data(), hostData.size() * sizeof(T),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor. The storageShape is computed based on the NZ block layout.
    std::vector<int64_t> nzShape = {CEIL_DIV(shape[1], 16), CEIL_DIV(shape[0], 16), 16, 16};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs.
    // This example is for the NZ scenario: the storage format of weightInt4Pack is FRACTAL_NZ, so the aclnnWeightQuantBatchMatmulNz API must be called for matmul.
    // (On Ascend 950, the weight input of the aclnnWeightQuantBatchMatmulV2/V3 APIs supports only the ND format.)
    // Computation scale: x is (m, k), weight is (k, n), and y is (m, n).
    // All input data values are 1, all antiquantScale values are 1, and each element of y is expected to be k.
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 64;
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {k, n};
    std::vector<int64_t> weightInt4PackShape = {k, n}; // The shape of INT4 is passed in as the number of elements.
    std::vector<int64_t> antiquantScaleShape = {n};    // In the perchannel scenario, the shape of scale is (n,).
    std::vector<int64_t> yShape = {m, n};

    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1); // INT32 weight, where each int32 carries one int4 value.
    std::vector<float> antiquantScaleHostData(n, 1);
    std::vector<float> yHostData(m * n, 0);
    // After INT4 is tightly packed, 2 int4 values are stored in 1 int8. The number of host data elements is half the number of weight elements.
    std::vector<int8_t> weightInt4PackHostData(k * n / 2, 0);

    // 3. Create the input and output aclTensors. Both the aclTensors and the device memory are managed by unique_ptr and automatically released on any exit path.
    // Create the x aclTensor (FP32).
    void* xDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    // Create the weight aclTensor (INT32).
    void* weightDeviceAddr = nullptr;
    aclTensor* weight = nullptr;
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    // Create the weightInt4Pack aclTensor (INT4, FRACTAL_NZ) to store the packed weight.
    void* weightInt4PackDeviceAddr = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    ret = CreateAclTensorNz(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                            aclDataType::ACL_INT4, &weightInt4Pack);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightInt4PackTensorPtr(weightInt4Pack, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightInt4PackDeviceAddrPtr(weightInt4PackDeviceAddr, aclrtFree);
    // Create the antiquantScale aclTensor (FP32).
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(antiquantScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
    // Create the y aclTensor (FP32).
    void* yDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    // Create the xFp16 aclTensor (FP16). Matmul requires the data types of x and antiquantScale to remain consistent.
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xFp16TensorPtr(xFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xFp16DeviceAddrPtr(xFp16DeviceAddr, aclrtFree);
    // Create the antiquantScaleFp16 aclTensor (FP16).
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleFp16TensorPtr(antiquantScaleFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleFp16DeviceAddrPtr(antiquantScaleFp16DeviceAddr, aclrtFree);
    // Create the yFp16 aclTensor (FP16), which is the output of matmul.
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);

    // 4. Call aclnnConvertWeightToINT4Pack to convert the weight from INT32 sparse storage to tightly packed INT4 storage,
    // When weightInt4Pack is in FRACTAL_NZ format, the operator also completes the ND-to-NZ layout conversion internally; this operator does not require workspace.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // For subsequent operators, request workspace as needed: the first-phase API returns workspaceSize, and the workspace is allocated before the second-phase API is executed and automatically released by unique_ptr.

    // 5. Call aclnnCast to convert x from FP32 to FP16.
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // Call aclnnCast to convert antiquantScale from FP32 to FP16.
    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize,
                                    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 6. Call aclnnWeightQuantBatchMatmulNz to perform pseudo-quantization matmul; passing 0 for antiquantGroupSize indicates the perchannel scenario.
    ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp16, nullptr, nullptr,
                                                        nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 7. Call aclnnCast to convert the output from FP16 to FP32.
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 8. (Boilerplate) Synchronize and wait for the task execution to complete, copy the result from the device side to the host side, and print it.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 9. (Boilerplate) Release resources such as the stream. The aclTensor and device memory have been automatically released by unique_ptr.
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

<!-- end id13 -->
<!-- npu="950" id14 -->
- <term>Ascend 950PR&950DT products</term>:
  The sample code is as follows (FLOAT input). It is for reference only. For details on the compilation and execution process, refer to [Compilation and Execution Sample](../../../docs/en/context/compile_and_run_sample.md).
  For the FRACTAL_NZ scenario on Ascend 950, call the aclnnWeightQuantBatchMatmulNz API. Here, aclnnWeightQuantBatchMatmulNz is used as an example.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_convert_weight_to_int4_pack.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  // Boilerplate: resource initialization.
  int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  // Create an aclTensor in ND format with contiguous layout, and copy hostData to the device.
  // The device memory size is allocated based on the actual bytes of hostData (each element occupies sizeof(T) bytes).
  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
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

  // Create an aclTensor with tightly packed 4-bit data (INT4/FLOAT4_E2M1) in FRACTAL_NZ storage format.
  // Pass in the logical shape of the original ND matrix (k, n). On Ascend 950, the NZ storageShape is (ceil(n/16), ceil(k/16), 16, 16),
  // where each block contains 16 * 16 4-bit data (occupying 128 bytes), and the device memory is allocated based on the storageShape size.
  // hostData is the packed byte data (two 4-bit values stored in one int8), and only hostData.size() bytes are copied.
  // The tail-aligned part does not need to be initialized, because the entire memory block is overwritten after calling aclnnConvertWeightToINT4Pack.
  template <typename T>
  int CreateAclTensorNz(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = CEIL_DIV(shape[1], 16) * CEIL_DIV(shape[0], 16) * 128;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, hostData.size() * sizeof(T), hostData.data(), hostData.size() * sizeof(T),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor, and compute the storageShape based on the NZ block layout.
    std::vector<int64_t> nzShape = {CEIL_DIV(shape[1], 16), CEIL_DIV(shape[0], 16), 16, 16};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. (boilerplate) Initialize the device/stream. For details, see the acl API reference.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs
    // This example is for the NZ use case: the storage format of weightInt4Pack is FRACTAL_NZ, so matmul needs to call the aclnnWeightQuantBatchMatmulNz API
    // (On Ascend 950, the weight of the aclnnWeightQuantBatchMatmulV2/V3 APIs supports only the ND format)
    // Computation scale: x is (m, k), weight is (k, n), y is (m, n), and antiquantScale is (ceil(k/groupSize), n)
    // All input data is 1, antiquantScale is 1, and each element of the expected y is k
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 64;
    int64_t antiquantGroupSize = 32; // groupSize for the pergroup use case
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {k, n};
    std::vector<int64_t> weightInt4PackShape = {k, n}; // Pass in the shape of FLOAT4_E2M1 as the number of elements
    std::vector<int64_t> antiquantScaleShape = {k / antiquantGroupSize, n}; // In the pergroup use case, the shape of scale is (ceil(k/groupSize), n)
    std::vector<int64_t> yShape = {m, n};

    std::vector<float> xHostData(m * k, 1);
    std::vector<float> weightHostData(k * n, 1); // FLOAT weight, where each float carries 1 fp4 data
    // Use uint8 to carry float8_e8m0 data. 0b01111111 indicates 1.0 of fp8_e8m0
    std::vector<uint8_t> antiquantScaleHostData(k * n / antiquantGroupSize, 0b01111111);
    std::vector<float> yHostData(m * n, 0);
    // After FLOAT4_E2M1 is tightly packed, 2 fp4 values are stored in 1 int8. The number of host data items is half the number of weight elements
    std::vector<int8_t> weightInt4PackHostData(k * n / 2, 0);

    // 3. Create the aclTensors of the inputs and outputs. Both aclTensors and device memory are managed by unique_ptr and automatically released on exiting from any path
    // Create the x aclTensor (FP32)
    void* xDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    // Create the weight aclTensor (INT32)
    void* weightDeviceAddr = nullptr;
    aclTensor* weight = nullptr;
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    // Create the weightInt4Pack aclTensor (FLOAT4_E2M1, FRACTAL_NZ) to store the packed weight
    void* weightInt4PackDeviceAddr = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    ret = CreateAclTensorNz(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                            aclDataType::ACL_FLOAT4_E2M1, &weightInt4Pack);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightInt4PackTensorPtr(weightInt4Pack, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightInt4PackDeviceAddrPtr(weightInt4PackDeviceAddr, aclrtFree);
    // Create the antiquantScaleFp8 aclTensor (FLOAT8_E8M0), which is directly used by matmul without conversion
    void* antiquantScaleFp8DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp8 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp8DeviceAddr,
                          aclDataType::ACL_FLOAT8_E8M0, &antiquantScaleFp8);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleFp8TensorPtr(antiquantScaleFp8, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleFp8DeviceAddrPtr(antiquantScaleFp8DeviceAddr, aclrtFree);
    // Create the y aclTensor (FP32)
    void* yDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    // Create the xFp16 aclTensor (FP16). When antiquantScale is FLOAT8_E8M0, the data type of x supports FLOAT16
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xFp16TensorPtr(xFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xFp16DeviceAddrPtr(xFp16DeviceAddr, aclrtFree);
    // Create the yFp16 aclTensor (FP16), the output of matmul
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);

    // 4. Call aclnnConvertWeightToINT4Pack to convert the weight in FLOAT sparse storage to tightly packed FLOAT4_E2M1 storage,
    // When weightInt4Pack is in FRACTAL_NZ format, the operator also completes the ND-to-NZ layout conversion internally. This operator does not need workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // Subsequent operators apply for workspace as needed: the first-phase API returns workspaceSize, and the second-phase API applies for workspace before execution. unique_ptr automatically releases it

    // 5. Call aclnnCast to convert x from FP32 to FP16
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 5. Call aclnnWeightQuantBatchMatmulNz to perform pseudo-quantization matmul. antiquantGroupSize is the groupSize for the pergroup use case
    ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp8, nullptr, nullptr,
                                                        nullptr, nullptr, antiquantGroupSize, yFp16, &workspaceSize,
                                                        &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 6. Call aclnnCast to convert the output from FP16 to FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 7. (boilerplate) Synchronize and wait until the task execution completes, copy the result from the device to the host, and print it
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    ```
    // 8. (Boilerplate) Release the stream and other resources. The aclTensor and device memory are automatically released by unique_ptr.
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

<!-- end id14 -->
