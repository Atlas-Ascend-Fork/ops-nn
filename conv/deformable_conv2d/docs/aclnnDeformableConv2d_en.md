# aclnnDeformableConv2d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T18:01:47.354Z -->

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

- Description: Implements the convolution function, supporting 2D convolution, as well as deformable convolution and grouped convolution.

- Formula:

  Assume that the shape of input is [N, inC, inH, inW] and the shape of out is [N, outC, outH, outW]. outH and outW are calculated based on the existing parameters:

  $$
  outH = (inH + padding[0] + padding[1] - ((K_H - 1) * dilation[2] + 1)) // stride[2] + 1
  $$

  $$
  outW = (inW + padding[2] + padding[3] - ((K_W - 1) * dilation[3] + 1)) // stride[3] + 1
  $$

  Standard convolution computes the sampling point indices:

  $
  x = -padding[2] + ow*stride[3] + kw*dilation[3], kw ranges from 0 to K\_W-1
  $

  $
  y = -padding[0] + oh*stride[2] + kh*dilation[2], kh ranges from 0 to K\_H-1
  $

  Based on the input offset, perform deformable convolution and calculate the offset subscripts:

  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  Use bilinear interpolation to calculate the value of the offset point:

  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$

  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\
  $$

  $$
  deformOut(x, y) = weight_{00} * input(x0, y0) + weight_{01} * input(x0,y1) + weight_{10} * input(x1, y0) + weight_{11} * input(x1,y1)
  $$

  Perform convolution computation to obtain the final output:

  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{deformOut}(N_i, k)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnDeformableConv2dGetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the operator computation process. Then, call "aclnnDeformableConv2d" to perform the computation.

```cpp
aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(
  const aclTensor*   x,
  const aclTensor*   weight,
  const aclTensor*   offset,
  const aclTensor*   biasOptional,
  const aclIntArray* kernelSize,
  const aclIntArray* stride,
  const aclIntArray* padding,
  const aclIntArray* dilation,
  int64_t            groups,
  int64_t            deformableGroups,
  bool               modulated,
  aclTensor*         out,
  aclTensor*         deformOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnDeformableConv2d(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnDeformableConv2dGetWorkspaceSize

- **Parameters**:

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
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Original input data. Corresponds to `input` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is [N, inC, inH, inW], where inH * inW cannot exceed 2147483647.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>4D tensor of learnable filters. Corresponds to `weight` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type and data format are the same as those of input parameter `x`.</li><li>The shape is [outC, inC/groups, K_H, K_W].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>A 4D tensor for x-y coordinate offsets and masks. Corresponds to `offset` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type and data format are the same as those of input parameter `x`.</li><li>When `modulated` is True, the shape is [N, 3 * deformableGroups * K_H * K_W, outH, outW]; when `modulated` is False, the shape is [N, 2 * deformableGroups * K_H * K_W, outH, outW].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Optional input parameter, indicating the 1D tensor for the additional bias of the filter output, corresponding to `bias` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type must be the same as that of input parameter `x`.</li><li>Null pointer if not required. The shape is [outC] if present.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Specifies the kernel size, corresponding to `K_H` and `K_W` in the formula.</td>
      <td>The size is 2 (K_H, K_W). Each element must be greater than 0. K_H * K_W cannot exceed 2048. K_H * K_W * inC/groups cannot exceed 65535.</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Specifies the sliding window stride of each input dimension. The corresponding formula variable is `stride`.</td>
      <td>The value of size is 4, and each element is greater than 0. The dimension order is interpreted based on the data format of `x`. The N and C dimensions must be set to 1.</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Specifies the number of pixels to be added to each side of the input (top, bottom, left, and right). The corresponding formula variable is `padding`.</td>
      <td>The size is 4.</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input</td>
      <td>Dilation coefficient of each dimension of the input, corresponding to `dilation` in the formula.</td>
      <td>The size is 4. Each element is greater than zero. The dimension order is interpreted based on the data format of x. The N and C dimensions must be set to 1.</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>Input</td>
      <td>Number of grouped connections from input channels to output channels.</td>
      <td>inC and outC must both be divisible by groups, and groups must be greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformableGroups</td>
      <td>Input</td>
      <td>Indicates the number of deformable groups.</td>
      <td>inC must be divisible by deformableGroups, and deformableGroups must be greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>Input</td>
      <td>Reserved parameter. Indicates whether `offset` contains a mask. If true, the mask is contained in `offset`; if false, it is not.</td>
      <td>Currently, only true is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the output data. Corresponds to `out` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type and data format are the same as those of `x`.</li><li>The shape is [N, outC, outH, outW].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>deformOutOptional</td>
      <td>Output</td>
      <td>Optional output. Represents the sampling points of deformable convolution. It is `deformOut` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type and data format are the same as the data type and data format of `x`.</li><li>The shape is [N, inC, outH * K_H, outW * K_W].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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

- **Return value**:

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
      <td>x, weight, offset, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or data format of x, weight, offset, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>If deformOutOptional is not a null pointer, its data type or data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>When biasOptional is not a null pointer, the data type or data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of x, weight, offset, biasOptional, out, and deformOutOptional are inconsistent with those described in the parameter description.</td>
    </tr>
    <tr>
      <td>The sizes of kernelSize, stride, padding, and dilation are inconsistent with those described in the parameter description.</td>
    </tr>
    <tr>
      <td>K_H*K_W exceeds 2048, or K_H*K_W*inC/groups exceeds 65535.</td>
    </tr>
  </tbody></table>

## aclnnDeformableConv2d

- **Parameters:**

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory requested on the device, obtained by the first-phase API aclnnDeformableConv2dGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the computation flow of the operator.</td>
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

- Deterministic computation:
  - aclnnDeformableConv2d defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 1146px"><colgroup>
    <col style="width:345px">
    <col style="width:491px">
    <col style="width:310px">
    </colgroup>
   <thead>
    <tr>
     <th>Parameter</th>

     <th><term>Atlas A2 products</term>, <term>Atlas A3 products</term></th>


     <th><term>Ascend 950PR&950DT products</term></th>

   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">input, weight, offset, out, deformOutOptional</th>
     <td>           Data formats ND and NCHW are supported.
     </td>
     <td>           Only the NCHW data format is supported.
     </td>
   </tr>
   </tbody>
  </table>

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_deformable_conv2d.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device.
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

    auto format = shape.size() == 1 ? ACL_FORMAT_ND : ACL_FORMAT_NCHW;
    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (boilerplate) Initialize device/stream. Refer to the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> xShape = {1, 6, 2, 4};
    std::vector<int64_t> weightShape = {4, 3, 5, 5};
    std::vector<int64_t> offsetShape = {1, 75, 2, 4};
    std::vector<int64_t> biasShape = {4};
    std::vector<int64_t> outShape = {1, 4, 2, 4};
    std::vector<int64_t> deformOutShape = {1, 6, 10, 20};
    std::vector<int64_t> kernelSize = {5, 5};
    std::vector<int64_t> stride = {1, 1, 1, 1};
    std::vector<int64_t> padding = {2, 2, 2, 2};
    std::vector<int64_t> dilation = {1, 1, 1, 1};
    int64_t groups = 2;
    int64_t deformableGroups = 1;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* deformOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* deformOut = nullptr;
    std::vector<float> xHostData(1 * 6 * 2 * 4, 1);
    std::vector<float> weightHostData(4 * 3 * 5 * 5, 1);
    std::vector<float> offsetHostData(1 * 75 * 2 * 4, 1);
    std::vector<float> biasHostData(4, 0);
    std::vector<float> outHostData(1 * 4 * 2 * 4, 0);
    std::vector<float> deformOutHostData(1 * 6 * 10 * 20, 0);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the bias aclTensor.
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the deformOut aclTensor.
    ret = CreateAclTensor(deformOutHostData, deformOutShape, &deformOutDeviceAddr, aclDataType::ACL_FLOAT, &deformOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create kernelSize aclIntArray.
    const aclIntArray* kernelSizeArray = aclCreateIntArray(kernelSize.data(), kernelSize.size());
    CHECK_RET(kernelSizeArray != nullptr, return ret);
    // Create stride aclIntArray.
    const aclIntArray* strideArray = aclCreateIntArray(stride.data(), stride.size());
    CHECK_RET(strideArray != nullptr, return ret);
    // Create padding aclIntArray.
    const aclIntArray* paddingArray = aclCreateIntArray(padding.data(), padding.size());
    CHECK_RET(paddingArray != nullptr, return ret);
    // Create dilation aclIntArray.
    const aclIntArray* dilationArray = aclCreateIntArray(dilation.data(), dilation.size());
    CHECK_RET(dilationArray != nullptr, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnDeformableConv2d.
    ret = aclnnDeformableConv2dGetWorkspaceSize(
        x, weight, offset, bias, kernelSizeArray, strideArray, paddingArray, dilationArray, groups, deformableGroups,
        true, out, deformOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnDeformableConv2d.
    ret = aclnnDeformableConv2d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2d failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values, and copy the results from device memory to host memory. Modify them according to the definition of the specific API.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(offset);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(deformOut);

    // 7. Release device resources. Modify the code based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(deformOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
