# aclnnAdaptiveMaxPool3d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:54:18.149Z -->

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

- Description: Computes the kernel size for each pooling operation based on the input outputSize, performs 3D max pooling on the input self, and outputs the pooled values outputOut and indices indicesOut. The difference between aclnnAdaptiveMaxPool3d and aclnnMaxPool3d is that you only need to specify the outputSize, and the pooling regions are divided according to the outputSize.

- Formula:
  N (Batch) represents the batch size, C (Channels) represents the feature map channels, D (Depth) represents the feature map depth, H (Height) represents the feature map height, and W (Width) represents the feature map width.
  For the scenario where the input self has dimensions $[N,C,D,H,W]$ and the outputSize value is $[D_o,H_o,W_o]$, the output dimensions are $[N,C,D_o,H_o,W_o]$, the indices dimensions are $[N,C,D_o,H_o,W_o]$, and the calculation formula for each element $(l,m,n)$ in the corresponding tensor is as follows:

  $$
  D^{l}_{left} = \lfloor(l*D)/D_o\rfloor
  $$

  $$
  D^{l}_{right} = \lceil((l+1)*D)/D_o\rceil
  $$

  $$
  H^{m}_{left} = \lfloor(m*H)/H_o\rfloor
  $$

  $$
  H^{m}_{right} = \lceil((m+1)*H)/H_o\rceil
  $$

  $$
  W^{n}_{left} = \lfloor(n*W)/W_o\rfloor
  $$

  $$
  W^{n}_{right} = \lceil((n+1)*W)/W_o\rceil
  $$

  $$
  output(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{max} input(N,C,i,j,k)
  $$

  $$
  indices(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{argmax} input(N,C,i,j,k)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnAdaptiveMaxPool3dGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the computation process of the operator. Then, call the "aclnnAdaptiveMaxPool3d" API to perform the computation.

```Cpp
aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *outputSize,
  aclTensor         *outputOut,
  aclTensor         *indicesOut,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAdaptiveMaxPool3d(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaptiveMaxPool3dGetWorkspaceSize

- **Parameters:**

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
        <td>Tensor to be computed.</td>
        <td>The product D*H*W of the three dimensions D, H, and W cannot exceed the maximum value of int32, and the data type must be the same as that of outputOut.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>NCHW, NCDHW</td>
        <td>4-5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputSize</td>
        <td>Input</td>
        <td>Output window size.</td>
        <td>Represents the spatial size of the output result in the Dout, Hout, and Wout dimensions.</td>
        <td>-</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputOut</td>
        <td>Output</td>
        <td>Pooling result.</td>
        <td>Same data type as self, and the shape is the same as indicesOut.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>NCHW, NCDHW</td>
        <td>4-5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indicesOut</td>
        <td>Output</td>
        <td>Indices of outputOut elements in the input self.</td>
        <td>The shape is the same as that of outputOut.</td>
        <td>INT32, INT64</td>
        <td>NCHW, NCDHW</td>
        <td>4-5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type of the parameter <code>indicesOut</code> does not support INT64,

  <!-- end id7 -->

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
      <td>self, outputSize, outputOut, or indicesOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type, shape, format, or parameter value of self, outputOut, or indicesOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape or parameter value of outputSize is not within the supported range.</td>
    </tr>
    <tr>
      <td>self and outputOut have inconsistent data types.</td>
    </tr>
    <tr>
      <td>outputOut and indicesOut have inconsistent shapes.</td>
    </tr>
    <tr>
      <td>The platform is not supported.</td>
    </tr>
    <tr>
      <td>depth * height * width > max int32, which exceeds the representable range of indices.</td>
    </tr>
  </tbody>
  </table>

## aclnnAdaptiveMaxPool3d

- **Parameters:**

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
        <td>Workspace size allocated on the device, obtained by the first-phase API aclnnAdaptiveMaxPool3dGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, which contains the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnAdaptiveMaxPool3d defaults to a deterministic implementation.

- Shape description:
  - self.shape = (N, C, Din, Hin, Win)
  - outputSize = [Dout, Hout, Wout]
  - outputOut.shape = (N, C, Dout, Hout, Wout)
  - indicesOut.shape = (N, C, Dout, Hout, Wout)

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_max_pool3d.h"

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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {1, 1, 1, 4, 4};
  std::vector<int64_t> outShape = {1, 1, 1, 2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* indDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4.1, 5, 6, 7,
                                     8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData = {0, 0, 0, 0.0};
  std::vector<int32_t> indicesHostData = {0, 0, 0, 0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, outShape, &indDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize = {1, 2, 2};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize.data(), arraySize.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAdaptiveMaxPool3d.
  ret = aclnnAdaptiveMaxPool3dGetWorkspaceSize(self, outputSize, out, indices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAdaptiveMaxPool3d.
  ret = aclnnAdaptiveMaxPool3d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool3d failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values. Copy the result from the device memory to the host. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  std::vector<int32_t> indicesData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indDeviceAddr,
                    size * sizeof(indicesData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release the aclTensor. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);

  // 7. Release Device resources. Modify it according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(indDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
