# aclnnFusedSgd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:49:32.969Z -->

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

- Description: Implements the fusedSgd operator. Fuses multiple fine-grained operations (such as gradient scaling, weight decay, momentum update, and parameter assignment) originally scattered in the traditional SGD update process into a single NPU Kernel for execution.
- Formula:

  $$
  \begin{aligned}
  &\tilde{g}_t = \begin{cases}
  g_t / s & s \neq \text{None} \\
  g_t & \text{otherwise}
  \end{cases} \\

  &\hat{g}_t = \begin{cases}
  -\tilde{g}_t & \text{maximize} \\
  \tilde{g}_t & \text{otherwise}
  \end{cases} \\

  &\bar{g}_t = \hat{g}_t + weightDecay \cdot \theta_t \\

  &v_{t+1} = \begin{cases}
  \bar{g}_t & \text{first step} \\
  \mu v_t + (1-dampening)\bar{g}_t & \text{otherwise}
  \end{cases} \\

  &g_t^{\text{final}} = \begin{cases}
  \bar{g}_t + \mu v_{t+1} & \text{nesterov} \\
  v_{t+1} & v_t \neq \text{None} \\
  \bar{g}_t & \text{otherwise}
  \end{cases} \\

  &\theta_{t+1} = \theta_t - lr \cdot g_t^{\text{final}} \\
  &g_{t+1} = \tilde{g}_t

  \end{aligned}
  $$

## Function Prototype

The operator execution API is the [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFusedSgdGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnFusedSgd" API to perform the computation.

```cpp
aclnnStatus aclnnFusedSgdGetWorkspaceSize(
    const aclTensorList   *paramsRef,
    const aclTensorList   *gradsRef,
    const aclTensorList   *momentumBufferListOptionalRef,
    const aclTensor       *gradScaleOptional,
    float                  weightDecay,
    float                  momentum,
    float                  lr,
    float                  dampening,
    bool                   nesterov,
    bool                   maximize,
    bool                   isFirstStep,
    uint64_t              *workspaceSize,
    aclOpExecutor        **executor);
```

```cpp
aclnnStatus aclnnFusedSgd(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnFusedSgdGetWorkspaceSize

- **Parameters**

  <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1497px"><colgroup>
     <col style="width: 150px">
     <col style="width: 120px">
     <col style="width: 300px">
     <col style="width: 330px">
     <col style="width: 212px">
     <col style="width: 100px">
     <col style="width: 140px">
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
      <td>paramsRef</td>
      <td>Input/Output</td>
      <td>Parameters to update.</td>
      <td>Empty tensors are not supported.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>8 dimensions or fewer</td>
      <td>√</td>
      </tr>
      <tr>
      <td>gradsRef</td>
      <td>Input/Output</td>
      <td>Updates the gradient.</td>
      <td>Empty tensors are not supported.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>8 dimensions or fewer</td>
      <td>√</td>
      </tr>
      <tr>
      <td>momentumBufferListOptionalRef</td>
      <td>Input/Output</td>
      <td>Updates the momentum.</td>
      <td>Empty tensors are supported.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>Less than or equal to 8 dimensions</td>
      <td>√</td>
      </tr>
      <tr>
      <td>gradScaleOptional</td>
      <td>Input</td>
      <td>Gradient scaling value.</td>
      <td>Supports empty tensors.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>shape equals [1].</td>
      <td>√</td>
      </tr>
      <tr>
      <td>weightDecay</td>
      <td>Attribute</td>
      <td>Weight decay value.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>momentum</td>
      <td>Attribute</td>
      <td>Momentum value.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>lr</td>
      <td>Attribute</td>
      <td>Learning rate.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>dampening</td>
      <td>Attribute</td>
      <td>Dampening coefficient of momentum.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>nesterov</td>
      <td>Attribute</td>
      <td>Whether to enable Nesterov momentum.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>maximize</td>
      <td>Attribute</td>
      <td>Whether to maximize the objective function.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>isFirstStep</td>
      <td>Attribute</td>
      <td>Whether the update is the first step.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to request on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
      </tr>
    </tbody></table>
    </div>
- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 250px">
    <col style="width: 100px">
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
      <td>The input paramsRef, gradsRef, momentumBufferListOptionalRef, or grad_scale is a null pointer.</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><ul><li>The data type of paramsRef, gradsRef, momentumBufferListOptionalRef, or grad_scale is not within the supported range.</li>
      <li>The data types of paramsRef, gradsRef, momentumBufferListOptionalRef, and the tensors in them are different.</li>
      <li>The shape dimension of tensors in paramsRef, gradsRef, and momentumBufferListOptionalRef is greater than 8.</li>
      <li>Tensors at the same index in paramsRef, gradsRef, and momentumBufferListOptionalRef (when momentumBufferListOptionalRef is not null) have different shapes.</li>
      <li>paramsRef, gradsRef, or momentumBufferListOptionalRef (when momentumBufferListOptionalRef is not null) contains a null pointer.</li>
      <li>The value of lr, momentum, weightDecay, or dampening is less than 0.</li>
      </ul></td>
      </tr>
  </tbody>
    </table>
    </div>

## aclnnFusedSgd

* **Parameters**:

<div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 650px">
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
      <td>Workspace memory address requested on the Device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the device. It is obtained by calling the first-phase API aclnnFusedSgd.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor. It contains the computation flow of the operator.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
        <td>Stream for task execution.</td>
      </tr>
    </tbody></table>
    </div>

* **Return value**:
  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. The data types of paramsRef, gradsRef, momentumBufferListOptionalRef, and grad_scale are in the supported range.
2. paramsRef, gradsRef, momentumBufferListOptionalRef, and each tensor in them have the same data type.
3. The shape dimension of each tensor in paramsRef, gradsRef, and momentumBufferListOptionalRef is less than or equal to 8.
4. Tensors at the same index in paramsRef, gradsRef, and momentumBufferListOptionalRef (when momentumBufferListOptionalRef is not empty) have the same shape.
5. paramsRef, gradsRef, and momentumBufferListOptionalRef (when momentumBufferListOptionalRef is not empty) must not contain null pointers.
6. The values of lr, momentum, weightDecay, and dampening are greater than or equal to 0.
7. gradScale is not equal to 0.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_sgd.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  std::vector<float> paramsRefHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> gradsRefHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<float> momentumHostData1 = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> paramsRefHostData2 = {9, 10, 11, 12};
  std::vector<float> gradsRefHostData2 = {0.9, 1.0, 1.1, 1.2};
  std::vector<float> momentumHostData2 = {0, 0, 0, 0};
  std::vector<float> gradScaleOptionalHostData = {1.0};
  std::vector<int64_t> inputShape1 = {2, 2, 2};
  std::vector<int64_t> inputShape2 = {2, 2};
  std::vector<int64_t> scalarShape = {1};

  void *paramsRef1DeviceAddr = nullptr;
  void *gradsRef1DeviceAddr = nullptr;
  void *momentum1DeviceAddr = nullptr;
  void *paramsRef2DeviceAddr = nullptr;
  void *gradsRef2DeviceAddr = nullptr;
  void *momentum2DeviceAddr = nullptr;
  void *gradScaleOptionalDeviceAddr = nullptr;

  aclTensor *paramsRef1 = nullptr;
  aclTensor *gradsRef1 = nullptr;
  aclTensor *momentum1 = nullptr;
  aclTensor *paramsRef2 = nullptr;
  aclTensor *gradsRef2 = nullptr;
  aclTensor *momentum2 = nullptr;
  aclTensor *gradScaleOptional = nullptr;

  ret = CreateAclTensor(paramsRefHostData1, inputShape1, &paramsRef1DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradsRefHostData1, inputShape1, &gradsRef1DeviceAddr, aclDataType::ACL_FLOAT, &gradsRef1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(momentumHostData1, inputShape1, &momentum1DeviceAddr, aclDataType::ACL_FLOAT, &momentum1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(paramsRefHostData2, inputShape2, &paramsRef2DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradsRefHostData2, inputShape2, &gradsRef2DeviceAddr, aclDataType::ACL_FLOAT, &gradsRef2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(momentumHostData2, inputShape2, &momentum2DeviceAddr, aclDataType::ACL_FLOAT, &momentum2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradScaleOptionalHostData, scalarShape, &gradScaleOptionalDeviceAddr, aclDataType::ACL_FLOAT, &gradScaleOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> paramsRefListData = {paramsRef1, paramsRef2};
  std::vector<aclTensor*> gradsRefListData = {gradsRef1, gradsRef2};
  std::vector<aclTensor*> momentumListData = {momentum1, momentum2};
  aclTensorList* paramsRefList = aclCreateTensorList(paramsRefListData.data(), paramsRefListData.size());
  aclTensorList* gradsRefList = aclCreateTensorList(gradsRefListData.data(), gradsRefListData.size());
  aclTensorList* momentumList = aclCreateTensorList(momentumListData.data(), momentumListData.size());

  float weightDecay = 0.01f;
  float momentumVal = 0.9f;
  float lr = 0.001f;
  float dampening = 0.0f;
  bool nesterov = false;
  bool maximize = false;
  bool isFirstStep = true;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  ret = aclnnFusedSgdGetWorkspaceSize(paramsRefList, gradsRefList, momentumList, gradScaleOptional,
                                       weightDecay, momentumVal, lr, dampening,
                                       nesterov, maximize, isFirstStep,
                                       &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnFusedSgdGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  ret = aclnnFusedSgd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnFusedSgd failed. ERROR: %d\n", ret);
            return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  LOG_PRINT("====== Tensor 1 paramsRef results ======\n");
  PrintOutResult(inputShape1, &paramsRef1DeviceAddr);
  LOG_PRINT("====== Tensor 1 gradsRef results ======\n");
  PrintOutResult(inputShape1, &gradsRef1DeviceAddr);
  LOG_PRINT("------ Momentum buffer 1 ------\n");
  PrintOutResult(inputShape1, &momentum1DeviceAddr);
  LOG_PRINT("====== Tensor 2 paramsRef results ======\n");
  PrintOutResult(inputShape2, &paramsRef2DeviceAddr);
  LOG_PRINT("====== Tensor 2 gradsRef results ======\n");
  PrintOutResult(inputShape2, &gradsRef2DeviceAddr);
  LOG_PRINT("------ Momentum buffer 2 ------\n");
  PrintOutResult(inputShape2, &momentum2DeviceAddr);

  aclDestroyTensorList(paramsRefList);
  aclDestroyTensorList(gradsRefList);
  aclDestroyTensorList(momentumList);
  aclDestroyTensor(gradScaleOptional);

  aclrtFree(paramsRef1DeviceAddr);
  aclrtFree(gradsRef1DeviceAddr);
  aclrtFree(momentum1DeviceAddr);
  aclrtFree(paramsRef2DeviceAddr);
  aclrtFree(gradsRef2DeviceAddr);
  aclrtFree(momentum2DeviceAddr);
  aclrtFree(gradScaleOptionalDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  ret = aclrtDestroyStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("destroy stream failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtResetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("reset device failed. ERROR: %d\n", ret);
            return ret);
  ret = aclFinalize();
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("finalize acl failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

```
