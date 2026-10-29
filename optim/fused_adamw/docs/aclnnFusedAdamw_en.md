# aclnnFusedAdamw

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:48:48.389Z -->

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

- Description: Implements the fused AdamW optimizer. Multiple parameter lists (TensorList) can be updated in a single call, including parameter update, momentum update, and other operations.

- Formula:

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t}
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}
  $$

  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}
  $$

  $$
  if(amsgrad) : maxGradNorm = max(maxGradNorm,\hat{v}_{t})
  $$

  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnFusedAdamwGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnFusedAdamw" API to perform the computation.

```cpp
aclnnStatus aclnnFusedAdamwGetWorkspaceSize(
    const aclTensorList* paramsRef,
    const aclTensorList* grads,
    const aclTensorList* expAvgsRef,
    const aclTensorList* expAvgSqsRef,
    const aclTensorList* maxExpAvgSqsRef,
    const aclTensorList* stateSteps,
    const aclTensor*     gradScaleOptional,
    const aclTensor*     foundInfOptional,
    double                lr,
    double                beta1,
    double                beta2,
    double                weightDecay,
    double                eps,
    bool                 amsgrad,
    bool                 maximize,
    uint64_t*            workspaceSize,
    aclOpExecutor**      executor)
```

```cpp
aclnnStatus aclnnFusedAdamw(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnFusedAdamwGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1520px"><colgroup>
    <col style="width: 230px">
    <col style="width: 120px">
    <col style="width: 330px">
    <col style="width: 220px">
    <col style="width: 230px">
    <col style="width: 115px">
    <col style="width: 130px">
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
        <td>paramsRef (aclTensorList*)</td>
        <td>Input/Output</td>
        <td><ul><li>Does not support null Tensor.</li><li>List of weights to be computed, θ in the formula.</li></ul></td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>grads (aclTensorList*)</td>
        <td>Input</td>
        <td><ul><li>Does not support null Tensor.</li><li>List of gradient data, g<sub>t</sub> in the formula.</li></ul></td>
        <td>-</td>
        <td>Consistent with the "paramsRef" parameter.</td>
        <td>ND</td>
        <td>Consistent with the "paramsRef" parameter.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expAvgsRef (aclTensorList*)</td>
        <td>Input/Output</td>
        <td><ul><li>Does not support null tensors.</li><li>List of first moments, m in the formula.</li></ul></td>
        <td>-</td>
        <td>Consistent with the "paramsRef" parameter.</td>
        <td>ND</td>
        <td>Consistent with the "paramsRef" parameter.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expAvgSqsRef (aclTensorList*)</td>
        <td>Input/Output</td>
        <td><ul><li>Does not support null Tensor.</li><li>List of second-order momentums, corresponding to v in the formula. The value cannot be negative.</li></ul></td>
        <td>-</td>
        <td>Consistent with the "paramsRef" parameter.</td>
        <td>ND</td>
        <td>Consistent with the "paramsRef" parameter.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxExpAvgSqsRef (aclTensorList*)</td>
        <td>Input/Output</td>
        <td><ul><li>Does not support null Tensor.</li><li>Stores the list of maximum second moments and outputs the maximum value after comparing it with the updated expAvgSqsRef.</li></ul></td>
        <td>This parameter is required when amsgrad is true and optional when amsgrad is false.</td>
        <td>Consistent with the "paramsRef" parameter.</td>
        <td>ND</td>
        <td>Consistent with the "paramsRef" parameter.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>stateSteps (aclTensorList*)</td>
        <td>Input</td>
        <td>List of iteration counts, corresponding to t in the formula. The value must be greater than 0.</td>
        <td>Each Tensor contains one element.</td>
        <td>INT64, FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>x</td>
      </tr>
      <tr>
        <td>gradScaleOptional (aclTensor*)</td>
        <td>Input</td>
        <td>Optional input. Gradient scaling factor.</td>
        <td>When foundInfOptional=1, the gradient is cleared to zero and the parameter update stops.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>-</td>
        <td>x</td>
      </tr>
      <tr>
        <td>foundInfOptional (aclTensor*)</td>
        <td>Input</td>
        <td>Optional input that indicates whether Inf/NaN occurs.</td>
        <td>Update stops when the value is 1.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>-</td>
        <td>x</td>
      </tr>
      <tr>
        <td>lr (double)</td>
        <td>Input</td>
        <td>Learning rate, η in the formula.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta1 (double)</td>
        <td>Input</td>
        <td>β<sub>1</sub> parameter.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta2 (double)</td>
        <td>Input</td>
        <td>β<sub>2</sub> parameter.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightDecay (double)</td>
        <td>Input</td>
        <td>Weight decay coefficient, λ in the formula.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps (double)</td>
        <td>Input</td>
        <td>Prevents division by zero.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>amsgrad (bool)</td>
        <td>Input</td>
        <td>Whether to use the AMSGrad variable of the algorithm.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maximize (bool)</td>
        <td>Input</td>
        <td>Whether to maximize the parameters.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied on the Device side.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>When the passed paramsRef, grads, expAvgsRef, expAvgSqsRef, and stateSteps are null pointers.</td>
      </tr>
      <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>When the data types of paramsRef, grads, expAvgsRef, expAvgSqsRef, and maxExpAvgSqsRef are not within the supported range.</td>
      </tr>
      <tr>
      <td>When the data types of paramsRef, grads, expAvgsRef, expAvgSqsRef, and maxExpAvgSqsRef are different.</td>
      </tr>
      <tr>
      <td>When the shapes of grads, expAvgsRef, expAvgSqsRef, and paramsRef are inconsistent.</td>
      </tr>
    </tbody></table>

## aclnnFusedAdamw

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
  <col style="width: 200px">
  <col style="width: 162px">
  <col style="width: 882px">
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
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnFusedAdamwGetWorkspaceSize.</td>
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
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The data types of paramsRef, grads, expAvgsRef, expAvgSqsRef, and maxExpAvgSqsRef in the input tensors must be the same. The supported data types are FLOAT16, BFLOAT16, and FLOAT32.
- The shapes at corresponding positions of paramsRef, grads, expAvgsRef, expAvgSqsRef, and maxExpAvgSqsRef in the input tensors must be the same.
- stateSteps supports INT64 and FLOAT32, and each Tensor contains 1 element.
- When amsgrad is false, maxExpAvgSqsRef can be null; when amsgrad is true, maxExpAvgSqsRef is required and its shape must be the same as that of paramsRef.
- Deterministic computation:
  - aclnnFusedAdamw defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_adamw.h"
#include <iostream>
#include <vector>

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return );
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<float> paramsRefHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> gradsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> expavgsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> expavgsqsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> maxexpavgsqsHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> stepsHostData1 = {3};


    std::vector<float> gradScaleOptionalHostData = {2};

    std::vector<float> paramsRefHostData2 = {9, 10, 11, 12};
    std::vector<float> gradsHostData2 = {0.9, 1.0, 1.1, 1.2};
    std::vector<float> expavgsHostData2 = {-1, -2, -3, -4};
    std::vector<float> expavgsqsHostData2 = {1, 2, 3, 4};
    std::vector<float> maxexpavgsqsHostData2 = {2, 2, 2, 2};
    std::vector<float> stepsHostData2 = {4};


    std::vector<int64_t> inputShape1 = {2, 2, 2};
    std::vector<int64_t> inputShape2 = {2, 2};
    std::vector<int64_t> scalarShape = {1};

    void* paramsRef1DeviceAddr = nullptr;
    void* grads1DeviceAddr = nullptr;
    void* expavgs1DeviceAddr = nullptr;
    void* expavgsqs1DeviceAddr = nullptr;
    void* maxexpavgsqs1DeviceAddr = nullptr;
    void* steps1DeviceAddr = nullptr;

    void* paramsRef2DeviceAddr = nullptr;
    void* grads2DeviceAddr = nullptr;
    void* expavgs2DeviceAddr = nullptr;
    void* expavgsqs2DeviceAddr = nullptr;
    void* maxexpavgsqs2DeviceAddr = nullptr;
    void* steps2DeviceAddr = nullptr;

    void* gradScaleOptionalDeviceAddr = nullptr;

    aclTensor* paramsRef1 = nullptr;
    aclTensor* grads1 = nullptr;
    aclTensor* expavgs1 = nullptr;
    aclTensor* expavgsqs1 = nullptr;
    aclTensor* maxexpavgsqs1 = nullptr;
    aclTensor* steps1 = nullptr;

    aclTensor* paramsRef2 = nullptr;
    aclTensor* grads2 = nullptr;
    aclTensor* expavgs2 = nullptr;
    aclTensor* expavgsqs2 = nullptr;
    aclTensor* maxexpavgsqs2 = nullptr;
    aclTensor* steps2 = nullptr;

    aclTensor* gradScaleOptional = nullptr;

    aclTensor* foundInfOptional = nullptr;

    ret = CreateAclTensor(paramsRefHostData1, inputShape1, &paramsRef1DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradsHostData1, inputShape1, &grads1DeviceAddr, aclDataType::ACL_FLOAT, &grads1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsHostData1, inputShape1, &expavgs1DeviceAddr, aclDataType::ACL_FLOAT, &expavgs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsqsHostData1, inputShape1, &expavgsqs1DeviceAddr, aclDataType::ACL_FLOAT, &expavgsqs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxexpavgsqsHostData1, inputShape1, &maxexpavgsqs1DeviceAddr, aclDataType::ACL_FLOAT, &maxexpavgsqs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stepsHostData1, inputShape1, &steps1DeviceAddr, aclDataType::ACL_FLOAT, &steps1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(paramsRefHostData2, inputShape2, &paramsRef2DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradsHostData2, inputShape2, &grads2DeviceAddr, aclDataType::ACL_FLOAT, &grads2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsHostData2, inputShape2, &expavgs2DeviceAddr, aclDataType::ACL_FLOAT, &expavgs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsqsHostData2, inputShape2, &expavgsqs2DeviceAddr, aclDataType::ACL_FLOAT, &expavgsqs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxexpavgsqsHostData2, inputShape2, &maxexpavgsqs2DeviceAddr, aclDataType::ACL_FLOAT, &maxexpavgsqs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stepsHostData2, inputShape2, &steps2DeviceAddr, aclDataType::ACL_FLOAT, &steps2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(gradScaleOptionalHostData, scalarShape, &gradScaleOptionalDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradScaleOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<aclTensor*> paramsRefListData = {paramsRef1, paramsRef2};
    std::vector<aclTensor*> gradsListData = {grads1, grads2};
    std::vector<aclTensor*> expavgsListData = {expavgs1, expavgs2};
    std::vector<aclTensor*> expavgsqsListData = {expavgsqs1, expavgsqs2};
    std::vector<aclTensor*> maxexpavgsqsData = {maxexpavgsqs1, maxexpavgsqs2};
    std::vector<aclTensor*> stepsListData = {steps1, steps2};
    aclTensorList* paramsRefList = aclCreateTensorList(paramsRefListData.data(), paramsRefListData.size());
    aclTensorList* gradsList = aclCreateTensorList(gradsListData.data(), gradsListData.size());
    aclTensorList* expavgsList = aclCreateTensorList(expavgsListData.data(), expavgsListData.size());
    aclTensorList* expavgsqsList = aclCreateTensorList(expavgsqsListData.data(), expavgsqsListData.size());
    aclTensorList* maxexpavgsqsList = aclCreateTensorList(maxexpavgsqsData.data(), maxexpavgsqsData.size());
    aclTensorList* stepsList = aclCreateTensorList(stepsListData.data(), stepsListData.size());

    double lr = 0.001f;
    double beta1 = 0.9f;
    double beta2 = 0.999f;
    double weightDecay = 0.0f;
    double eps = 1e-8;
    bool amsgrad = true;
    bool maximize = false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnFusedAdamwGetWorkspaceSize(paramsRefList, gradsList, expavgsList, expavgsqsList, maxexpavgsqsList,
                                        stepsList, gradScaleOptional, foundInfOptional, lr, beta1, beta2, weightDecay,
                                        eps, amsgrad, maximize, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedAdamwGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnFusedAdamw(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedAdamw failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("====== Tensor 1 paramsRef1 results ======\n");
    PrintOutResult(inputShape1, &paramsRef1DeviceAddr);
    LOG_PRINT("====== Tensor 2 expavgs1 results ======\n");
    PrintOutResult(inputShape1, &expavgs1DeviceAddr);
    LOG_PRINT("====== Tensor 3 expavgsqs1 results ======\n");
    PrintOutResult(inputShape1, &expavgsqs1DeviceAddr);
    LOG_PRINT("===== Tensor 4 maxexpavgsqs1 results ======\n");
    PrintOutResult(inputShape1, &maxexpavgsqs1DeviceAddr);

    LOG_PRINT("====== Tensor 5 paramsRef results ======\n");
    PrintOutResult(inputShape2, &paramsRef2DeviceAddr);
    LOG_PRINT("====== Tensor 6 expavgs2 results ======\n");
    PrintOutResult(inputShape2, &expavgs2DeviceAddr);
    LOG_PRINT("====== Tensor 7 expavgsqs2 results ======\n");
    PrintOutResult(inputShape2, &expavgsqs2DeviceAddr);
    LOG_PRINT("====== Tensor 8 maxexpavgsqs2 results ======\n");
    PrintOutResult(inputShape2, &maxexpavgsqs2DeviceAddr);

    aclDestroyTensorList(paramsRefList);
    aclDestroyTensorList(gradsList);
    aclDestroyTensorList(expavgsList);
    aclDestroyTensorList(expavgsqsList);
    aclDestroyTensorList(maxexpavgsqsList);
    aclDestroyTensorList(stepsList);
    aclDestroyTensor(gradScaleOptional);
    aclDestroyTensor(foundInfOptional);

    aclrtFree(paramsRef1DeviceAddr);
    aclrtFree(grads1DeviceAddr);
    aclrtFree(expavgs1DeviceAddr);
    aclrtFree(expavgsqs1DeviceAddr);
    aclrtFree(maxexpavgsqs1DeviceAddr);
    aclrtFree(steps1DeviceAddr);

    aclrtFree(paramsRef2DeviceAddr);
    aclrtFree(grads2DeviceAddr);
    aclrtFree(expavgs2DeviceAddr);
    aclrtFree(expavgsqs2DeviceAddr);
    aclrtFree(maxexpavgsqs2DeviceAddr);
    aclrtFree(steps2DeviceAddr);

    aclrtFree(gradScaleOptionalDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    ret = aclrtDestroyStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("destroy stream failed. ERROR: %d\n", ret); return ret);
    ret = aclrtResetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("reset device failed. ERROR: %d\n", ret); return ret);
    ret = aclFinalize();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("finalize acl failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
