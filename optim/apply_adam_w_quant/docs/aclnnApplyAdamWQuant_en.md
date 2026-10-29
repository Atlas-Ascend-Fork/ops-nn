# aclnnApplyAdamWQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:44:18.796Z -->

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

- **Description:** Uses the optimizer inputs m and v as indices to look up values in their respective qmaps, multiplies them by the absmax corresponding to each blockSize for dequantization, and then implements the AdamW optimizer functionality. For the updated m and v, one maximum value is taken per blockSize, with each blockSize of m and v corresponding to one absmax. A norm normalization is performed, and binary search is used to find the indices in the qmaps corresponding to m and v as outputs. The absmax also serves as the input for the next round of quantization.

- **Optimizer Formula:**

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$

  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$

  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnApplyAdamWQuantGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call the "aclnnApplyAdamWQuant" API to perform the computation.

```Cpp
aclnnStatus aclnnApplyAdamWQuantGetWorkspaceSize(
    aclTensor       *varRef,
    const aclTensor *grad,
    aclTensor       *mRef,
    aclTensor       *vRef,
    const aclTensor *qmapM,
    const aclTensor *qmapV,
    aclTensor       *absmaxMRef,
    aclTensor       *absmaxVRef,
    const aclTensor *step,
    double          lr,
    double          beta1,
    double          beta2,
    double          weightDecay,
    double          eps,
    double          gnormScale,
    char *          quantModeOptional,
    int64_t         blockSize,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnApplyAdamWQuant(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnApplyAdamWQuantGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1498px"><colgroup>
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 380px">
  <col style="width: 230px">
  <col style="width: 138px">
  <col style="width: 115px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimensions (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">varRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Weight input to be updated, which is also the output, θ in the formula.</td>
      <td class="tg-0pky"></td>
      <td class="tg-0pky">FLOAT16, BFLOAT16, FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">grad (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input gradient, g in the formula.</td>
      <td class="tg-0pky">shape must match the input varRef.</td>
      <td class="tg-0pky">Must remain consistent with varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">mRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Index value of the m parameter in the AdamW optimizer formula before quantization. The specific value in qmapM is derived based on the index.</td>
      <td class="tg-0pky">The shape must match the input varRef.</td>
      <td class="tg-0pky">uint8_t</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">vRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Index value of the v parameter before quantization in the AdamW optimizer formula. The specific value in qmapV is derived based on the index.</td>
      <td class="tg-0pky">The shape must match the input varRef.</td>
      <td class="tg-0pky">uint8_t</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">qmapM (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Quantization mapping table in ascending order. Each m is selected based on the index value in mRef.</td>
      <td class="tg-0pky">The shape must be [256,].</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">qmapV (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The quantization mapping table is sorted in ascending order. Each v is selected based on the index in vRef.</td>
      <td class="tg-0pky">The shape must be [256,].</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">absmaxMRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Every blockSize (256) elements of mRef correspond to one maximum value, which is used to select values in qmapM by indexing with mRef and multiply them by the corresponding absmaxMRef to perform dequantization. Then, one maximum value is selected from every blockSize (256) elements of the updated mRef as the output of absmaxMRef.</td>
      <td class="tg-0pky">The shape must be "absmaxMRef.size = ceil(mRef.size/blockSize)".</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">absmaxVRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Each blockSize (256) vRef values correspond to one maximum value, used to perform dequantization by multiplying the value selected from qmapV using the index in vRef by the corresponding absmaxVRef. Then, one maximum value is selected from each blockSize (256) values of the updated vRef as the output of absmaxVRef.</td>
      <td class="tg-0pky">The shape must be "absmaxVRef.size = ceil(vRef.size/blockSize)".</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">step (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">t in the formula, the number of iterations, greater than 0.</td>
      <td class="tg-0pky">shape must be [1].</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">lr (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Learning rate, which is η in the formula. The recommended values are 1e-3, 1e-5, and 1e-8, with a range of 0-1.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1 (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta1 parameter in the AdamW optimizer formula. Recommended value: 0.9. Range: 0-1.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2 (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta2 parameter in the AdamW optimizer formula. Recommended value: 0.999. Range: 0-1.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightDecay (double)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight decay coefficient, the λ parameter in the AdamW optimizer formula. Recommended value: 0.999, range: 0-1.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">eps (double)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">The ϵ parameter in the AdamW optimizer formula, added to the denominator to prevent division by 0. The recommended value is 1e-8.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">DOUBLE</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">gnormScale (double)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Parameter used to scale the input parameter grad. The recommended value is 0.999, and the range is 0-1.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">DOUBLE</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">blockSize (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">The size of each block participating in the computation, fixed to 256. As described for absmax above, a maximum value is selected for each blockSize.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">quantModeOptional(char*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Reserved parameter, currently meaningless.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size that needs to be applied on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 270px">
  <col style="width: 130px">
  <col style="width: 750px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The input or output tensor is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky">161002</td>
      <td class="tg-0pky">The data types and data formats of the input and output tensors are not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyAdamWQuant

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnApplyAdamWQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  The shape of varRef meets the following constraints:

  - varRef.shape = grad.shape
  - varRef.shape = mRef.shape
  - varRef.shape = vRef.shape
  - ceil(varRef.size/blockSize) = absmaxMRef.size
  - ceil(varRef.size/blockSize) = absmaxVRef.size

  Deterministic computation:

  - aclnnApplyAdamWQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_w_quant.h"
#define FAILED 1

#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR] " fmt "\n", ##args)
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}


int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
aclDataType dataType, aclTensor** tensor) {
auto size = GetShapeSize(shape) * sizeof(T);
// Call aclrtMalloc to apply for memory on the device side.
auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
// Call aclrtMemcpy to copy data from the host side to the device side.
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

int main() {
// 1. Boilerplate: initialize the device/stream. For details, see the acl API manual.
// Set deviceId based on your actual device.
int32_t deviceId = 0;
aclrtStream stream;
auto ret = Init(deviceId, &stream);
// Handle the check as needed.
CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 2. Construct the input and output according to the API definition.
std::vector<int64_t> VarRefShape = {1,256};
std::vector<int64_t> GradShape = {1,256};
std::vector<int64_t> mRefShape = {1,256};
std::vector<int64_t> vRefShape = {1,256};
std::vector<int64_t> mapMShape = {256,};
std::vector<int64_t> mapVShape = {256,};
std::vector<int64_t> absmaxMRefShape = {1,};
std::vector<int64_t> absmaxVRefShape = {1,};
std::vector<int64_t> stepShape = {1,};

void *VarRefDeviceAddr = nullptr;
void *GradDeviceAddr = nullptr;
void *mRefDeviceAddr = nullptr;
void *vRefDeviceAddr = nullptr;
void *qmapMDeviceAddr = nullptr;
void *qmapVDeviceAddr = nullptr;
void *absmaxMRefDeviceAddr = nullptr;
void *absmaxVRefDeviceAddr = nullptr;
void *stepDeviceAddr = nullptr;

aclTensor *varRef = nullptr;
aclTensor *grad = nullptr;
aclTensor *mRef = nullptr;
aclTensor *vRef = nullptr;
aclTensor *qmapM = nullptr;
aclTensor *qmapV = nullptr;
aclTensor *absmaxMRef = nullptr;
aclTensor *absmaxVRef = nullptr;
aclTensor *step = nullptr;

std::vector<float> inputVarHostData(256);
std::vector<float> inputGradHostData(256);
std::vector<uint8_t> inputMHostData(256);
std::vector<uint8_t> inputVHostData(256);
std::vector<float> inputmapMHostData(256);
std::vector<float> inputmapVHostData(256);
std::vector<float> inputabsmaxMHostData = {5};
std::vector<float> inputabsmaxVHostData = {3};
std::vector<int64_t> inputstepHostData(1);

const float lr = 0.1;
const float beta1 = 0.1;
const float beta2 = 0.1;
const float weightDecay = 0.1;
const float eps = 0.01;
const float gnormScale = 0.1;
const int64_t blockSize = 256;
char* quantModeOptional = "BLOCKWISE";

// Create the gradOutput aclTensor.
ret = CreateAclTensor(inputVarHostData, VarRefShape, &VarRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputGradHostData, GradShape, &GradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputMHostData, mRefShape, &mRefDeviceAddr, aclDataType::ACL_UINT8, &mRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputVHostData, vRefShape, &vRefDeviceAddr, aclDataType::ACL_UINT8, &vRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputmapMHostData, mapMShape, &qmapMDeviceAddr, aclDataType::ACL_FLOAT, &qmapM);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputmapVHostData, mapVShape, &qmapVDeviceAddr, aclDataType::ACL_FLOAT, &qmapV);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputabsmaxMHostData, absmaxMRefShape, &absmaxMRefDeviceAddr, aclDataType::ACL_FLOAT, &absmaxMRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputabsmaxVHostData, absmaxVRefShape, &absmaxVRefDeviceAddr, aclDataType::ACL_FLOAT, &absmaxVRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputstepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

// 3. Call the CANN operator library API. Replace it with the specific API.
uint64_t workspaceSize = 0;
aclOpExecutor* executor;
// Call the first-phase API aclnnApplyAdamWQuantGetWorkspaceSize.
ret = aclnnApplyAdamWQuantGetWorkspaceSize(varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step, lr, beta1, beta2, weightDecay, eps, gnormScale, quantModeOptional, blockSize, &workspaceSize, &executor);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
// Allocate device memory using the workspaceSize computed by the first-phase API.
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
}
// Call the second-phase API aclnnApplyAdamWQuant.
ret = aclnnApplyAdamWQuant(workspaceAddr, workspaceSize, executor, stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWQuant failed. ERROR: %d\n", ret); return ret);
// 4. Boilerplate: wait for the task execution to finish.
ret = aclrtSynchronizeStream(stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

// 5. Obtain the output value and copy the result from device memory to the host.
auto size = GetShapeSize(VarRefShape);
std::vector<float> resultData(size, 0);
ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), VarRefDeviceAddr,
                size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
for (int64_t i = 0; i < size; i++) {
LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
// 6. Release aclTensor, which needs to be modified according to the specific API definition.
aclDestroyTensor(varRef);
aclDestroyTensor(grad);
aclDestroyTensor(mRef);
aclDestroyTensor(vRef);
aclDestroyTensor(qmapM);
aclDestroyTensor(qmapV);
aclDestroyTensor(absmaxMRef);
aclDestroyTensor(absmaxVRef);
aclDestroyTensor(step);

// 7. Release device resources, which needs to be modified according to the specific API definition.
aclrtFree(VarRefDeviceAddr);
aclrtFree(GradDeviceAddr);
aclrtFree(mRefDeviceAddr);
aclrtFree(vRefDeviceAddr);
aclrtFree(qmapMDeviceAddr);
aclrtFree(qmapVDeviceAddr);
aclrtFree(absmaxMRefDeviceAddr);
aclrtFree(absmaxVRefDeviceAddr);
aclrtFree(stepDeviceAddr);
if (workspaceSize > 0) {
aclrtFree(workspaceAddr);
}
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();
return 0;
}
```
