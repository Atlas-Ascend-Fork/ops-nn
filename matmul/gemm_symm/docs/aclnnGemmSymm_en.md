# aclnnGemmSymm

## Applicable Products

<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id2 -->
<!-- npu="310b" id3 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Not supported
<!-- end id5 -->

## Function

- Description: Computes α multiplied by the product of A and B, then adds the product of β and C, and writes the result to an independent output Y. The caller guarantees that A @ B and C are symmetric. The kernel writes the full square matrix: a diagonal tile is stored once, and an off-diagonal tile stores both (m, n) and (n, m) from the same product.
- Formula (transposeA and transposeB are both false; a is (…, m, k) and b is (…, k, m), rank 2 or rank 3 with one batch axis):

  <div>
  Y = α × (A @ B) + β × C
  </div>

  When transposeA is true, a is stored as (…, k, m). When transposeB is true, b is stored as (…, m, k). The transpose flags change only the input layout. The output is always (…, m, m).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call aclnnGemmSymmGetWorkspaceSize first to obtain the workspace size and the executor that contains the computation flow, and then call aclnnGemmSymm to run the computation.

```cpp
aclnnStatus aclnnGemmSymmGetWorkspaceSize(
  const aclTensor *a,
  const aclTensor *b,
  const aclTensor *c,
  aclTensor       *y,
  const aclScalar *alphaOptional,
  const aclScalar *betaOptional,
  bool            transposeA,
  bool            transposeB,
  const char      *fillMode,
  int8_t          cubeMathType,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGemmSymm(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnGemmSymmGetWorkspaceSize

- **Parameters**:
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 301px">
  <col style="width: 331px">
  <col style="width: 237px">
  <col style="width: 111px">
  <col style="width: 111px">
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
      <td>a</td>
      <td>Input</td>
      <td>Left matrix in the formula, A.</td>
      <td><ul><li>FLOAT16 or BFLOAT16. a, b, c, and y must use the same type.</li>
      <li>Shape is (…, m, k) when transposeA is false, and (…, k, m) when transposeA is true.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2–3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>Right matrix in the formula, B.</td>
      <td><ul><li>The data type and rank match a.</li>
      <li>Shape is (…, k, m) when transposeB is false, and (…, m, k) when transposeB is true.</li>
      <li>For rank 3, the batch axis must match a.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2–3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>c</td>
      <td>Input</td>
      <td>Residual matrix in the formula, C. Optional.</td>
      <td><ul><li>Required when β is not 0. Shape is (…, m, m), and the batch axis matches a.</li>
      <li>May be nullptr when β is 0. C is not read in that case.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2–3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Result in the formula, Y.</td>
      <td>Shape is (…, m, m). The rank and batch axis match a. The result is written to y and does not update c in place.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2–3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alphaOptional(α)</td>
      <td>Input</td>
      <td>α in the formula.</td>
      <td>Defaults to 1.0 when nullptr. When α is 0, the matmul is skipped and the output is β × C.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>betaOptional(β)</td>
      <td>Input</td>
      <td>β in the formula.</td>
      <td>Defaults to 1.0 when nullptr. c is required in that case.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeA</td>
      <td>Input</td>
      <td>Whether to interpret a as transposed storage.</td>
      <td>When true, a is stored as (…, k, m).</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeB</td>
      <td>Input</td>
      <td>Whether to interpret b as transposed storage.</td>
      <td>When true, b is stored as (…, m, k).</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fillMode</td>
      <td>Input</td>
      <td>Output region mode.</td>
      <td>Pass nullptr or "full". Only the full symmetric matrix is written. Any other string returns a parameter error.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Selects the accumulator precision after the cube.</td>
      <td>4 keeps the fp32 accumulator through Fixpipe and casts to the input type after the vector add. It takes effect only when C is passed and beta is not 0. With no x3 it matches 0, 1, 2, and 3, which cast to fp16 or bf16 in Fixpipe. Any other value returns a parameter error.</td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to allocate on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the operator executor, which contains the computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**:

  aclnnStatus: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API validates the inputs and reports an error in the following cases:
  <table style="undefined;table-layout: fixed; width: 809px"><colgroup>
  <col style="width: 257px">
  <col style="width: 121px">
  <col style="width: 431px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>a, b, or y is a null pointer.</td>
    </tr>
    <tr>
      <td>c is a null pointer when β is not 0.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type is not FLOAT16 or BFLOAT16, a, b, c, and y do not share one data type, or the data format is not supported.</td>
    </tr>
    <tr>
      <td>The rank is not 2 or 3, or the ranks of a, b, y, and a provided c do not match.</td>
    </tr>
    <tr>
      <td>m is not greater than 0, k is less than 0, or the shape of b or y does not match a.</td>
    </tr>
    <tr>
      <td>fillMode is not a null pointer and is not "full".</td>
    </tr>
    <tr>
      <td>cubeMathType is outside 0 to 4.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current product is not Ascend 950.</td>
    </tr>
  </tbody>
  </table>

## aclnnGemmSymm

- **Parameters**:

  <div style="overflow-x: auto;">
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
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
      <td>Address of the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained from aclnnGemmSymmGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, which contains the computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which the task runs.</td>
    </tr>
  </tbody>
  </table>
  </div>

- **Return value**:

  aclnnStatus: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism:

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnGemmSymm is deterministic by default. Each output element is written from a single cube product. A diagonal tile is stored once, and both stores of an off-diagonal tile use the same product.
  <!-- end id11 -->

- FLOAT16 and BFLOAT16 are supported. The format is ND and the rank is 2 or 3. a, b, c, and y must use the same data type in one call. For rank 3, the leading dimension is the batch, and the batch of a, b, c, and y must match.
- m must be greater than 0. k may be 0. When k is 0 or α is 0, the matmul is skipped and the output is β × C. If β is also 0, the output is all zeros.
- c is required when β is not 0. The caller must guarantee that A @ B is symmetric. When β is not 0, symmetry of the output also depends on C being symmetric.
- y is an independent output. c is not updated in place.
- fillMode accepts only a null pointer or "full".

## Sample

The following sample is for reference only. For the compilation and running process, see [Compilation and Running Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gemm_symm.h"

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
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

float Bf16ToFloat(uint16_t h) {
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Y = 1 * (A @ B). A is (4, 2) of ones, B is (2, 4) of ones, beta is 0, and c is omitted. The result is all twos.
  std::vector<int64_t> aShape = {4, 2};
  std::vector<int64_t> bShape = {2, 4};
  std::vector<int64_t> yShape = {4, 4};
  void* aDeviceAddr = nullptr;
  void* bDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* a = nullptr;
  aclTensor* b = nullptr;
  aclTensor* y = nullptr;
  std::vector<uint16_t> aHostData(GetShapeSize(aShape), 0x3f80); // bf16 1.0
  std::vector<uint16_t> bHostData(GetShapeSize(bShape), 0x3f80);
  std::vector<uint16_t> yHostData(GetShapeSize(yShape), 0);
  ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_BF16, &a);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_BF16, &b);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  float alphaValue = 1.0f;
  float betaValue = 0.0f;
  aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  aclScalar* beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr && beta != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnGemmSymmGetWorkspaceSize(a, b, nullptr, y, alpha, beta, false, false, "full", 0, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSymmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnGemmSymm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSymm failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(yShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    float diff = std::fabs(Bf16ToFloat(resultData[i]) - 2.0f);
    CHECK_RET(diff < 1e-2f, LOG_PRINT("result[%ld] is: %f, expected 2.0\n", i, Bf16ToFloat(resultData[i]));
              return ACL_ERROR_FAILURE);
  }
  LOG_PRINT("aclnnGemmSymm result check: PASS\n");

  aclDestroyTensor(a);
  aclDestroyTensor(b);
  aclDestroyTensor(y);
  aclDestroyScalar(alpha);
  aclDestroyScalar(beta);
  aclrtFree(aDeviceAddr);
  aclrtFree(bDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
