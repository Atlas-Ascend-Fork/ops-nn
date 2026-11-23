# aclnnGemmSymm

## 产品支持情况

<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="310b" id3 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id5 -->

## 功能说明

- 接口功能：计算 α 乘以 A 与 B 的乘积，再与 β 和 C 的乘积求和，结果写入独立输出 Y。调用方保证 A @ B 与 C 对称；kernel 按对称矩阵写出完整方阵，对角块只写一次，非对角块用同一次乘积同时写出 (m, n) 和 (n, m)。
- 计算公式（transposeA、transposeB 均为 false，a 为 (…, m, k)，b 为 (…, k, m)，2 维或带一个 batch 轴的 3 维）：

  <div>
  Y = α × (A @ B) + β × C
  </div>

  transposeA 为 true 时，a 以 (…, k, m) 存储；transposeB 为 true 时，b 以 (…, m, k) 存储。两种转置只改变读入布局，输出始终是 (…, m, m)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 aclnnGemmSymmGetWorkspaceSize 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 aclnnGemmSymm 接口执行计算。

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

- **参数说明：**
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
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>输入</td>
      <td>表示矩阵乘的左矩阵，公式中的 A。</td>
      <td><ul><li>支持 FLOAT16、BFLOAT16，且 a、b、c、y 必须是同一种。</li>
      <li>transposeA 为 false 时 shape 为 (…, m, k)；为 true 时 shape 为 (…, k, m)。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2~3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>表示矩阵乘的右矩阵，公式中的 B。</td>
      <td><ul><li>数据类型、维度数与 a 一致。</li>
      <li>transposeB 为 false 时 shape 为 (…, k, m)；为 true 时 shape 为 (…, m, k)。</li>
      <li>3 维时 batch 轴需要与 a 一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2~3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>c</td>
      <td>输入</td>
      <td>表示残差矩阵，公式中的 C。可选。</td>
      <td><ul><li>β 不为 0 时必须传入，shape 为 (…, m, m)，batch 轴与 a 一致。</li>
      <li>β 为 0 时可以传 nullptr，此时不读取 C。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2~3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示计算结果，公式中的 Y。</td>
      <td>shape 为 (…, m, m)，维度数和 batch 轴与 a 一致。结果写入 y，不原地覆盖 c。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2~3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alphaOptional(α)</td>
      <td>输入</td>
      <td>表示公式中的 α。</td>
      <td>为 nullptr 时默认取 1.0。为 0 时跳过矩阵乘，输出为 β × C。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>betaOptional(β)</td>
      <td>输入</td>
      <td>表示公式中的 β。</td>
      <td>为 nullptr 时默认取 1.0，此时必须传入 c。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeA</td>
      <td>输入</td>
      <td>是否按转置布局解读 a。</td>
      <td>为 true 时 a 以 (…, k, m) 存储。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeB</td>
      <td>输入</td>
      <td>是否按转置布局解读 b。</td>
      <td>为 true 时 b 以 (…, m, k) 存储。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fillMode</td>
      <td>输入</td>
      <td>输出区域模式。</td>
      <td>可传 nullptr 或 "full"。当前只写出完整对称矩阵，传入其他字符串返回参数错误。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>输入</td>
      <td>指定 Cube 计算后的累加精度。</td>
      <td>4 表示高精度，且只在传入 C 且 beta 不为 0 时生效：Fixpipe 把 fp32 累加器原样写入 UB，向量加完再 cast 回输入类型。没有 x3 时与 0、1、2、3 相同，Fixpipe 直接 cast 成 fp16 或 bf16。其他值返回参数错误。</td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 809px"><colgroup>
  <col style="width: 257px">
  <col style="width: 121px">
  <col style="width: 431px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>传入的 a、b 或 y 是空指针。</td>
    </tr>
    <tr>
      <td>β 不为 0 时 c 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>a、b、c、y 的数据类型不是 FLOAT16 或 BFLOAT16，彼此不一致，或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>维度数不是 2 或 3，或 a、b、y 以及已传入的 c 维度数不一致。</td>
    </tr>
    <tr>
      <td>m 不大于 0、k 小于 0，或 b、y 的 shape 与 a 不匹配。</td>
    </tr>
    <tr>
      <td>fillMode 不是空指针且不为 "full"。</td>
    </tr>
    <tr>
      <td>cubeMathType 不在 0 到 4 之间。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前产品不是 Ascend 950。</td>
    </tr>
  </tbody>
  </table>

## aclnnGemmSymm

- **参数说明：**

  <div style="overflow-x: auto;">
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnGemmSymmGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 stream。</td>
    </tr>
  </tbody>
  </table>
  </div>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnnGemmSymm 默认确定性实现。每个输出元素由单条 cube 乘积写出，对角块只写一次，非对角块的两处写出使用同一次乘积。
  <!-- end id11 -->

- 支持 FLOAT16 和 BFLOAT16，格式只支持 ND，维度只支持 2 或 3。同一调用里 a、b、c、y 的数据类型必须一致。3 维的最高维是 batch，a、b、c、y 的 batch 必须一致。
- m 必须大于 0，k 可以是 0。k 为 0 或 α 为 0 时不做矩阵乘，输出为 β × C；此时若 β 也为 0，输出为全 0。
- β 不为 0 时必须传入 c。调用方需要保证 A @ B 对称；β 不为 0 时，输出的对称性还依赖 C 对称。
- y 是独立输出，不原地更新 c。
- fillMode 只支持空指针或 "full"。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

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

  // Y = 1 * (A @ B)，A 为 (4, 2) 全 1，B 为 (2, 4) 全 1，beta 为 0，不传 c。结果为全 2。
  std::vector<int64_t> aShape = {4, 2};
  std::vector<int64_t> bShape = {2, 4};
  std::vector<int64_t> yShape = {4, 4};
  void* aDeviceAddr = nullptr;
  void* bDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* a = nullptr;
  aclTensor* b = nullptr;
  aclTensor* y = nullptr;
  std::vector<uint16_t> aHostData(GetShapeSize(aShape), 0x3f80); // bf16 的 1.0
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
