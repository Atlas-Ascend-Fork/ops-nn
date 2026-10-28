# aclnnConfidenceHeadFused

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：将 hidden 与可选的 markovEmbOptional 按最后一维进行逻辑拼接，与 weight 做矩阵向量乘；加上可选 biasOptional，经 temperatureOptional 缩放、截断和 Sigmoid 后，输出每行的置信度。计算使用 FP32 累加及激活，输出为 FP32。
- 计算公式：

  $$
  z_{b,n}=\sum_{k=0}^{K_1-1}hidden_{b,n,k}\,weight_k
  +\sum_{k=0}^{K_2-1}markovEmb_{b,n,k}\,weight_{K_1+k}+bias_n
  $$

  $$
  u_{b,n}=-\frac{z_{b,n}}{T_{\mathrm{eff}}},\qquad out_{b,n}=\frac{1}{1+\exp(u_{b,n})}
  $$

  二维输入省略下标b。markovEmbOptional为nullptr时，第二项为0，K2=0；biasOptional为nullptr时，不加偏置；temperatureOptional为nullptr，或当前值为+0/-0时，有效温度T_eff=1；其余情况T_eff取当前值，允许负值，不修改输入Tensor。定义 `C=FP32(INT32_MAX)=2147483648.0`，两个clamp参数分别转为FP32；nullptr分别映射为 `L=-C`、`H=C`。计算指数之前，L不等于-C时令 `u=max(u,L)`，H不等于C时再令 `u=min(u,H)`；等于对应约定值时跳过该侧截断。截断作用于指数的输入 `-z/T_eff`，不是概率输出。biasOptional[N] 在不同 batch 之间广播。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnConfidenceHeadFusedGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConfidenceHeadFused”接口执行计算。

```cpp
aclnnStatus aclnnConfidenceHeadFusedGetWorkspaceSize(
  const aclTensor *hidden,
  const aclTensor *weight,
  const aclTensor *markovEmbOptional,
  const aclTensor *temperatureOptional,
  const aclTensor *biasOptional,
  const aclScalar *clampLowOptional,
  const aclScalar *clampHighOptional,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnConfidenceHeadFused(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

接口调用时须显式传入全部参数；不使用某个可选输入时，在对应位置传入nullptr。

## aclnnConfidenceHeadFusedGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
        <th>维度</th>
        <th>非连续</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>hidden</td>
        <td>输入</td>
        <td>待投影的特征。</td>
        <td><li>必选，shape 为 [N,K1] 或 [B,N,K1]。</li></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND、NCL（三维）</td>
        <td>2、3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>所有行共用的投影权重。</td>
        <td><li>必选，shape 为 [K1+K2]，数据类型与 hidden 一致。</li></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>markovEmbOptional</td>
        <td>输入</td>
        <td>可选的 Markov 特征。</td>
        <td><li>可选参数。传入nullptr时不拼接Markov特征，K2=0；非空时shape为[N,K2]或[B,N,K2]，rank、前导维度和数据类型与hidden一致。</li></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND、NCL（三维）</td>
        <td>2、3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>temperatureOptional</td>
        <td>输入</td>
        <td>温度系数。</td>
        <td><li>可选参数。非空时为[1] Tensor；传入nullptr或当前值为+0/-0时按温度1计算，其他值（含负值）使用当前值，不修改输入Tensor；接口仅校验类型、形状等元数据，数值由Kernel读取。</li></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>投影后的逐行偏置。</td>
        <td><li>可选参数。传入nullptr时不加偏置；非空时shape为[N]，同一n在各batch使用相同的biasOptional[n]。N&gt;1时不接受[1]广播。</li></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>clampLowOptional</td>
        <td>输入</td>
        <td>指数输入的截断下限 L。</td>
        <td><li>可选参数。传入nullptr时使用-2147483648.0，不做下限截断；非空时使用aclScalar，其数据类型须可转换为FLOAT32，转换结果等于该约定值时也不做下限截断。</li></td>
        <td>aclScalar</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>clampHighOptional</td>
        <td>输入</td>
        <td>指数输入的截断上限 H。</td>
        <td><li>可选参数。传入nullptr时使用2147483648.0，不做上限截断；非空时使用aclScalar，其数据类型须可转换为FLOAT32，转换结果等于该约定值时也不做上限截断。</li></td>
        <td>aclScalar</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>每行的置信度。</td>
        <td><li>必选；由调用方创建，shape 为 [N] 或 [B,N]，与 hidden 的前导维度一致。</li></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1、2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
        <td>必选输出指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>必选输出指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景报错：

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的hidden、weight、out、workspaceSize或executor是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>当前平台不支持该接口。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>输入或输出的数据类型、数据格式、维度不满足参数说明。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>hidden、markovEmbOptional、weight、biasOptional或out的shape不匹配，或维度、展平行数及K1+K2超出支持范围。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>clampLowOptional或clampHighOptional的数据类型不能转换为FLOAT32。</td>
      </tr>
  </tbody></table>

  计算模板所需片上容量不足时，Tiling 失败，第一段接口返回非 SUCCESS；内部错误的包装状态以执行框架为准。temperatureOptional 的+0/-0是合法输入，执行时按温度1计算，不作为参数错误。

## aclnnConfidenceHeadFused

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
        <th>维度</th>
        <th>非连续</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
        <td>传入第一段接口返回大小对应的Device内存地址；第一段返回大小为0时可传nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>Device侧workspace的字节数。</td>
        <td>传入第一段接口aclnnConfidenceHeadFusedGetWorkspaceSize返回的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>包含了算子计算流程的执行器。</td>
        <td>传入第一段接口aclnnConfidenceHeadFusedGetWorkspaceSize返回的执行器。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
        <td>传入执行算子任务的Stream。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 支持合法基础格式，不支持 FRACTAL_NZ。接口内部将输入转连续并统一为 ND，按 shape 的实际维度顺序计算，不因格式名称重排维度。
- hidden、weight、markovEmbOptional 必须同为 FLOAT16 或同为 BFLOAT16，不支持混合类型。temperatureOptional、biasOptional、out 仅支持 FLOAT32。
- **非连续Tensor**：hidden、weight、markovEmbOptional、temperatureOptional和biasOptional支持合法非连续视图，out支持合法非连续输出视图。调用方可传入切片或转置后的Tensor，无需自行调用contiguous。接口内部进行输入转连续、格式统一和输出视图拷贝，所需workspace以第一段接口返回的大小为准。
- 各维度、展平后的行数 B*N 和 K1+K2 均须位于 [0,UINT32_MAX]。二维输入按 B=1 理解。
- 三维输入将 B、N 合并为行维计算，权重在所有 batch 共用。biasOptional[N] 按 batch 重复，不能传 [B,N] 或 [B*N]。
- K1+K2=0 时仍执行 biasOptional、temperatureOptional、可选 clamp 和 Sigmoid。biasOptional为nullptr，且已启用的截断不改变 0 时，输出全部为 0.5。B=0 或 N=0 时返回空输出，仍校验输入输出元数据；不读取temperatureOptional的数据值，也不安排计算任务。
- temperatureOptional 是输入 Tensor，非空计算时由Kernel从GM读取当前值，不在Host Tiling中读取或固化。缺省或当前值为+0/-0时按温度1计算，其他值（含负值）使用当前值，不修改输入Tensor。ACLNN和Torch均不在Host读取温度数值；Tiling仅校验FP32 [1]元数据，不注册温度值依赖。
- clampLowOptional、clampHighOptional分别控制下限和上限截断，按FP32计算。约定下限-2147483648.0、上限2147483648.0表示不启用对应截断；nullptr映射为该侧约定值，显式传入并转换为相同值时也跳过。显式0是有效阈值。两侧均启用时先 max 下限、再 min 上限；L&gt;H 时该顺序使有限输入截断为 H。不校验上下限的大小关系。
- 正常非空 GEMV 采用 weight 全量驻留 L1 的模板，K 的实际可用上限受片上容量限制；不能将 UINT32_MAX 理解为任意设备均可执行的 K 上限。
- 确定性计算

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnnConfidenceHeadFused默认确定性实现。
  <!-- end id8 -->

## 调用示例

示例使用 FP16 三维特征、Markov 特征、biasOptional[N] 和值为2的temperatureOptional；两个 batch 的预期输出均约为 [0.5621765,0.6791787,0.7772999]。FP16 数据按真实 16 位编码传入。切换为 BF16 时，将特征常数 1 的编码改为 0x3F80，权重常数 0.5 的编码改为 0x3F00，并将对应 Tensor 的 dtype 改为 ACL_BF16。

完整示例请参见 [test_aclnn_confidence_head_fused.cpp](../examples/arch35/test_aclnn_confidence_head_fused.cpp)。编译与执行过程参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)，通过 `--soc=ascend950` 或 `--soc=ascend350` 选择目标芯片。

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_confidence_head_fused.h"

#define CHECK_RET(expr)                          \
    do {                                        \
        auto ret = (expr);                       \
        if (ret != ACL_SUCCESS) {               \
            std::printf("error: %d\n", ret);    \
            return 1;                           \
        }                                       \
    } while (0)

struct Runtime {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    ~Runtime()
    {
        if (stream != nullptr) { aclrtDestroyStream(stream); }
        if (deviceSet) { aclrtResetDevice(0); }
        if (initialized) { aclFinalize(); }
    }
};

struct DeviceTensor {
    void* data = nullptr;
    aclTensor* tensor = nullptr;
    ~DeviceTensor()
    {
        if (tensor != nullptr) { aclDestroyTensor(tensor); }
        if (data != nullptr) { aclrtFree(data); }
    }

    template <typename T>
    int Create(const std::vector<T>& values, const std::vector<int64_t>& shape, aclDataType dtype)
    {
        size_t bytes = values.size() * sizeof(T);
        CHECK_RET(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_RET(aclrtMemcpy(data, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), data);
        return tensor == nullptr ? 1 : 0;
    }
};

struct Workspace {
    void* data = nullptr;
    ~Workspace() { if (data != nullptr) { aclrtFree(data); } }
};

int main()
{
    Runtime runtime;
    CHECK_RET(aclInit(nullptr));
    runtime.initialized = true;
    CHECK_RET(aclrtSetDevice(0));
    runtime.deviceSet = true;
    CHECK_RET(aclrtCreateStream(&runtime.stream));

    DeviceTensor hidden, weight, markov, temperatureOptional, biasOptional, out;
    CHECK_RET(hidden.Create(std::vector<uint16_t>(12, 0x3C00), {2, 3, 2}, ACL_FLOAT16));
    CHECK_RET(weight.Create(std::vector<uint16_t>(3, 0x3800), {3}, ACL_FLOAT16));
    CHECK_RET(markov.Create(std::vector<uint16_t>(6, 0x3C00), {2, 3, 1}, ACL_FLOAT16));
    CHECK_RET(temperatureOptional.Create(std::vector<float>{2.0f}, {1}, ACL_FLOAT));
    CHECK_RET(biasOptional.Create(std::vector<float>{-1.0f, 0.0f, 1.0f}, {3}, ACL_FLOAT));
    CHECK_RET(out.Create(std::vector<float>(6, 0.0f), {2, 3}, ACL_FLOAT));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_RET(aclnnConfidenceHeadFusedGetWorkspaceSize(hidden.tensor, weight.tensor, markov.tensor,
                 temperatureOptional.tensor, biasOptional.tensor, nullptr, nullptr, out.tensor, &workspaceSize, &executor));
    Workspace workspace;
    if (workspaceSize != 0) {
        CHECK_RET(aclrtMalloc(&workspace.data, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    auto status = aclnnConfidenceHeadFused(workspace.data, workspaceSize, executor, runtime.stream);
    // Synchronize before releasing device buffers, including the execution-error path.
    auto syncStatus = aclrtSynchronizeStream(runtime.stream);
    CHECK_RET(status);
    CHECK_RET(syncStatus);
    std::vector<float> result(6);
    CHECK_RET(aclrtMemcpy(result.data(), result.size() * sizeof(float), out.data,
                         result.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (float value : result) { std::printf("%.7f\n", value); }
    return 0;
}
```
