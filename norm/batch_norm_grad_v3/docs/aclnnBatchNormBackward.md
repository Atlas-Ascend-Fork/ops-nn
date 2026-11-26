# aclnnBatchNormBackward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

- 接口功能：计算[aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm.md)反向传播中的输入梯度、缩放参数梯度和偏置参数梯度。

- 计算公式：

  每个通道的元素数`n`为除通道轴外各维长度的乘积。$x_i$、$\mathrm{gradOut}_i$分别为该通道的输入元素和上游梯度元素，$i=0,\ldots,n-1$。

  - 当`training`为true时：

    `rstd`为计算中使用的标准差倒数。

    <!-- npu="A3,910b,910" id12 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas训练系列产品</term>：`saveInvstd`传入保存的方差，算子计算$\mathrm{rstd}=1/\sqrt{\mathrm{saveInvstd}+\mathrm{eps}}$。
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT系列产品</term>：`saveInvstd`传入前向保存的标准差倒数，直接作为`rstd`使用。
    <!-- end id13 -->

    $$
    \mathrm{gradInput}_i=
    \frac{\mathrm{weight}\,\mathrm{rstd}}{n}
    \left(n\,\mathrm{gradOut}_i-\mathrm{gradBias}
    -(x_i-\mathrm{saveMean})\,\mathrm{rstd}\,\mathrm{gradWeight}\right)
    $$

    $$
    \mathrm{gradWeight}=\mathrm{rstd}\sum_{i=0}^{n-1}
    \mathrm{gradOut}_i(x_i-\mathrm{saveMean})
    $$

    $$
    \mathrm{gradBias}=\sum_{i=0}^{n-1}\mathrm{gradOut}_i
    $$

  - 当`training`为false时：

    $$
    \mathrm{gradInput}_i=\mathrm{gradOut}_i\,
    \frac{\mathrm{weight}}{\sqrt{\mathrm{runningVar}+\mathrm{eps}}}
    $$

    $$
    \mathrm{gradWeight}=\frac{1}{\sqrt{\mathrm{runningVar}+\mathrm{eps}}}
    \sum_{i=0}^{n-1}\mathrm{gradOut}_i(x_i-\mathrm{runningMean})
    $$

    $$
    \mathrm{gradBias}=\sum_{i=0}^{n-1}\mathrm{gradOut}_i
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBatchNormBackwardGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBatchNormBackward”接口执行计算。

```Cpp
aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(
  const aclTensor    *gradOut,
  const aclTensor    *input,
  const aclTensor    *weight,
  const aclTensor    *runningMean,
  const aclTensor    *runningVar,
  const aclTensor    *saveMean,
  const aclTensor    *saveInvstd,
  bool                training,
  double              eps,
  const aclBoolArray *outputMask,
  aclTensor          *gradInput,
  aclTensor          *gradWeight,
  aclTensor          *gradBias,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnBatchNormBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormBackwardGetWorkspaceSize

- **参数说明**

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
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOut（aclTensor*）</td>
      <td>输入</td>
      <td>表示梯度Tensor，对应公式中的`gradOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>支持的shape和格式有：2维（对应的格式为NC），3维（对应的格式为NCL），4维（对应的格式为NCHW、NHWC），5维（对应的格式为NCDHW、NDHWC），6-8维（对应的格式为ND）。</li><li>NHWC、NDHWC的通道轴为最后一维，其余所列格式的通道轴为第2维。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NC、NCL、NCHW、NHWC、NCDHW、NDHWC、ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向的输入Tensor，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape、数据格式均需要与`gradOut`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NC、NCL、NCHW、NHWC、NCDHW、NDHWC、ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示权重Tensor，对应公式中的`weight`。</td>
      <td><ul><li>可选。传空指针时，各通道使用1。</li><li>主输入非空时，传入的张量Shape为[C]，其中C为通道数。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示训练期间计算的平均值，对应公式中的`runningMean`。</td>
      <td><ul><li>可选。传空指针时，各通道使用0。</li><li>主输入非空时，传入的张量Shape为[C]，其中C为通道数。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示训练期间计算的方差，对应公式中的`runningVar`。</td>
      <td><ul><li>可选。传空指针时，各通道使用1。</li><li>主输入非空时，传入的张量Shape为[C]，其中C为通道数。</li><li>请传入有限非负值。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveMean（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示保存的均值，对应公式中的`saveMean`。</td>
      <td><ul><li>可选。传空指针时，各通道使用0。</li><li>训练且主输入非空时，传入的张量Shape为[C]。推理时该统计量不参与计算，也可传Shape为[0]的张量。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveInvstd（aclTensor*）</td>
      <td>可选输入</td>
      <td>训练时使用的保存方差或标准差倒数，取决于产品。参见训练公式和产品约束。</td>
      <td><ul><li>可选。传空指针时，各通道使用1。</li><li>训练且主输入非空时，传入的张量Shape为[C]。推理时该统计量不参与计算，也可传Shape为[0]的张量。</li><li>请传入有限非负值。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>training（bool）</td>
      <td>输入</td>
      <td>表示标记是否训练场景，对应公式描述中的`training`。</td>
      <td>true表示训练场景，false表示推理场景。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps（double）</td>
      <td>输入</td>
      <td>表示添加到方差中的值，以避免出现除以零的情况，对应公式中的`eps`。</td>
      <td>必填，由调用方指定。建议使用有限正数，例如示例中的1e-5。训练时的用法参见计算公式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask（aclBoolArray*）</td>
      <td>输入</td>
      <td>表示输出的掩码。</td>
      <td>传入长度为3的非空指针，元素顺序为gradInput、gradWeight、gradBias。对应元素为true时，需要传入对应的输出Tensor。为false时跳过该结果，可传空指针。</td>
      <td>BoolArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput（aclTensor*）</td>
      <td>输出</td>
      <td>表示输入Tensor的梯度，对应公式中的`gradInput`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选输出，若outputMask[0]为true，则需要输出，否则不输出。</li><li>数据类型、shape、数据格式均需要与`gradOut`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NC、NCL、NCHW、NHWC、NCDHW、NDHWC、ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight（aclTensor*）</td>
      <td>输出</td>
      <td>表示缩放参数的梯度，对应公式中的`gradWeight`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选输出，若outputMask[1]为True，则需要输出，否则不输出。</li><li>长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias（aclTensor*）</td>
      <td>输出</td>
      <td>表示偏置参数的梯度，对应公式中的`gradBias`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选输出，若outputMask[2]为True，则需要输出，否则不输出。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>：
    - 参数`gradOut`、`input`、`weight`、`runningMean`、`runningVar`、`saveMean`、`saveInvstd`、`gradInput`、`gradWeight`、`gradBias`的数据类型不支持BFLOAT16。
    - 参数`weight`、`runningMean`、`runningVar`、`saveMean`、`saveInvstd`、`gradWeight`、`gradBias`的数据类型与`gradOut`的保持一致。
    - 参数`gradOut`、`input`、`gradInput`的数据格式不支持NHWC、NDHWC。
    - 参数`saveInvstd`表示input的方差。
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 参数`weight`、`runningMean`、`runningVar`、`saveMean`、`saveInvstd`、`gradWeight`、`gradBias`的数据类型与`gradOut`的保持一致。
    - 参数`gradOut`、`input`、`gradInput`的数据格式不支持NHWC、NDHWC。
    - 参数`saveInvstd`表示input的方差。
  <!-- end id8 -->

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - 当`gradOut`为FLOAT32，`weight`、`runningMean`、`runningVar`、`saveMean`、`saveInvstd`必须为FLOAT32。
    - 当`gradOut`为FLOAT16，`weight`、`runningMean`、`runningVar`、`saveMean`、`saveInvstd`必须为FLOAT16或FLOAT32。
    - 当`gradOut`为BFLOAT16，`weight`、`runningMean`、`runningVar`、`saveMean`、`saveInvstd`必须为BFLOAT16或FLOAT32。
    - 当`weight`为FLOAT32，`gradWeight`必须为FLOAT32；当`weight`为FLOAT16，`gradWeight`必须为FLOAT16或FLOAT32；当`weight`为BFLOAT16，`gradWeight`必须为BFLOAT16或FLOAT32。
    - `gradWeight`、`gradBias`的数据类型需要保持一致。
    - `runningVar`数据类型需要与`runningMean`的数据类型一致。
    - `saveInvstd`数据类型需要与`saveMean`的数据类型一致。
    - 参数`saveInvstd`为前向保存的标准差倒数，已包含epsilon。
  <!-- end id9 -->

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="5">161001</td>
      <td>传入的gradOut、input指针是空指针。</td>
    </tr>
    <tr>
      <td>当gradOut和input均为非空Tensor时，传入的outputMask是空指针。</td>
    </tr>
    <tr>
      <td>当outputMask[0]为true，传入的gradInput是空指针时。</td>
    </tr>
    <tr>
      <td>当outputMask[1]为true，传入的gradWeight是空指针时。</td>
    </tr>
    <tr>
      <td>当outputMask[2]为true，传入的gradBias是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>input，gradOut，数据类型、数据格式和shape不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>weight，runningMean，runningVar，saveMean，saveInvstd非空时，数据类型、数据格式和shape不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>outputMask的长度为0。</td>
    </tr>
    <tr>
      <td>当outputMask[0]为true，gradInput数据类型、数据格式和shape不在支持的范围之内时。</td>
    </tr>
    <tr>
      <td>当outputMask[1]为true，gradWeight数据类型、数据格式和shape不在支持的范围之内时。</td>
    </tr>
    <tr>
      <td>当outputMask[2]为true，gradBias数据类型、数据格式和shape不在支持的范围之内时。</td>
    </tr>
    <tr>
      <td>weight，runningMean，runningVar，saveMean，saveInvstd，gradWeight（非空时），gradBias（非空时）的shape长度与input shape中channel轴的长度不相等。</td>
    </tr>
    <tr>
      <td>input、gradOut、gradInput（非空时）的数据格式不一致。</td>
    </tr>
    <tr>
      <td>input、gradOut、gradInput（非空时）的数据类型不一致。</td>
    </tr>
    <tr>
      <td>input、gradOut、gradInput（非空时）的shape不一致，或者shape的维度大于8维或者小于2维。</td>
    </tr>
  </tbody></table>

  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当gradOut和input均为非空Tensor时，outputMask的长度必须为3，否则返回ACLNN_ERR_PARAM_INVALID（161002）。
  <!-- end id14 -->

## aclnnBatchNormBackward

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBatchNormBackwardGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- ACLNN接口接收具体的Tensor shape。`-1`（未知维）和`[-2]`（未知秩）属于GE图定义，应在执行前具体化。参见[图算子约束](../README.md#约束说明)。
- 当`input`或`gradOut`是零元素Tensor时，第一段接口跳过常规参数校验。执行时会将掩码已启用且提供了非空Tensor的`gradWeight`、`gradBias`置零。

- 确定性计算：

  <!-- npu="A3,910b,910" id10 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas训练系列产品</term>：aclnnBatchNormBackward默认非确定性实现，支持通过aclrtSetSysParamOpt开启确定性。
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnnBatchNormBackward默认确定性实现。
  <!-- end id11 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cmath>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_backward.h"

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

bool CheckOutput(void* deviceAddr, const std::vector<float>& expected, const char* name)
{
    std::vector<float> actual(expected.size());
    const size_t bytes = actual.size() * sizeof(float);
    const auto ret = aclrtMemcpy(actual.data(), bytes, deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Copy %s failed. ERROR: %d\n", name, ret);
        return false;
    }
    constexpr float tolerance = 1e-4f;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > tolerance * (1.0f + std::abs(expected[i]))) {
            LOG_PRINT("%s[%zu]: expected %f, got %f\n", name, i, expected[i], actual[i]);
            return false;
        }
    }
    LOG_PRINT("PASS %s\n", name);
    return true;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutShape = {1, 2, 4};
    std::vector<int64_t> selfShape = {1, 2, 4};
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> rMeanShape = {2};
    std::vector<int64_t> rVarShape = {2};
    std::vector<int64_t> sMeanShape = {2};
    std::vector<int64_t> sVarShape = {2};
    std::vector<int64_t> gradInShape = {1, 2, 4};
    std::vector<int64_t> gradWeightShape = {2};
    std::vector<int64_t> gradBiasShape = {2};
    void* gradOutDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* rMeanDeviceAddr = nullptr;
    void* rVarDeviceAddr = nullptr;
    void* sMeanDeviceAddr = nullptr;
    void* sVarDeviceAddr = nullptr;
    void* gradInDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;
    void* gradBiasDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* self = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* rMean = nullptr;
    aclTensor* rVar = nullptr;
    aclTensor* sMean = nullptr;
    aclTensor* sVar = nullptr;
    aclBoolArray* outMask = nullptr;
    aclTensor* gradIn = nullptr;
    aclTensor* gradWeight = nullptr;
    aclTensor* gradBias = nullptr;
    std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> rMeanHostData = {0, 0};
    std::vector<float> rVarHostData = {1, 1};
    std::vector<float> sMeanHostData = {0, 0};
    std::vector<float> sVarHostData = {1, 1};
    std::vector<float> gradInHostData(8, 0);
    std::vector<float> gradWeightHostData(2, 0);
    std::vector<float> gradBiasHostData(2, 0);

    bool training = true;
    double eps = 1e-5;
    // 创建gradOut aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rMean aclTensor
    ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rVar aclTensor
    ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sMean aclTensor
    ret = CreateAclTensor(sMeanHostData, sMeanShape, &sMeanDeviceAddr, aclDataType::ACL_FLOAT, &sMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sVar aclTensor
    ret = CreateAclTensor(sVarHostData, sVarShape, &sVarDeviceAddr, aclDataType::ACL_FLOAT, &sVar);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outMask aclBoolArray
    bool maskData[3] = {true, true, true};
    outMask = aclCreateBoolArray(&(maskData[0]), 3);
    // 创建gradIn aclTensor
    ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradWeight aclTensor
    ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradBias aclTensor
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNormBackward接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnBatchNormBackward第一段接口
    ret = aclnnBatchNormBackwardGetWorkspaceSize(gradOut, self, weight, rMean, rVar, sMean, sVar, training, eps,
                                                 outMask, gradIn, gradWeight, gradBias, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnBatchNormBackward第二段接口
    ret = aclnnBatchNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Check all three outputs against the supplied mean=0 and invstd=1.
    // Per channel: dbias=sum(dy), dweight=sum(dy*x), dx=dy-(dbias+x*dweight)/4.
    const bool dxOk = CheckOutput(gradInDeviceAddr, {-1.5f, -4.0f, -6.5f, -9.0f, -127.5f, -158.0f, -188.5f, -219.0f},
                                  "dx");
    const bool weightOk = CheckOutput(gradWeightDeviceAddr, {14.0f, 126.0f}, "dweight");
    const bool biasOk = CheckOutput(gradBiasDeviceAddr, {6.0f, 22.0f}, "dbias");

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOut);
    aclDestroyTensor(self);
    aclDestroyTensor(weight);
    aclDestroyTensor(rMean);
    aclDestroyTensor(rVar);
    aclDestroyTensor(sMean);
    aclDestroyTensor(sVar);
    aclDestroyBoolArray(outMask);
    aclDestroyTensor(gradIn);
    aclDestroyTensor(gradWeight);
    aclDestroyTensor(gradBias);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(selfDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(rMeanDeviceAddr);
    aclrtFree(rVarDeviceAddr);
    aclrtFree(sMeanDeviceAddr);
    aclrtFree(sVarDeviceAddr);
    aclrtFree(gradInDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return dxOk && weightOk && biasOk ? 0 : 1;
}
```
