# aclnnSituGluGroupQuant

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

- 接口功能：面向MoE分组场景的稀疏SiTU门控激活与MXFP8（e4m3fn，blocksize=32）尾轴动态量化融合算子。输入张量x的最后一维为拼接的gate/up两半（2\*EPR）；激活结果驻留片上不回写GM，随后按MX算法量化为y(FLOAT8_E4M3FN) + y_scale(FLOAT8_E8M0)。
- 计算公式：

  令$h = x.shape[-1] // 2$。根据输入参数activateLeft对x进行前后切分：

  - activateLeft为true时：$gate = x[:, :h], \quad up = x[:, h:]$
  - activateLeft为false时：$gate = x[:, h:], \quad up = x[:, :h]$

  SiTU激活（全程fp32域计算，单次退出舍入）：

  $$
  g = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \frac{1}{1+e^{-gate}}, \qquad
  u = linearBeta \cdot \tanh\left(\frac{up}{linearBeta}\right) \quad (linearBeta > 0)
  $$

  linearBeta ≤ 0时禁用up支路的tanh软限幅，u = up（直通参与相乘）。

  $$
  act = g \times u
  $$

  可选输入weightOptional存在时按行缩放：$act = fp32(act) \times w[row]$。

  MXFP8量化：沿axis（当前仅支持 -1）每32个元素为一组，由组内最大绝对值Amax的指数域导出FLOAT8_E8M0缩放因子s（OCP MX语义），组内每个元素除以s后按roundModeOptional舍入并饱和到FLOAT8_E4M3FN范围得到y。特殊值语义（设备实测口径）：激活结果中的NaN元素逐个置为+0（不整组清零，与NaN同组的有限元素正常量化）；±Inf及有限溢出经饱和转换钳位到输入类型的最大有限值后参与量化；组内全部为NaN时Amax为0，整组y=+0、y_scale=0。当组数为奇数时y_scale每行末尾含1个补位字节，该字节不写出（保留输出缓冲区原值），调用方不应将其读取为有效scale。

- 分组语义：可选输入groupIndexOptional的第i个元素为专家i分到的token行数（count模式）；kernel逐元素累加得到有效行数validRows = min(sum(groupIndexOptional), totalRows)，仅处理前validRows行，其余padding行不写出；groupIndexOptional不传时为全行模式（所有行参与计算）。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSituGluGroupQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSituGluGroupQuant”接口执行计算。

```cpp
aclnnStatus aclnnSituGluGroupQuantGetWorkspaceSize(
    const aclTensor   *x,
    const aclTensor   *weightOptional,
    const aclTensor   *groupIndexOptional,
    double            beta,
    double            linearBeta,
    bool              activateLeft,
    int64_t           axis,
    int64_t           quantMode,
    int64_t           dstType,
    double            dstTypeMax,
    const char        *roundModeOptional,
    int64_t           groupListType,
    const aclTensor   *yOut,
    const aclTensor   *yScaleOut,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnSituGluGroupQuant(
    void              *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream       stream)
```

## aclnnSituGluGroupQuantGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 160px">
  <col style="width: 90px">
  <col style="width: 260px">
  <col style="width: 330px">
  <col style="width: 140px">
  <col style="width: 80px">
  <col style="width: 100px">
  <col style="width: 120px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>待激活量化的数据，最后一维为拼接的gate/up两半。</td>
      <td><ul><li>最后一维必须为正偶数。</li><li>EPR = 末维/2必须为32的倍数且在[128, 3072]内（tiling防呆校验拒收）。</li><li>零长维度退化为0行计算，不拒收。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-24</td>
      <td>√（自动转连续）</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入。逐token（行）的缩放因子，按全局行号索引；存在时激活结果在fp32域按行缩放后再量化。</td>
      <td><ul><li>不传（nullptr）则不缩放。</li><li>元素个数必须等于x展开后的总行数（tiling校验拒收）。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入。每个专家分到的token行数（count模式）。</td>
      <td><ul><li>不传（nullptr）为全行模式（所有行参与计算）。</li><li>传入为分组语义：validRows = min(sum, totalRows)，padding行不写出。</li><li>groupListType必须为1。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>beta（double）</td>
      <td>输入</td>
      <td>gate支路的SiTU激活参数。</td>
      <td>必须大于0。建议取值在[2^-10, 2^10]范围内。tanh采用两路实现（与situ_glu/situ_mx_quant算子及标准tanh.h参考同口径）：|gate/beta| &lt; 0.6用9阶多项式（无exp、无相消），≥0.6用sigmoid分解（饱和区）——不存在“极大beta配合小幅度输入时结果退化为0”的病态区间。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linearBeta（double）</td>
      <td>输入</td>
      <td>up支路的软限幅参数。</td>
      <td>≤0禁用软限幅（up直通），>0启用（up幅度钳位在 ±linearBeta）。建议取值在[2^-10, 2^10]范围内。tanh同beta采用两路实现，无相消精度边界。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft（bool）</td>
      <td>输入</td>
      <td>gate/up方向选择。</td>
      <td>true：gate取x前半部分；false：gate取x后半部分。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis（int64_t）</td>
      <td>输入</td>
      <td>量化轴。</td>
      <td>当前仅支持 -1（尾轴）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td>当前仅支持0（MX量化）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>目标量化数据类型。</td>
      <td>当前仅支持36（FLOAT8_E4M3FN）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax（double）</td>
      <td>输入</td>
      <td>目标数据类型数值上限。</td>
      <td>取值 >= 0，当前不参与计算。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional（char*）</td>
      <td>输入</td>
      <td>舍入模式。</td>
      <td>当前仅支持 "rint"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType（int64_t）</td>
      <td>输入</td>
      <td>groupIndexOptional的语义模式。</td>
      <td>当前必须显式传1（count模式）；0（cumsum预留模式）会被拒收。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化结果。</td>
      <td>shape = x.shape[:-1] + [EPR]。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>1-24</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>MX量化缩放因子。</td>
      <td>shape = x.shape[:-1] + [ceil(ceil(EPR/32)/2), 2]。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-25（x为24维时25维）</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
  <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
  </tr></thead>
  <tbody>
  <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>x、yOut、yScaleOut、roundModeOptional、workspaceSize、executor存在空指针（weightOptional与groupIndexOptional为可选输入，允许空指针）。</td>
  </tr>
  <tr>
  </tr>
  <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>beta小于等于0。</td>
  </tr>
  <tr>
    <td>dstTypeMax小于0；groupListType不为1（0为预留cumsum模式，同样拒收）。</td>
  </tr>
  <tr>
    <td>x的数据类型不在FLOAT16、BFLOAT16范围之内；weightOptional传入时不是FLOAT32；groupIndexOptional传入时不是INT64。</td>
  </tr>
  <tr>
    <td>yOut的数据类型不是FLOAT8_E4M3FN，或yScaleOut的数据类型不是FLOAT8_E8M0。</td>
  </tr>
  <tr>
    <td>张量的数据格式不是ND、NCL、NCHW、NCDHW等效线性格式。</td>
  </tr>
  <tr>
    <td>x的维度小于1，或x最后一维不是正偶数。</td>
  </tr>
  <tr>
    <td>yOut、yScaleOut的shape与x推导的期望shape不一致。</td>
   </tr>
   <tr>
       <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
       <td rowspan="3">561002</td>
       <td>axis不为 -1、quantMode不为0、dstType不为36、roundModeOptional不为 "rint"（host tiling校验拒收）。</td>
   </tr>
  <tr>
    <td>EPR（x末维 / 2）不是32的倍数，或不在[128, 3072]区间内（host tiling校验拒收）。</td>
  </tr>
  <tr>
    <td>weightOptional传入时元素个数不等于x展开后的总行数（host tiling校验拒收）。</td>
  </tr>
  </tbody></table>

## aclnnSituGluGroupQuant

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSituGluGroupQuantGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- x.shape[-1]必须为正偶数；EPR = x.shape[-1] / 2必须为32的倍数且在[128, 3072]区间内（越界场景对应原算子已知缺陷：EPR &lt; 128时y_scale静默丢失、EPR &gt; 3072时整行拷贝越界踩踏UB、EPR % 32 != 0时尾组垃圾，tiling防呆直接拒收）。
- beta > 0。linearBeta ≤ 0禁用软限幅（up直通），> 0启用，任意取值合法。
- axis当前仅支持 -1；quantMode仅支持0；dstType仅支持36（FLOAT8_E4M3FN）；roundModeOptional仅支持 "rint"；groupListType必须显式传1。非法值均被校验拒收。
- weightOptional存在时元素个数必须等于x展开后的总行数，否则拒收。
- x支持1-24维（y_scale比x多一维，受框架shape 25维上限约束，x上限为24维），不拒空Tensor（零长维度退化为0行计算）。
- 确定性说明：aclnnSituGluGroupQuant默认确定性实现。

## 调用示例

示例代码如下（完整可编译运行工程另见[test_aclnn_situ_glu_group_quant](../examples/arch35/test_aclnn_situ_glu_group_quant.cpp)），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdio>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_situ_glu_group_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

// float32 转 bfloat16 位模式（就近舍入），用于构造宿主侧输入数据
static uint16_t FloatToBf16(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

// 由 shape 计算连续内存的 strides
static std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

int main()
{
    // 1. 初始化设备与 stream
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return -1);

    // 2. 构造输入：x [8, 256] BF16（gate/up 在末维拼接，EPR=128），
    //    groupIndex [2] = {4, 4}（2 个专家各 4 行，count 模式）
    const int64_t rows = 8;
    const int64_t epr = 128;
    std::vector<int64_t> xShape = {rows, 2 * epr};
    std::vector<int64_t> yShape = {rows, epr};
    std::vector<int64_t> scaleShape = {rows, (epr / 32 + 1) / 2, 2};  // ceil(ceil(EPR/32)/2)=2
    std::vector<int64_t> tokensShape = {2};

    std::vector<uint16_t> xHost(rows * 2 * epr);
    for (size_t i = 0; i < xHost.size(); i++) {
        xHost[i] = FloatToBf16((static_cast<float>(i % 17) - 8.0f) * 0.25f);  // [-2.0, 2.0] 简单数据
    }
    std::vector<int64_t> tokensHost = {4, 4};

    void *xDevice = nullptr, *tokensDevice = nullptr, *yDevice = nullptr, *scaleDevice = nullptr;
    CHECK_RET(aclrtMalloc(&xDevice, xHost.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&tokensDevice, tokensHost.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&yDevice, rows * epr, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&scaleDevice, rows * ((epr / 32 + 1) / 2) * 2, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMemcpy(xDevice, xHost.size() * sizeof(uint16_t), xHost.data(),
                          xHost.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMemcpy(tokensDevice, tokensHost.size() * sizeof(int64_t), tokensHost.data(),
                          tokensHost.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, return -1);

    // 3. 创建 aclTensor：weightOptional 不使用传 nullptr；x 末维为 2*EPR
    auto xs = MakeStrides(xShape);
    auto ys = MakeStrides(yShape);
    auto ss = MakeStrides(scaleShape);
    auto ts = MakeStrides(tokensShape);
    aclTensor *x = aclCreateTensor(xShape.data(), xShape.size(), ACL_BF16, xs.data(), 0, ACL_FORMAT_ND,
                                   xShape.data(), xShape.size(), xDevice);
    aclTensor *tokens = aclCreateTensor(tokensShape.data(), tokensShape.size(), ACL_INT64, ts.data(), 0,
                                        ACL_FORMAT_ND, tokensShape.data(), tokensShape.size(), tokensDevice);
    aclTensor *y = aclCreateTensor(yShape.data(), yShape.size(), ACL_FLOAT8_E4M3FN, ys.data(), 0, ACL_FORMAT_ND,
                                   yShape.data(), yShape.size(), yDevice);
    aclTensor *yScale = aclCreateTensor(scaleShape.data(), scaleShape.size(), ACL_FLOAT8_E8M0, ss.data(), 0,
                                        ACL_FORMAT_ND, scaleShape.data(), scaleShape.size(), scaleDevice);

    // 4. 两段式调用：beta=1.0, linearBeta=2.0, activateLeft=true（gate 取前半），
    //    扩展属性当前均为单一支持值，groupListType 必须显式传 1（count 模式）
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnSituGluGroupQuantGetWorkspaceSize(x, nullptr, tokens, 1.0, 2.0, true, -1, 0, 36, 0.0,
                                                      "rint", 1, y, yScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, printf("GetWorkspaceSize failed. ERROR: %d\n", static_cast<int>(ret)); return -1);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    }
    ret = aclnnSituGluGroupQuant(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclnnSituGluGroupQuant failed. ERROR: %d\n", static_cast<int>(ret)); return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);

    // 5. 回读输出（前 8 个 y 字节与第 0 行 scale 字节，仅供参考）
    uint8_t yFirst8[8] = {0};
    uint8_t scaleRow0[4] = {0};
    (void)aclrtMemcpy(yFirst8, sizeof(yFirst8), yDevice, sizeof(yFirst8), ACL_MEMCPY_DEVICE_TO_HOST);
    (void)aclrtMemcpy(scaleRow0, sizeof(scaleRow0), scaleDevice, sizeof(scaleRow0), ACL_MEMCPY_DEVICE_TO_HOST);
    printf("y[0..7] =");
    for (uint8_t b : yFirst8) {
        printf(" %02x", b);
    }
    printf("\nscale[0][0..3] =");
    for (uint8_t b : scaleRow0) {
        printf(" %02x", b);
    }
    printf("\n");

    // 6. 释放资源
    aclDestroyTensor(x);
    aclDestroyTensor(tokens);
    aclDestroyTensor(y);
    aclDestroyTensor(yScale);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtFree(xDevice);
    aclrtFree(tokensDevice);
    aclrtFree(yDevice);
    aclrtFree(scaleDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
