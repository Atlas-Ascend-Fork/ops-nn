# SituGluGroupQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>                                 |    ×     |
| <term>Atlas A2系列产品</term>                                 |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                        |    ×     |
| <term>Atlas推理系列产品</term>                                 |    ×     |
| <term>Atlas训练系列产品</term>                                 |    ×     |

## 功能说明

- 算子功能：面向MoE分组场景，将稀疏SiTU门控激活与MXFP8（e4m3fn，blocksize=32）尾轴动态量化融合为一个算子。输入`x`最后一维为拼接的gate/up两半（2\*EPR）；激活结果驻留片上不回写GM，随后按MX算法量化为`y`(FLOAT8_E4M3FN) + `y_scale`(FLOAT8_E8M0)。激活部分与两个小算子串联`npu_situ_and_mul_sparse -> npu_dynamic_mx_quant(dst=e4m3fn)`在有效行内逐位一致。
- 激活公式（gate/up的选取方向由属性`activate_left`决定）：

  $$
  g = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \frac{1}{1+e^{-gate}},\qquad
  u = linear\_beta \cdot \tanh\left(\frac{up}{linear\_beta}\right)
  $$

  $$
  act = g \times u
  $$

  全程在fp32域计算（单次退出舍入），无中间低精度回退；可选输入`weight`存在时按行缩放：$act = fp32(act) \times w[row]$。

- 量化说明：沿`axis`（当前仅 -1轴）每32个元素为一个MX block，从block内最大绝对值Amax的指数域导出FLOAT8_E8M0缩放因子s（OCP MX语义），block内每个元素除以s后按`round_mode`舍入并饱和到FLOAT8_E4M3FN范围，得到y。特殊值语义：激活结果中的NaN元素逐个置为+0（不整组清零）；±Inf及有限溢出钳位到输入类型的最大有限值后参与量化；全NaN组的Amax为0，整组y=+0、y_scale=0。组数为奇数时y_scale每行末尾有1个补位字节不写出，调用方不应读取。
- 分组语义（`group_index`存在时）：group_index第i个元素为专家i分到的token行数（count模式），kernel逐元素累加得到有效行数validRows = min(sum(group_index), totalRows)，仅处理前validRows行，其余padding行不写出；`group_index`不传时为全行模式（所有行参与计算）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待激活量化的数据，最后一维为拼接的gate/up两半（2*EPR），最后一维必须为正偶数。零长维度退化为0行计算，不拒收。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>每个token（行）的缩放因子，元素个数与x展开后的总行数一致，按全局行号索引。存在时激活结果在fp32域按行缩放后再量化；不传则不缩放。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>每个专家分到的token行数。不传 = 全行模式；传入 = 分组稀疏语义（validRows = min(sum, totalRows)，padding行不写出）。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>属性</td>
      <td>gate支路的SiTU激活参数，必须大于0。</td>
      <td>double</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>属性</td>
      <td>up支路软限幅参数：≤0禁用（up直通），>0启用（幅度钳位 ±linear_beta）。默认0.0（禁用）。</td>
      <td>double</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td>gate/up方向选择：true时gate取x前半部分（移植前原始方向）；false（默认）时gate取x后半部分。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>量化轴，当前仅支持 -1（尾轴）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>量化模式，当前仅支持0（MX量化）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>目标量化数据类型，当前仅支持36（FLOAT8_E4M3FN）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>属性</td>
      <td>目标数据类型数值上限，取值 >= 0，当前不参与计算。</td>
      <td>double</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>属性</td>
      <td>舍入模式，当前仅支持rint。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>属性</td>
      <td>group_index语义模式：0 = cumsum模式（预留，传入会被拒收）；1 = count模式（当前语义，必须显式传入）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化结果，shape = x.shape[:-1] + [EPR]。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>输出</td>
      <td>MX量化缩放因子，shape = x.shape[:-1] + [ceil(ceil(EPR/32)/2), 2]，按每32个元素一组与y对应。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x.shape[-1]必须为正偶数；EPR = x.shape[-1] / 2必须为32的倍数且在[128, 3072]区间内（tiling防呆校验拒收，越界场景对应原算子已知缺陷：EPR &lt; 128时y_scale静默丢失、EPR &gt; 3072时整行拷贝越界踩踏UB、EPR % 32 != 0时尾组垃圾）。
- beta > 0。linear_beta ≤ 0禁用up支路软限幅（up直通），> 0启用，任意取值合法。
- x支持1维及以上（无8维上限校验），不拒空Tensor（零长维度退化为0行，与移植前原算子口径一致）；数据格式支持ND/NCL/NCHW/NCDHW（连续存储视同ND）。
- **group_list_type默认值为0（cumsum预留模式，会被拒收），不传即报错，必须显式传1（count模式）方可调用。**

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn调用 | [test_aclnn_situ_glu_group_quant](./examples/arch35/test_aclnn_situ_glu_group_quant.cpp) | 通过aclnnSituGluGroupQuant接口方式调用SituGluGroupQuant算子，接口签名与参数详见[aclnn_situ_glu_group_quant.h](./op_host/op_api/aclnn_situ_glu_group_quant.h)。 |
| 图模式（C++ GE API） | [test_geir_situ_glu_group_quant](./examples/arch35/test_geir_situ_glu_group_quant.cpp) | 通过op_graph proto建图 + GE图编译执行，覆盖分组/zero-token/padding场景。 |
| torchAPI调用 | [situ_glu_group_quant](./torch_extension/situ_glu_group_quant.py) | 通过torchAPI方式调用SituGluGroupQuant算子，安装torch扩展whl后可直接运行，使用说明见[torchapi_situ_glu_group_quant](./docs/torchapi_situ_glu_group_quant.md)。 |

构建 / 测试 / TTK golden命令：

```bash
bash build.sh -u --ophost   --ops=situ_glu_group_quant --soc=ascend950   # host UT
bash build.sh -u --opkernel --ops=situ_glu_group_quant --soc=ascend950   # kernel UT
bash build.sh --opkernel    --ops=situ_glu_group_quant --soc=ascend950   # kernel binary
bash build.sh --pkg         --ops=situ_glu_group_quant --soc=ascend950   # 出包
bash build.sh --run_example situ_glu_group_quant eager cust --soc=ascend950  # 上板 example（aclnn eager）
bash build.sh --run_example situ_glu_group_quant graph --soc=ascend950       # 上板 example（C++ GE 图模式）
bash build.sh --torch_extension --ops=situ_glu_group_quant                   # torch 扩展 whl
