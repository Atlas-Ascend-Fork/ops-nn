# InplaceApplyAdadelta

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：执行Adadelta优化器的单步参数更新。根据当前梯度、梯度平方累积状态accum和更新量平方累积状态accum_update计算参数更新量。三个输出var、accum和accum_update分别与同名输入共享Device内存（inplace），实现原地更新。

- 计算公式：

  $$
      \begin{aligned}
      accum_{t} &= \rho \cdot accum_{t-1} + (1 - \rho) \cdot grad^2 \\
      update &= \frac{\sqrt{accum\_update_{t-1} + \epsilon}}{\sqrt{accum_{t} + \epsilon}} \cdot grad \\
      var_{t} &= var_{t-1} - lr \cdot update \\
      accum\_update_{t} &= \rho \cdot accum\_update_{t-1} + (1 - \rho) \cdot update^2
      \end{aligned}
  $$

  其中 `rho` 为衰减系数（数值直接参与计算，典型取值为 0.9），`epsilon` 为数值稳定常数（通常取正值），`lr` 为学习率。

## 参数说明

<table style="table-layout: fixed; width: 1430px">
  <colgroup>
    <col style="width: 127px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>var</td>
      <td>输入</td>
      <td>权重参数，对应公式中的var。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>梯度平方累积状态，对应公式中的accum。shape/dtype必须与var一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update</td>
      <td>输入</td>
      <td>更新量平方累积状态，对应公式中的accum_update。shape/dtype必须与var一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率，对应公式中的lr。Rank 0 scalar Tensor（shape={}），dtype必须与var一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rho</td>
      <td>输入</td>
      <td>衰减系数，对应公式中的rho。Rank 0 scalar Tensor（shape={}），数值直接参与计算。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>数值稳定常数，对应公式中的epsilon。Rank 0 scalar Tensor（shape={}）；通常取正值，非正值按公式计算。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>当前梯度Tensor，对应公式中的grad。shape/dtype必须与var一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>仅保留接口兼容性，与A2老算子ApplyAdadeltaD的实现保持兼容一致性。默认false，不参与计算流程或并发锁控制。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>var(output)</td>
      <td>输出</td>
      <td>更新后的权重参数，与输入var共享Device内存（inplace）。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum(output)</td>
      <td>输出</td>
      <td>更新后的梯度平方累积状态，与输入accum共享Device内存（inplace）。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update (output)</td>
      <td>输出</td>
      <td>更新后的更新量平方累积状态，与输入accum_update共享Device内存（inplace）。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- var/accum/accum_update/grad四者shape必须完全一致（ND 格式）。
- 输出var、accum和accum_update分别与同名输入共享Device内存（inplace）。
- GE图模式支持静态shape、未知维度（`-1`）和未知Rank（`-2`）构图；运行时四个主张量的具体shape须一致，三个输出采用对应输入的具体shape。
- lr/rho/epsilon必须为Rank 0 scalar Tensor（shape `{}`，物理存储为1个元素），dtype与var一致。
- 仅支持FLOAT/FLOAT16；FP16中间结果保存为half，accum和var更新中的乘加使用Fma并在加法后舍入为half；开方和倒数开方使用FP32暂存，结果转回half。
- FP32路径采用DAZ/FTZ数值语义：var、accum、accum_update、grad以及lr、rho、epsilon中的有限次正规数在参与计算前转为同符号的零；各FP32算术步骤产生的有限次正规结果在进入下一步前同样转为同符号的零。判定条件为`|x| < 1.17549435e-38`，恰好等于该阈值的最小正规数保留。NaN和Inf不由DAZ/FTZ处理改写，后续结果按具体浮点运算传播。
- 支持合法同形的空Tensor。当var、accum、accum_update和grad的shape完全一致且包含零长度维时（例如`[0]`或`[0, 3]`），算子成功返回；var、accum和accum_update三个输出分别保持对应输入的shape、dtype及inplace关系，不执行元素计算。lr、rho和epsilon仍须为Rank 0 scalar Tensor。

## 调用说明

| 调用方式   | 样例代码                                                | 说明                            |
| :--------- | :------------------------------------------------------ | :------------------------------ |
| GE图模式 | [test_geir_inplace_apply_adadelta](./examples/test_geir_inplace_apply_adadelta.cpp) | 通过 [算子IR](./op_graph/inplace_apply_adadelta_proto.h) 构图方式调用InplaceApplyAdadelta算子；此样例演示静态shape。 |
