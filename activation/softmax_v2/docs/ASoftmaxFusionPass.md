# ASoftmaxFusionPass

## 融合模式

该规则处理带有`need_fusion`标记的SoftmaxV2节点，将输入从指定轴开始的后缀维度展平为二维张量，在二维张量的最后一维执行SoftmaxV2，再恢复原始shape。该转换用于需要按后缀维度展平计算的Softmax场景。仓库中的[ONNX导入插件](../framework/softmax_onnx_plugin.cpp)对注册的ONNX 8–12版本Softmax设置`need_fusion=1`，默认轴为1；注册的13–18版本导入逻辑不设置该标记，默认轴为-1。

- 转换前：

  ![带need_fusion标记的SoftmaxV2](../../../docs/zh/figures/a_softmax_fusion_pass_1.png)

  **转换为**

  ![Reshape、SoftmaxV2、Reshape替换子图](../../../docs/zh/figures/a_softmax_fusion_pass_2.png)

设输入shape为`[d0, ..., d(r-1)]`，归一化后的轴为`a`，则第一个Reshape的目标shape为`[M, N]`，其中`M`为第0维至第`a-1`维的乘积，`N`为第`a`维至最后一维的乘积；当`a=0`时，`M=1`。新SoftmaxV2的`axes`为`[1]`，第二个Reshape恢复原始输入shape。图中的Const_0为`[M, N]`，Const_1为原始输入shape，分别作为Reshape_0和Reshape_1的shape输入。

例如，输入shape为`[2, 3, 4]`、`axes=[1]`时，替换子图为`Reshape([2, 12]) → SoftmaxV2(axes=[1]) → Reshape([2, 3, 4])`。

新SoftmaxV2保留原节点的`half_to_float`属性，正常读取原节点输出描述时，输出的数据类型和格式沿用该描述；`half_to_float=true`且原输出为FLOAT32时，恢复shape仍使用FLOAT32。若原输出描述读取失败，当前实现回退到输入的数据类型和格式。新节点不再设置`need_fusion`标记。

## 使用约束

- 输入约束：
  - 原SoftmaxV2必须具有非零的整型`need_fusion`属性；属性缺失或值为0时不转换。
  - `need_fusion`标记应由前序导入或处理流程设置，不能对普通单轴Softmax任意添加，因为展平后计算的是整个后缀维度上的Softmax。

- 数据格式和shape约束：
  - 输入维数必须大于等于3。
  - 使用时应提供静态shape。动态shape不会被统一过滤，可能导致替换图的shape推导失败。
  - 输入数据类型和格式需满足SoftmaxV2与Reshape本身的约束。

- axes参数约束：
  - `axes`属性必须存在且非空，使用时应提供单个合法轴，不支持多轴或越界轴。
  - 负轴通过加上输入维数归一化；归一化后的轴为最后一维时跳过转换。

- 替换图必须能够完成shape推导；推导失败时转换失败。

## 支持的型号

<!-- npu="A3" id1 -->
<term>Atlas A3系列产品</term>
<!-- end id1 -->

<!-- npu="950" id2 -->
<term>Ascend 950PR&950DT系列产品</term>
<!-- end id2 -->
