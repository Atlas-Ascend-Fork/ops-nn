# RmsNormFusionPass

## 融合模式

- 场景一：将平方、均值归约、加epsilon、归一化及gamma乘法融合为RmsNorm。输入Cast、gamma的Reshape及rstd输出为可选。

  ![rms_norm_fusion_pass_1](../../../docs/zh/figures/rms_norm_fusion_pass_1.png)

  ![rms_norm_fusion_pass_2](../../../docs/zh/figures/rms_norm_fusion_pass_2.png)

- 场景二：当gamma不满足直接融合条件，或归一化后存在低精度转换时，保留外部Mul及必要的Cast、除法节点。

  ![rms_norm_fusion_pass_3](../../../docs/zh/figures/rms_norm_fusion_pass_3.png)

  ![rms_norm_fusion_pass_4](../../../docs/zh/figures/rms_norm_fusion_pass_4.png)

- 场景三：支持ReduceMean归约尾维为动态维度的场景，使用运行时shape构造RmsNorm所需的单位gamma。

  ![rms_norm_fusion_pass_5](../../../docs/zh/figures/rms_norm_fusion_pass_5.png)

## 使用约束

- 数据类型约束：

  | 场景 | 归一化计算类型 | 与gamma相乘前的数据类型 | gamma的数据类型 |
  |:-:|:-:|:-:|:-:|
  | 场景一 | FLOAT32 | FLOAT32 | FLOAT32 |
  | 场景二、三 | FLOAT32 | FLOAT32、FLOAT16、BFLOAT16 | FLOAT32、FLOAT16、BFLOAT16 |

- Cast算子约束：
  - 平方、归约和归一化使用FLOAT32计算。输入升精度的Cast可由平方和归一化两条支路共用，也可分别使用同源、同目标类型的Cast。
  - 归一化后转为FLOAT16或BFLOAT16的低精度Cast须在融合后保留。不支持倒数平方根先降精度再参与归一化。

- shape和归约约束：
  - 仅支持ND格式，输入x的rank为1～8，不支持空张量。归约轴须由INT32或INT64常量指定，等价于连续尾轴，且keep\_dims为true。
  - ReduceMean后不接缩放节点；ReduceSum后必须乘以1/N或除以N，N为归约元素数，x须为静态shape。
  - gamma的rank须已知且满足原图的广播规则。直接输入RmsNorm时，其shape须静态并与归约尾维一致；含前导单维时可通过Reshape适配，对应的shape常量与Reshape同时存在。其他合法广播保留外部Mul。

- 算子和输出约束：
  - 平方支持Square、相同张量相乘的Mul及指数为2的Pow。图1、3、5中，Rsqrt或Pow(-0.5)后不接倒数节点；Sqrt或Pow(0.5)后必须接Reciprocal或Div(1,RMS)。Div也支持RealDiv，分子1及Pow指数均须精确匹配。
  - epsilon仅支持Add的常量输入，shape为标量、单元素Tensor或各维均为1的Tensor；常量类型支持FLOAT16、BFLOAT16、FLOAT32，数值须有限、非负且可用FLOAT32精确表示，不支持正次正规值和动态epsilon。
  - 必须先归一化再与gamma相乘。不支持省略gamma、调整乘法结合顺序，以及输出直接或经Cast后连接偏置加法或残差加法。
  - 输出支持y，或同时输出y和FLOAT32类型的rstd。图2的倒数支路仅用于原图输出rstd；除rstd输出外，不支持内部统计量被其他分支复用。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
