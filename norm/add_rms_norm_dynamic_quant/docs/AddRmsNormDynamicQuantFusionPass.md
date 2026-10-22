# AddRmsNormDynamicQuantFusionPass

## 融合模式

该融合将符合图融合pattern的AddRmsNorm算子和DynamicQuant算子融合成融合算子AddRmsNormDynamicQuant，其中AddRmsNorm算子的输出y作为DynamicQuant算子的第一个输入。

场景1：单路模式

![](../../../docs/zh/figures/add_rms_norm_dynamic_quant_fusion_pass_1.png)

场景2：双路模式

![](../../../docs/zh/figures/add_rms_norm_dynamic_quant_fusion_pass_2.png)

## 使用约束

- 融合前AddRmsNorm和DynamicQuant算子输入类型需要保持一致（即全是fp16或者bf16）。
  <!-- npu="A3,910b" id4 -->
- DynamicQuant的输出yout的约束：
  <!-- npu="910b" id5 -->
  - Atlas A2系列产品：yout仅支持量化类型为int8。
  <!-- end id5 -->
  <!-- npu="A3" id6 -->
  - Atlas A3系列产品：yout仅支持量化类型为int8。
  <!-- end id6 -->
  <!-- end id4 -->

- dtype约束：有平滑系数场景下，DynamicQuant的smooth\_scales dtype须和AddRmsNorm的x1的dtype相同。

- shape约束：输入gamma的shape必须为1维，并且shape取值和x1，x2输入shape的尾轴相同，即：gamma.shape = \[x1.shape\[-1\]\]。

- attr约束：
  <!-- npu="950" id7 -->
  - Ascend 950PR&950DT系列产品：dst\_type仅支持DT\_INT8、DT\_HIFLOAT8、DT\_FLOAT8\_E4M3FN、DT\_FLOAT8\_E5M2。
  <!-- end id7 -->
  <!-- npu="910b" id8 -->
  - Atlas A2系列产品：dst\_type仅支持DT\_INT8。
  <!-- end id8 -->
  <!-- npu="A3" id9 -->
  - Atlas A3系列产品：dst\_type仅支持DT\_INT8。
  <!-- end id9 -->
  - 双路模式下，DynamicQuant0和DynamicQuant1的dst\_type须相同。

## 支持的型号

<!-- npu="910b" id1 -->
Atlas A2系列产品
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3系列产品
<!-- end id2 -->

<!-- npu="950" id3 -->
Ascend 950PR&950DT系列产品
<!-- end id3 -->
