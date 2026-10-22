# AddRmsNormFusionGraphPass

## 融合模式

该融合将符合图融合pattern的Add算子和RmsNorm算子融合成融合算子AddRmsNorm，其中Add算子的输出作为RmsNorm算子的第一个输入。

![](../../../docs/zh/figures/add_rms_norm_fusion_graph_pass_1.png)

## 使用约束

- Add算子的输入X1和X2的shape/dtype要一致。
- format只支持ND。

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
