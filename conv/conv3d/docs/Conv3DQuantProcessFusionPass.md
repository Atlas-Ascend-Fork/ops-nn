# Conv3DQuantProcessFusionPass

## 融合模式

该融合规则处理AscendQuant（可隔Pad节点）+Conv3D+AscendDequant结构，根据输入通道数与Conv3D算子对数据类型的支持情况，选择以下两种模式之一：

### Bias优化模式

输入通道数大于fp16的C0（通常为16）或Conv3D不支持float16时，保留量化计算结构，新增QuantBiasOptimization host cpu算子：将bias与AscendDequant的deq_scale、AscendQuant的offset/scale联合折算为int32量化bias；AscendQuant的offset写入Conv3D的offset_x属性；无bias输入时自动补充全零bias。Conv3D保持int8量化输入。

![Bias优化模式](../../../docs/zh/figures/Conv3DQuantProcessFusionPass_1.png)

### 量化回退模式

输入通道数不大于fp16的C0且Conv3D支持float16时，删除AscendQuant与AscendDequant节点，weight与bias分别经QuantWeightRollBack、QuantBiasRollBack host cpu算子按scale预乘回退为float16数据，Conv3D及其输入输出整链调整为float16。

![量化回退模式](../../../docs/zh/figures/Conv3DQuantProcessFusionPass_2.png)

## 使用约束

- Conv3D的输入节点必须为AscendQuant，AscendQuant与Conv3D之间可以存在一个Pad节点。
- Conv3D的输出下游必须存在AscendDequant节点。
- AscendQuant、Pad、Conv3D、AscendDequant候选节点融合后不能构成环。
- Bias优化模式下，AscendDequant的deq_scale输入必须为常量且为一维Tensor。
- AscendQuant输入的C维为未知（动态shape）时跳过融合。

## 支持的型号

<!-- npu="310p" id1 -->
Atlas推理系列产品
<!-- end id1 -->

<!-- npu="310b" id2 -->
Atlas 200I/500 A2推理产品
<!-- end id2 -->

<!-- npu="910" id3 -->
Atlas训练系列产品
<!-- end id3 -->

<!-- npu="910b" id4 -->
Atlas A2系列产品
<!-- end id4 -->

<!-- npu="950" id5 -->
Ascend 950PR&950DT系列产品
<!-- end id5 -->
