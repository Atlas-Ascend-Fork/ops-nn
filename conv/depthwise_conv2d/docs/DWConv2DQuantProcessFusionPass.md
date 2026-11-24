# DWConv2DQuantProcessFusionPass

## 融合模式

该融合规则处理AscendQuant（可隔Pad节点）+DepthwiseConv2D+AscendDequant结构，根据平台能力与输入通道数，选择以下两种模式之一：

### Bias优化模式

平台支持DepthwiseConv2D量化实现且输入通道数大于fp16的C0（通常为16）或DepthwiseConv2D不支持float16时，保留量化计算结构，新增QuantBiasOptimization host cpu算子：将bias与AscendDequant的deq_scale、AscendQuant的offset/scale联合折算为int32量化bias；AscendQuant的offset写入DepthwiseConv2D的offset_x属性；无bias输入时自动补充全零bias。DepthwiseConv2D保持int8量化输入。

![Bias优化模式](../../../docs/zh/figures/DWConv2DQuantProcessFusionPass_1.png)

### 量化回退模式

平台不支持DepthwiseConv2D量化实现，或输入通道数不大于fp16的C0且DepthwiseConv2D支持float16时，删除AscendQuant与AscendDequant节点，weight与bias分别经QuantWeightRollBack、QuantBiasRollBack host cpu算子按scale预乘回退为float16数据，DepthwiseConv2D及其输入输出整链调整为float16。

![量化回退模式](../../../docs/zh/figures/DWConv2DQuantProcessFusionPass_2.png)

## 使用约束

- DepthwiseConv2D的输入节点必须为AscendQuant，AscendQuant与DepthwiseConv2D之间可以存在一个Pad节点。
- DepthwiseConv2D的输出下游必须存在AscendDequant节点。
- AscendQuant、Pad、DepthwiseConv2D、AscendDequant候选节点融合后不能构成环。
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
