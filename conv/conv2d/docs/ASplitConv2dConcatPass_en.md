# ASplitConv2dConcatPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:36:48.105Z pushedAt=2026-11-16T06:28:26.310Z nl=2 -->

## Description

Fuses Split/SplitV2+Conv2d \*N+Concat/ConcatV2 into a group convolution (GroupConv2d) to simplify the graph structure.

![ASplitConv2dConcatPass fusion diagram](../../../docs/en/figures/ASplitConv2dConcatPass_1.png)

## Constraints

- The number of inputs of each Conv2d must be the same and greater than or equal to 2.
- The shape and format of the filter of each Conv2d must be the same (only HWCN or NCHW is supported).
- Each Conv2d can have only a single output, and the output node must be the same Concat/ConcatV2 node.
- The filter and bias of Conv2d must be of one of the following types: {"AscendWeightQuant", "Const", "Constant", "QuantBiasOptimization", "QuantBiasRollBack", "QuantWeightRollBack"}.
- The dim input of the Split/SplitV2 node and the concat_dim input of the Concat/ConcatV2 node must be of const type, and the number of outputs of the Split/SplitV2 node must be the same as the number of inputs of Concat/ConcatV2.
- The split axis of the Split/SplitV2 node and the concat axis of the Concat/ConcatV2 node must both be in the channel direction. The values of the split axis and the concat axis must be equal. When the axis value is -1, only the NHWC format is supported.
- The formats of the Split data input, the Concat output, and the Conv2d input must be consistent.
- Each output branch of Split can have only one node, and the node must be Conv2d.
- The filter and bias input data types of Conv2d support only float, float16, int8, and int32.

## Applicable Products

<!-- npu="310p" id1 -->
Atlas inference products
<!-- end id1 -->

<!-- npu="310b" id2 -->
Atlas 200I/500 A2 inference products
<!-- end id2 -->

<!-- npu="910" id3 -->
Atlas training products
<!-- end id3 -->

<!-- npu="950" id4 -->
Ascend 950PR&950DT products
<!-- end id4 -->
