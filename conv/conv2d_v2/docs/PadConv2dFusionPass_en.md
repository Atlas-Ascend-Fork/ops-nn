# PadConv2dFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:42:46.658Z pushedAt=2026-11-16T06:28:26.376Z nl=2 -->

## Description

Fuses the Pad/PadV3+Conv2d operators into a Conv2d operator.

The forward fusion use case applies only to the Pad/PadV3+Conv2d graphs.
<!-- npu="A3,910b,910,310p,310b" id6 -->
The backward fusion use case applies only to the backward process corresponding to the forward use case in a training network. When Pad+Conv2DBackpropFilterD is encountered, it is fused into a new Conv2DBackpropFilterD to eliminate Pad. When Conv2DBackpropInputD+Slice is encountered, it is fused into a new Conv2DBackpropInputD to eliminate Slice. The products applicable to this use case are as follows:
<!-- npu="310p" id8 -->
- Atlas inference products
<!-- end id8 -->
<!-- npu="310b" id9 -->
- Atlas 200I/500 A2 inference products
<!-- end id9 -->
<!-- npu="910" id10 -->
- Atlas training products
<!-- end id10 -->
<!-- npu="910b" id11 -->
- Atlas A2 products
<!-- end id11 -->
<!-- npu="A3" id12 -->
- Atlas A3 products
<!-- end id12 -->
<!-- end id6 -->

![PadConv2dFusionPass fusion pattern 1](../../../docs/en/figures/PadConv2dFusionPass_1.png)

## Constraints

- Dynamic shapes are not supported.
- The Pad/PadV3 operator connected to multiple Conv2d structures is not supported. In the structure before fusion, the first node is connected only to the next node, not to multiple nodes.
- The control edge of the Pad/PadV3 node cannot be connected to the fused Cube node (Conv2d/Conv2DBackpropFilterD).
- `paddings` cannot be smaller than 0.
- The PadV3 operator supports fusion only when mode is constant and constant\_values is 0 (dtype is fp32).
- The paddings input of the Pad/PadV3 operator must be a Const node, and the data type supports only int32 or int64.
- The x input format of the Pad/PadV3 operator supports only NCHW or NHWC.
- The N/C dimension padding of the Pad/PadV3 operator supports only 0. Padding is supported on the H/W dimensions of Conv2d, and the fused pad size must be within the range [0, 255].
<!-- npu="A3,910b,910,310p,310b" id13 -->
- After fusion, both pad_top and pad_bottom must be less than kernel_h. This constraint applies to the following product models:
    <!-- npu="310p" id14 -->
    - Atlas inference products
    <!-- end id14 -->
    <!-- npu="310b" id15 -->
    - Atlas 200I/500 A2 inference products
    <!-- end id15 -->
    <!-- npu="910" id16 -->
    - Atlas training products
    <!-- end id16 -->
    <!-- npu="910b" id17 -->
    - Atlas A2 products
    <!-- end id17 -->
    <!-- npu="A3" id18 -->
    - Atlas A3 products
    <!-- end id18 -->
<!-- end id13 -->

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

<!-- npu="910b" id4 -->
Atlas A2 products
<!-- end id4 -->

<!-- npu="A3" id5 -->
Atlas A3 products
<!-- end id5 -->

<!-- npu="950" id7 -->
Ascend 950PR&950DT products
<!-- end id7 -->
