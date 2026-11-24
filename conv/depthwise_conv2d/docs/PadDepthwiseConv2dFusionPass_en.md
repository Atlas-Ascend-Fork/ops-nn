# PadDepthwiseConv2dFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:43:04.867Z pushedAt=2026-11-16T06:28:26.379Z nl=2 -->

## Description

Fuses the Pad and DepthwiseConv2D operators into a DepthwiseConv2D operator.

![PadDepthwiseConv2dFusionPass fusion diagram](../../../docs/en/figures/PadDepthwiseConv2dFusionPass_1.png)

## Constraints

- Pad does not support dynamic shape input or dynamic shape output.
- Before fusion, DepthwiseConv2D must have the padding attribute, and the value must be VALID.
- In the Pad output, the number of DepthwiseConv2D output nodes must be 1, and all output nodes of the Pad node must be DepthwiseConv2D.
- The paddings input of Pad must be a Const node, and the data type supports only int32 or int64.
- The x input format of the Pad operator supports only NCHW or NHWC.
- The Pad operator supports padding only in the H/W dimensions of DepthwiseConv2D. The padding in the N/C dimensions must be 0.
- The fused pad size must be within the range [0, 255].
- The Pad node cannot carry control edges.
<!-- npu="A3,910b,910,310p,310b" id6 -->
- After fusion, both pad_top and pad_bottom are smaller than kernel_h. This constraint applies to the following products:
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
