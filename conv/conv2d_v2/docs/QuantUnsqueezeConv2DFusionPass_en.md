# QuantUnsqueezeConv2DFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:44:30.806Z pushedAt=2026-11-16T06:28:26.395Z nl=2 -->

## Description

Restructures the AscendQuant+Unsqueeze+Conv2D+Squeeze+AscendDequant structure into Unsqueeze+AscendQuant+Conv2D+AscendDequant+Squeeze.

![QuantUnsqueezeConv2DFusionPass fusion diagram](../../../docs/en/figures/QuantUnsqueezeConv2DFusionPass_1.png)

## Constraints

- The input node of Conv2D must be Unsqueeze, and the input node of Unsqueeze must be AscendQuant.
- The output node of Conv2D must be Squeeze, and the output node of Squeeze must be AscendDequant.
- The AscendQuant, Unsqueeze, Conv2D, Squeeze, and AscendDequant nodes must not carry any control edge.
- The input dimension of Unsqueeze must be 3.

## Applicable Products

<!-- npu="950" id1 -->
Ascend 950PR&950DT products
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3 products
<!-- end id2 -->
