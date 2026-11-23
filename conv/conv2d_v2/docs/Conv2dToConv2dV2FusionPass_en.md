# Conv2dToConv2dV2FusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:38:39.408Z pushedAt=2026-11-16T06:28:26.334Z nl=2 -->

## Description

Converts the Conv2d operator into the Conv2dv2 operator. The fused Conv2dv2 operator has exactly the same input, output, and attributes as the original Conv2d operator.

![Conv2dToConv2dV2FusionPass fusion pattern](../../../docs/en/figures/Conv2dToConv2dV2FusionPass_1.png)

## Constraints

The input data type only supports Fp32, BF16, FP16, HIFloat8 (in the HIFloat8 case, bias is Fp32).

## Applicable Products

<!-- npu="950" id1 -->
Ascend 950PR&950DT products
<!-- end id1 -->
