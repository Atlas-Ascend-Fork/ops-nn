# Conv2DSqueezeBiasaddFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:38:41.261Z pushedAt=2026-11-16T06:28:26.337Z nl=2 -->

## Description

Converts the Conv2D+Squeeze+BiasAdd structure into Conv2D+BiasAdd+Squeeze.

![Conv2DSqueezeBiasaddFusionPass fusion diagram](../../../docs/en/figures/Conv2DSqueezeBiasaddFusionPass_1.png)

## Constraints

- The input node of the Squeeze node must be Conv2D, and the output of the Squeeze node must be BiasAdd.
- The bias input dimension of the BiasAdd node must be 1.
- If the bias input of the BiasAdd node comes from a Variable node, no fusion is performed.
- Both inputs of the BiasAdd node must have static shapes. Otherwise, no fusion is performed.

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
