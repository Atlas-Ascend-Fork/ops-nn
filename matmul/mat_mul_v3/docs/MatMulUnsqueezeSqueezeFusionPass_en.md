# MatMulUnsqueezeSqueezeFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:02:34.172Z pushedAt=2026-10-13T01:47:29.752Z -->

## Fusion Pattern

**Fusion Pattern 1**

For MatMul/MatMulV2/BatchMatMul/BatchMatMulV2 inputs of 1 dim x N dim and N dim x 1 dim, insert an Unsqueeze operator before the 1 dim input to expand it to 2D, and insert a Squeeze operator after the output to remove the corresponding expanded axis.

![](../../../docs/en/figures/MatMulUnsqueezeSqueezeFusionPass_1.png)

**Fusion Pattern 2**

For MatMul/MatMulV2/BatchMatMul/BatchMatMulV2 inputs of 1 dim x 1 dim, insert an Unsqueeze operator before the 1 dim input to expand it to 2D.

![](../../../docs/en/figures/MatMulUnsqueezeSqueezeFusionPass_2.png)

**Fusion Pattern 3**

MatMul/MatMulV2/BatchMatMul/BatchMatMulV2 support the AscendDequant use case. The 1 dim input expansion is the same as in Fusion Mode 1/Fusion Mode 2, and the Squeeze operator is inserted after the AscendDequant operator.

![](../../../docs/en/figures/MatMulUnsqueezeSqueezeFusionPass_3.png)

## Constraints

- At least one of x1 and x2 has a dimension of 1.
- In the AscendDequant use case, the output of MatMul can be connected to only one downstream operator, which must be AscendDequant.

## Applicable Products

<!-- npu="910b" id1 -->

Atlas A2 products

<!-- end id1 -->

<!-- npu="A3" id2 -->

Atlas A3 products

<!-- end id2 -->

<!-- npu="950" id3 -->

<term>Ascend 950PR&950DT products</term>

<!-- end id3 -->
