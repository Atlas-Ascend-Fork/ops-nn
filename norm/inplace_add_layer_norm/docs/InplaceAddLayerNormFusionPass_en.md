# InplaceAddLayerNormFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:57:20.373Z pushedAt=2026-10-13T01:47:29.735Z -->

## Fusion Pattern

<!-- npu="950,A3,910b" id1 -->
Fusion Mode 1: Replace the AddLayerNorm operator in the graph with the InplaceAddLayerNorm operator so that the layer normalization result and the addition result are written back in-place to x1 and x2 respectively, as shown in the figure below.

The products supported by this fusion mode are as follows.

<!-- npu="910b" id2 -->
<term>Atlas A2 products</term>
<!-- end id2 -->

<!-- npu="A3" id3 -->
<term>Atlas A3 products</term>
<!-- end id3 -->

<!-- npu="950" id4 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id4 -->

![](../../../docs/en/figures/InplaceAddLayerNormFusionPass_1.png)

<!-- end id1 -->

## Constraints

- Only inference use cases are supported.
- The bias of the AddLayerNorm operator is an optional input. Use cases with and without bias are supported.
- x1 and x2 must meet the in-place write-back safety requirements: they must not be consumed by any node other than the current AddLayerNorm operator, and must not come from multi-output nodes with contiguous memory layout.
- In static shape use cases, the memory size occupied by x1 must be greater than or equal to 1/8 of the L2 cache capacity and less than or equal to twice the L2 cache capacity.
- In dynamic shape use cases, the last dimension of x1 must be 5120.
- The x1, x2, and gamma inputs of the AddLayerNorm operator must have the same data type.
