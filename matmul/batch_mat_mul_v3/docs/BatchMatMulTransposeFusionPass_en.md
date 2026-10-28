# BatchMatMulTransposeFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:55:11.708Z pushedAt=2026-10-13T01:47:29.724Z -->

## Fusion Pattern

This fusion absorbs the Transpose operator that swaps only the last two dimensions on the MatMul input side into the MatMul operator: the Transpose node is deleted, and the transpose attribute of the corresponding MatMul input is flipped (BatchMatMul/BatchMatMulV2 flips adj_x1/adj_x2, and MatMul/MatMulV2 flips transpose_x1/transpose_x2).

![](../../../docs/en/figures/BatchMatMulTransposeFusionPass_1.png)

## Constraints

- Transpose1 and Transpose2 can both exist, or only one of them can exist.
- The Transpose type includes only Transpose.
- The Transpose node swaps only the last two dimensions of the input. For example, if the input shape of the Transpose node is [B,M,K], its output shape is [B,K,M].
- MatMul types include BatchMatMul/BatchMatMulV2/MatMul/MatMulV2.
- The input dtype of the MatMul node supports only FLOAT16, FLOAT32, and BFLOAT16.

## Applicable Products

<!-- npu="950" id1 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id1 -->
