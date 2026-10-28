# GemmToMatmulFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:56:09.937Z pushedAt=2026-10-13T01:47:29.730Z -->

## Fusion Pattern

Since the Gemm API has no single-operator implementation, this fusion splits Gemm operators that match the graph fusion pattern into combinations of matmul, mul, and add operators.

![](../../../docs/en/figures/GemmToMatmulFusionPass_1.png)

## Constraints

- The dtypes of inputs a and b support FLOAT16, FLOAT32, and INT32 (INT32 does not support dynamic shapes).
- The dtype of output c supports FLOAT16, FLOAT32, and INT32.
- Special use case: inputs a and b are INT8, and the output is FLOAT32.
- This fusion rule cannot be disabled.

## Applicable Products

<!-- npu="950" id1 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id1 -->
