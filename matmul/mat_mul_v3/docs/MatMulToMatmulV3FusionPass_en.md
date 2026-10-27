# MatMulToMatmulV3FusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:00:43.382Z pushedAt=2026-10-13T01:47:29.746Z -->

## Fusion Pattern

This fusion converts MatMulV2/MatMul and BatchMatMulV2/BatchMatMul operators that match the graph fusion pattern into MatMulV3/BatchMatMulV3 operators.

![](../../../docs/en/figures/MatMulToMatmulV3FusionPass_1.png)

## Constraints

- Data types supported for the inputs x1 and x2: FLOAT16, BFLOAT16, FLOAT32, and INT8.
- When one of x1 and x2 is INT8 and the other is FLOAT16, BFLOAT16, or FLOAT32, insert a Cast operator on the INT8 side to convert the INT8 input to the data type of the other side, and then feed the converted input into MatMulV3/BatchMatMulV3. The data type of output y is the same as that of the other side.

## Applicable Products

<!-- npu="950" id1 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id1 -->
