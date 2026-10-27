# MatMulBiasAddOpenFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:58:22.432Z pushedAt=2026-10-13T01:47:29.739Z -->

## Fusion Patterns

Fuses the MatMul/MatMulV2/BatchMatMul/BatchMatMulV2 operators with the BiasAdd/Add operator into the MatMul/MatMulV2/BatchMatMulV2 operator.

![](../../../docs/en/figures/MatMulBiasAddOpenFusionPass_1.png)

## Constraints

- The output of a MatMul node can connect to only one downstream operator.
- Only one-dimensional bias is supported: Of the two inputs of the Add operator, the input other than the MatMul output (that is, bias) must be one-dimensional. Multi-dimensional bias is not supported.
- The size of bias must be the same as that of the last dimension of the MatMul or MatMulV2/BatchMatMul/BatchMatMulV2 output. Broadcast is not supported.
- The output dimension of MatMul/MatMulV2 is 2.
- If the input is BatchMatMul, it is converted to BatchMatMulV2 after fusion.
- If the BiasAdd/Add computation before fusion is FLOAT16, bias is added in FLOAT32 inside MatMul after fusion, improving precision.

## Applicable Products

<!-- npu="950" id1 -->

<term>Ascend 950PR/Ascend 950DT products</term>

<!-- end id1 -->
