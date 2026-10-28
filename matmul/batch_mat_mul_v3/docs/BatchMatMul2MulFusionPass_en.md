# BatchMatMul2MulFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:54:20.986Z pushedAt=2026-10-13T01:47:29.719Z -->

## Fusion Pattern

Some MatMul/MatMulV2/MatMulV3/BatchMatmul/BatchMatMulV2/BatchMatMulV3 operators with k=1 in a network deliver poor performance. This graph fusion converts MatMul/MatMulv2/MatMulV3/BatchMatmul/BatchMatMulV2/BatchMatMulV3 to mul to resolve the performance issue.

>[!NOTE]
><term>Ascend 950PR&950DT products</term> fuse only MatMul/MatMulV2/MatMulV3 nodes and do not fuse BatchMatmul/BatchMatMulV2/BatchMatMulV3 nodes.

![](../../../docs/en/figures/BatchMatMul2MulFusionPass_1.png)

If the corresponding adj of an input is true, insert a reshape operator before the input.

![](../../../docs/en/figures/BatchMatMul2MulFusionPass_2.png)

## Constraints

- Applicable only to static use cases. Inputs do not contain bias.
- DType must meet the following conditions:
  - Both input and output are Float32.
  - Both input and output are Float16 or BFloat16.

## Applicable Products

<!-- npu="910b" id1 -->
<term>Atlas A2 products</term>
<!-- end id1 -->

<!-- npu="A3" id2 -->
<term>Atlas A3 products</term>
<!-- end id2 -->

<!-- npu="950" id3 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id3 -->
