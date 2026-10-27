# MatmulToGemmOpFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:02:09.273Z pushedAt=2026-10-13T01:47:29.749Z -->

## Fusion Patterns

<!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>: This fusion will perform operator conversion of MatMulV3/MatMulV2/MatMul operators that conform to graph fusion patterns into GemmV2 operators.
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: This fusion will perform operator conversion of MatMulV3/MatMulV2/MatMul operators that conform to graph fusion patterns into GemmV2 operators.
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR&950DT products</term>: This fusion will perform operator conversion of MatMulV3/MatMulV2/MatMul operators that conform to graph fusion patterns into GemmV3 operators.
<!-- end id3 -->

![](../../../docs/en/figures/MatmulToGemmOpFusionPass_1.png)

## Constraints

- The Cast node between the MatMul and AssignAdd nodes can be absent; matching without the Cast node is supported.
- The MatMul types include MatMul/MatMulV2/MatMulV3.
- Input dtypes of the MatMul node:
  <!-- npu="910b" id4 -->
  - <term>Atlas A2 products</term>: Only supports float16 and bfloat16.
  <!-- end id4 -->
  <!-- npu="A3" id5 -->
  - <term>Atlas A3 products</term>: Only supports float16 and bfloat16.
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT products</term>: Supports float16, float32, and bfloat16.
  <!-- end id6 -->
- Input shapes of the MatMul node:
  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>: Only supports shapes in the whitelist for fusion.
  <!-- end id7 -->
  <!-- npu="A3" id8 -->
  - <term>Atlas A3 products</term>: Only supports shapes in the whitelist for fusion.
  <!-- end id8 -->
- The input dtype of the AssignAdd node only supports float32.
- Disabling this pass is not recommended. Doing so may affect network accuracy.

## Applicable Products

<!-- npu="910b" id9 -->
Atlas A2 products
<!-- end id9 -->

<!-- npu="A3" id10 -->
Atlas A3 products
<!-- end id10 -->

<!-- npu="950" id11 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id11 -->
