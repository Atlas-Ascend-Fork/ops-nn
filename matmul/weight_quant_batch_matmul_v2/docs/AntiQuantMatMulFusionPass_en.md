# AntiQuantMatMulFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:51:50.366Z pushedAt=2026-10-13T01:47:29.713Z -->

## Fusion Patterns

<!-- npu="950,910b" id1 -->
This fusion combines AscendAntiQuant (with optional Add and Mul) and MatMul/BatchMatMul into the WeightQuantBatchMatmulV2 operator. Through constant folding, the fp16 antiquant_scale and antiquant_offset are precomputed at compile time, eliminating the runtime AntiQuant+Add+Mul computation. The following three fusion modes are supported.

Fusion pattern 1: AscendAntiQuant + Add + Mul + MatMul scenario. The output of AscendAntiQuant passes through Add (adding a constant offset) and Mul (multiplying a constant scale) before entering MatMul. After the fusion, the constant offset of Add and the constant scale of Mul are folded with the scale/offset attributes of AscendAntiQuant into antiquant_scale and antiquant_offset. See the figure below.

![](../../../docs/en/figures/AntiQuantMatMulFusionPass_1.png)

Fusion pattern 2: AscendAntiQuant + Mul + MatMul scenario. The output of AscendAntiQuant passes through Mul (multiplying a constant scale) before entering MatMul, and no Add node exists. After the fusion, the constant scale of Mul is folded with the scale attribute of AscendAntiQuant into antiquant_scale, and antiquant_offset defaults to 0. See the figure below.

![](../../../docs/en/figures/AntiQuantMatMulFusionPass_2.png)

Fusion pattern 3: AscendAntiQuant + MatMul scenario. The output of AscendAntiQuant directly enters MatMul, and no Add or Mul node exists. After the fusion, antiquant_scale defaults to 1, and antiquant_offset defaults to 0. See the figure below.

![](../../../docs/en/figures/AntiQuantMatMulFusionPass_3.png)

>[!NOTE] Note
>Constant folding formula: antiquant_scale = scale_data * anti_scale, antiquant_offset = offset_data / anti_scale + anti_offset. anti_scale and anti_offset are attributes of AscendAntiQuant, and scale_data and offset_data are the constant inputs of Mul and Add respectively. If the Add node is absent, antiquant_offset defaults to 0. If the Mul node is absent, antiquant_scale defaults to 1.

The products supported by this fusion are as follows.

<!-- npu="910b" id2 -->
Atlas A2 products
<!-- end id2 -->

<!-- npu="950" id3 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id3 -->
<!-- end id1 -->

## Constraints

- Supported MatMul operator types: MatMul, MatMulV2, BatchMatMul, and BatchMatMulV2.
- The outputs of the AscendAntiQuant, Add, and Mul nodes can be connected to only one downstream node; otherwise, fusion is not triggered.
- The constant inputs of the Add and Mul nodes must be Const nodes.
- The data types of the MatMul inputs x and weight must be FLOAT16, and the output data type must be FLOAT16.
- The input data type of AscendAntiQuant must be INT8, and the output data type must be FLOAT16.
- The input shapes of the MatMul and AscendAntiQuant nodes must be 2D. Dynamic shapes are not supported.
- The constant scale of Mul supports only 1 or N elements (the N dimension of weight), and the constant offset of Add supports only 1 or N elements.
- The shapes of the Add and Mul nodes must be the same: (1,n), (n,), or (1,). In the scenario where Add does not exist but Mul exists, the shape of the Mul node must be (1,).
<!-- npu="910b" -->
- For Atlas A2 products, the shape admission criteria must be met: M<=64, K>=5120, N>=5120, and (K,N) cannot be (5120,10240) or (10240,5120).
<!-- end -->
<!-- npu="950" -->
- For <term>Ascend 950PR&950DT products</term>, there is no shape admission restriction.
<!-- end -->
- The transpose_x1/transpose_x2 attributes of MatMul (or the adj_x1/adj_x2 attributes of BatchMatMul) are passed to the transpose_x/transpose_weight attributes of the fused WeightQuantBatchMatmulV2.
- The optional bias input is supported.
