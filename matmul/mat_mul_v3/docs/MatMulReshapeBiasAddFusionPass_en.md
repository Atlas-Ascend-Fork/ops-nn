# MatMulReshapeBiasAddFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:59:44.409Z pushedAt=2026-10-13T01:47:29.742Z -->

## Fusion Pattern

This fusion rule fuses the MatMul/MatMulV2 → Reshape → BiasAdd/Add structure into MatMul/MatMulV2 with a bias input (has_bias=true): Delete the BiasAdd/Add node, and connect the bias directly to MatMul/MatMulV2. Keep the Reshape node and move it after MatMul/MatMulV2. The output of Reshape is then directly connected to downstream nodes.

![](../../../docs/en/figures/MatMulReshapeBiasAddFusionPass_1.png)

## Constraints

- MatMul/MatMulV2 supports only two inputs.
- The outputs of both the MatMul node and the Reshape node can be connected to only one downstream operator.
- The inputs of BiasAdd/Add must be a bias and the output of Reshape.
- bias dimensions: The bias of the BiasAdd branch must be one-dimensional, and its total length must be equal to the last dimension of the Reshape output. The bias of the Add branch can be one-dimensional or multi-dimensional, with each dimension equal to the corresponding trailing dimension of the Reshape output (broadcast is not supported), and its total length must be equal to the size of the trailing axis of the MatMul output.
- The Reshape node cannot split the trailing axis of the MatMul output.
- The output of MatMul must be 2-dimensional, and empty tensors are not supported.
- The output dtype of Reshape must be the same as the output dtype of BiasAdd/Add.
- Only static graph mode is supported.

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
