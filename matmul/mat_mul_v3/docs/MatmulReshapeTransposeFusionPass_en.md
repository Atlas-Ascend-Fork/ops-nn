# MatmulReshapeTransposeFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:59:49.414Z pushedAt=2026-10-13T01:47:29.744Z -->

## Fusion Pattern

This fusion deletes the Transpose operators before and after the Matmul operator that matches the graph fusion pattern, and swaps the two inputs of the Matmul operator.

The structure in the dashed box can appear once or multiple times.

![](../../../docs/en/figures/MatmulReshapeTransposeFusionPass_1.png)

## Constraints

- The operator dtype supports only FP32.
- The format supports only ND/NCHW/NHWC.
- The left matrix input of the Matmul operator must be 2D, the right matrix input must be 3D, and the last dimension of the right matrix shape must be 1.
- The inputs of all Transpose operators must be 3D, with the last dimension of the shape being 1. They are used to swap the first two dimensions of the shape. Dynamic shapes are not supported.
- The Reshape operator before the Matmul operator deletes the last dimension of value 1 from the shape. The Reshape operator after the Matmul operator adds a dimension of value 1 to the end of the shape.
- All attributes transpose\_x1/transpose\_x2 of the Matmul operator must be false.
- After the first Reshape operator, at least one path with the Matmul+Reshape+Transpose structure exists. Multiple Matmul+Reshape+Transpose paths are supported. Each path must satisfy the preceding structure; otherwise, the fusion is not performed.

## Applicable Products

<!-- npu="310b" id1 -->
Atlas 200I/500 A2 inference products
<!-- end id1 -->

<!-- npu="910b" id2 -->
Atlas A2 products
<!-- end id2 -->

<!-- npu="A3" id3 -->
Atlas A3 products
<!-- end id3 -->

<!-- npu="950" id4 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id4 -->
