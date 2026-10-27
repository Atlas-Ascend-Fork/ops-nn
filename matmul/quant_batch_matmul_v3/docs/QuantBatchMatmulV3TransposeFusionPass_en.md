# QuantBatchMatmulV3TransposeFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:10:14.145Z pushedAt=2026-10-13T01:47:29.772Z -->

## Fusion Patterns

This fusion rule folds the explicit transpose in the `x1` or `x2` input branch of `QuantBatchMatmulV3` into the matmul computation, and can also process both branches at the same time. The recognized transpose node types are `Transpose` and `TransposeD`. When fusing the `x1` branch, invert the `transpose_x1` attribute of the `QuantBatchMatmulV3` node; when fusing the `x2` branch, invert the `transpose_x2` attribute of the node. Inverting means changing `false` to `true`, or `true` to `false`.

In this document, "bypass" refers to the following connection adjustment: connect the input of a conversion node directly to the successor node of that conversion node, so that the current branch no longer passes through the conversion node. After the bypass, keep the conversion node if it still has other data consumers; otherwise, delete it. The successor node can be `QuantBatchMatmulV3` or the retained `Bitcast`.

The following figures assume that the [Input and Transpose Requirements](#input-and-transpose-requirements) and [Dimension and Platform Restrictions](#dimension-and-platform-constraints) are met. The variables in the figures are described as follows:

- **Input Mapping**: Shows `x1`, `x2`, and the two `scale` inputs respectively. `x1_scale` denotes the input `pertoken_scale` (IR index 5), and `x2_scale` denotes the input `scale` (IR index 2); these two names are aliases used in the figures, not new operator parameters or operator attributes.
- **Symbol Definition**: `t1` and `t2` denote the values of the `transpose_x1` and `transpose_x2` attributes of `QuantBatchMatmulV3` before fusion, respectively; `!` denotes logical inversion.
- **Source Tensor**: A source tensor with the same name before and after fusion refers to the same tensor.
- **Omitted Content**: The figures omit `bias`, `offset`, the input specifying the transpose dimension order, and the input specifying the shape after reshape, and show only the data connections relevant to fusion.

### Fusion-Only x1 Side Transpose

When there exists one transpose node in each of the `x1` branch and the `x1_scale` branch, bypass respectively the two transpose nodes, and connect their inputs directly to the `x1` and `pertoken_scale` input ports of `QuantBatchMatmulV3`. Meanwhile, invert the `transpose_x1` attribute of the `QuantBatchMatmulV3` node. The connections of the `x2` and `scale` inputs and the `transpose_x2` attribute of the node remain unchanged.

![Fusion-only x1 and x1_scale transposes, four-way input independently connected](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_1.png)

### Fusion-only x2 Side Transpose

When there exists one transpose node each in the `x2` branch and the `x2_scale` branch, bypass the two transpose nodes respectively, and connect their inputs directly to the `x2` and `scale` input ports of `QuantBatchMatmulV3`. Also, invert the `transpose_x2` attribute of the `QuantBatchMatmulV3` node. Keep the connections of the `x1` and `pertoken_scale` inputs and the `transpose_x1` attribute of the node unchanged.

![Fusion-only x2 and x2_scale transposes, four-way input connected independently](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_2.png)

### Simultaneously Fusing Transposes on Both Sides

When there exists one transpose node in each of the four branches of `x1`, `x2`, `x1_scale`, and `x2_scale`, and both the `x1` and `x2` sides meet the usage constraints listed at the beginning of this section, the four transpose nodes are bypassed respectively. The inputs of the four transpose nodes connect to the `x1`, `x2`, `pertoken_scale`, and `scale` input ports of `QuantBatchMatmulV3`, respectively; the `transpose_x1` and `transpose_x2` attributes of the `QuantBatchMatmulV3` node are both inverted. Matching only the structure in the figure is insufficient to trigger fusion. Constraints such as data type, dimension, and platform capability must also be satisfied.

![Four transposes of two data inputs and two scale inputs fused respectively](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_3.png)

### Reshape Fusion for Normal Scale

When there exists one transpose node in each of the `x1` and `x2` input branches, and one `Reshape` equivalent to the required transpose in each of the two `scale` branches, bypass the two transpose nodes and the two `Reshape` nodes respectively. The four branches connect to their corresponding `QuantBatchMatmulV3` input ports, and the `transpose_x1` and `transpose_x2` attributes of the `QuantBatchMatmulV3` node are both inverted. For identification requirements of static normal `scale`, see [Reshape Requirements for scale](#reshape-requirements-for-scale). The figure shows the case where both sides can be fused. You can also fuse only the side that satisfies the constraints.

![Fusion of the Reshapes of two normal scale branches with their corresponding data transposes](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_4.png)

### Reshape Fusion of MX Scale

Taking the `x2` side as an example: the `x2` input is preceded by `Transpose` or `TransposeD`, `x2_scale` is of the `FLOAT8_E8M0` type, and its `Reshape` changes (1, N, 2) to (N, 1, 2). During fusion, the transpose node in the `x2` branch and the `Reshape` in the `x2_scale` branch are bypassed, and their inputs are connected to the `x2` and `scale` input ports of `QuantBatchMatmulV3` respectively; the `transpose_x2` attribute of the `QuantBatchMatmulV3` node is inverted. The connections of `x1` and `x1_scale` remain unchanged. The same processing logic applies when the `x1` side meets the corresponding conditions.

![MX scenario showing x1, x2, the two scales, and the 3D Reshape respectively](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_5.png)

### Transpose Fusion Retaining Bitcast

`Bitcast` reinterprets the bit representation of data as the target data type. When each of the four-way inputs is connected as "transpose node → `Bitcast` → `QuantBatchMatmulV3`", the fusion directly connects the inputs of the four transpose nodes to the subsequent `Bitcast` respectively, retains the four `Bitcast` nodes and their connections to `QuantBatchMatmulV3`, and performs attribute inversion on the `transpose_x1` and `transpose_x2` attributes of the `QuantBatchMatmulV3` node. After fusion, the data type of the `Bitcast` output remains unchanged.

![Fusion of transposes on four-way inputs with Bitcast retained respectively](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_6.png)

The input branches are not required to contain `Bitcast` at the same time. A transpose node can connect to the matmul directly or through a `Bitcast`; fusion across multiple consecutive `Bitcast` nodes is not supported. For the `scale` branch, the node before the `Bitcast` can also be a `Reshape` that meets the requirements described below. For platform restrictions and the applicability boundary of dynamic `Reshape`, see the usage constraints.

### Reshape Fusion for Dynamic Scale

If `x1` or `x2` contains unknown dimensions, handle it as a dynamic scenario. Taking the `x2` side as an example, the fusion bypasses the transpose node on the `x2` branch and the `Reshape` on the `x2_scale` branch, connects their inputs to the `x2` and `scale` input ports of `QuantBatchMatmulV3` respectively, and will invert the `transpose_x2` attribute of the `QuantBatchMatmulV3` node. `x1` and `x1_scale` keep their original connections. The dynamic scenario does not perform the dimension condition check for a static `Reshape`, but the input graph must still ensure transpose equivalence between the `Reshape` and the required `scale`.

![Four-way inputs shown separately in the dynamic scenario, with only the x2 side transpose and Reshape processed](../../../docs/en/figures/QuantBatchMatmulV3TransposeFusionPass_7.png)

The `scale` conversion nodes in each figure are the specific structures of the illustrated scenario. A conversion node is not required on every `scale` branch. A `scale` branch is processed only when the `Transpose` or `TransposeD` of the corresponding data input is fused; when `pertoken_scale` is not connected, that input is not processed. A bypassed node is retained if it still has other data consumers; otherwise, it is removed from the graph.

## Constraints

### Input and Transpose Requirements

- The original `shape` of both `x1` and `x2` must have at least 2 dimensions. At least one data input must connect to `Transpose` or `TransposeD`; otherwise, fusion is not triggered. A data input preceded only by `Reshape` does not trigger fusion.
- The data types of both `x1` and `x2` support `INT8`, `INT4`, `FLOAT8_E4M3FN`, `FLOAT8_E5M2`, `HIFLOAT8`, and `FLOAT4_E2M1`. The output data types support `INT8`, `FLOAT16`, `BF16`, `INT32`, and `FLOAT32`. The actual combinations must also meet the operator constraints of the target product.
- Dynamic `shape` is supported. If the `shape` of `x1` or `x2` contains unknown dimensions, it is processed as a dynamic scenario.
- The transpose of a data input must only swap the last two dimensions, with the order of other dimensions unchanged. For example, an input of shape `(B, M, K)` is transposed to `(B, K, M)`. Other transposes, such as swapping the batch dimension, are out of the scope of this rule. The corresponding `scale` conversion must also maintain the correspondence between quantization coefficients and data. For details, see the next section.
- `pertoken_scale` is an optional input. If it is not connected, the corresponding `scale` is not processed.

### Restrictions in the Bitcast Scenario

On some platforms, the matrix multiplication instruction supports mixed INT8 and INT4 inputs. On such platforms, if one of the four-way inputs matches the structure in the following table, this rule retains the entire `QuantBatchMatmulV3` node and all of its input branches, and does not perform this transpose fusion. This restriction applies even if the current node does not use mixed INT8 and INT4 inputs.

| Input Branch | Connection Structure That Triggers the Preceding Constraint |
| ---- | ---- |
| `x1`, `x2`, or either of the two `scale` branches | `Transpose` or `TransposeD` → `Bitcast` → `QuantBatchMatmulV3` |
| Either of the two `scale` branches | `Reshape` that meets the fusion requirements in the next section → `Bitcast` → `QuantBatchMatmulV3` |

When only `Bitcast` is present and no preceding conversion nodes described above exist, fusion is not skipped due to this restriction.

There is also a recognition restriction when a dynamic `Reshape` is used in combination with `Bitcast`: if only the `scale` branch contains a "dynamic `Reshape` → `Bitcast` → matrix multiplication" structure and no other branch contains a recognizable structure with `Bitcast` described above, the dynamic `Reshape` is not guaranteed to be fused. Therefore, the dynamic `Reshape` examples in this document show only connections that do not go through `Bitcast`.

### Reshape Requirements for scale

The `Reshape` before the `scale` input must meet the conditions in the following table. In a static scenario, the input graph must also provide the shapes and data types of the `Reshape` input and output, and the corresponding `QuantBatchMatmulV3` input must have at least 2 dimensions.

| Scenario | Recognition Condition |
| ---- | ---- |
| Condition A: static general condition | At least one of the last two dimensions of the `shape` of the corresponding `QuantBatchMatmulV3` input is 1. This check does not restrict the data type of `scale`. |
| Condition B: static MX condition | The data type is `FLOAT8_E8M0`. The `Reshape` input and output are both 3-dimensional, with the first two dimensions swapped and at least one of them being 1. For example, (1, N, 2) changes to (N, 1, 2). |
| Dynamic `shape` | The preceding static shape conditions are not required, but the `Reshape` must still be equivalent to the required `scale` transpose. |

In a static scenario, meeting either Condition A or Condition B in the preceding table is sufficient. Condition B refers to swapping the first two dimensions of a 3-dimensional `FLOAT8_E8M0` `scale`. If Condition B is not met, the pattern can still be recognized through Condition A. For a dynamic scenario, follow the "Dynamic `shape`" row in the preceding table. In the MX example in the figure, the last dimension 2 comes from the quantization layout of that example and is not a fixed value additionally required by Condition B. After the shape conditions in the table are met, the `Reshape` must still be equivalent to the required `scale` transpose and must not change the correspondence between quantization coefficients and data. This rule does not separately delete nodes such as `Shape`, `Gather`, and `Pack` that are used to compute dynamic shapes.

### Dimension and Platform Constraints

`QuantBatchMatmulV3TransposeFusionPass` is the general transpose fusion rule, and `QuantBatchMatmulV3TransposeLimitFusionPass` is a supplementary rule for specific dimension scenarios. The two rules are registered separately. Each merges eligible explicit transposes into matrix multiplication computation. Fusion is implemented by adjusting input connections and inverting the corresponding transpose attributes of `QuantBatchMatmulV3`. They differ in the applicable products and dimension conditions. The Limit rule still provides transpose fusion processing for branches that meet its conditions even when the general rule is disabled.

Let `outer` denote the penultimate dimension of the original `shape` of the data input corresponding to `QuantBatchMatmulV3` before fusion, and `inner` denote the last dimension. Both rules check `x1` and `x2` respectively, and only the side that satisfies the conditions may be fused. The processing conditions of the products listed in this document are as follows. The actual fusion must also satisfy the preceding usage constraints, such as input, transpose node, and `scale` requirements.

| Product | General Rule | Limit Supplementary Rule |
| ---- | ---- | ---- |
| <term>Ascend 950PR&950DT products</term>  | `x1` and `x2` are not subject to the 65535 dimension threshold in this section. Transpose fusion is performed when other usage constraints are met. | This supplementary rule is skipped, and transpose fusion is handled by the general rule. |
| Atlas A2 products and Atlas A3 products | `x1` requires `outer` ≤ 65535; `x2` requires `outer` ≤ 65535 or the `FRACTAL_NZ` format. | Checks `x1` and `x2` respectively; processes only branches that meet 0 < `outer` ≤ 65535 and `inner` > 65535. |

The dimension conditions of the Limit rule also apply to the `FRACTAL_NZ` format; they are not relaxed due to the format. If the `outer` or `inner` value involved in the check is the unknown value `-1`, the branch does not satisfy the dimension conditions of the Limit rule.

### Version Requirements

- The graph compiler version number must be no less than 90100000 at both compile time and run time.
- If the compile-time version does not meet the requirement, the two transpose rules are not enabled. If the run-time version does not meet the requirement, both rules keep the input graph unchanged.

## Applicable Products

See [QuantBatchMatmulV3 Operator Description](../README.md) for the data types and formats supported by the product. This rule determines fusion conditions based on platform instruction capabilities. Operator support for a product does not mean that the product supports all fusion modes of this rule.

<!-- npu="910b" id1 -->
Atlas A2 products
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3 products
<!-- end id2 -->

<!-- npu="950" id3 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id3 -->
