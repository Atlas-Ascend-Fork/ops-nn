# LayerNormRemoveBroadcastFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:58:19.228Z pushedAt=2026-10-13T01:47:29.737Z -->

## Fusion Patterns

<!-- npu="950" id1 -->
Fusion mode 1: On Ascend 950PR/950DT products, the BroadcastTo before the gamma and beta inputs of the LayerNorm operator is removed from the graph. gamma and beta are directly used as inputs of the LayerNorm operator, and begin_norm_axis and begin_params_axis are both set to the start axis of the normalization dimension. The following figure shows an example.

![](../../../docs/en/figures/LayerNormRemoveBroadcastFusionPass_1.png)

<!-- end id1 -->

## Constraints

- Only supported if BroadcastTo exists before both the gamma and beta inputs of the LayerNorm operator and the two BroadcastTo operators have the same shape input.
- Only static shapes are supported for the inputs x, gamma, and beta.
- Scalars are not supported for gamma and beta, and gamma and beta must have the same shape and data type.
- The number of dimensions of gamma cannot be greater than that of x, and the shape of gamma must be the same as the suffix shape of x starting from the normalization axis.
- After begin_norm_axis of the LayerNorm operator is converted to a non-negative axis, it must point to the start axis of the normalization dimension.
