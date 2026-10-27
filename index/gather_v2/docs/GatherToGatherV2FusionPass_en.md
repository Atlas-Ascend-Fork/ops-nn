# GatherToGatherV2FusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:55:46.482Z pushedAt=2026-10-13T01:47:29.727Z -->

## Fusion Pattern

This fusion changes a Gather operator that matches the graph fusion pattern to a GatherV2 operator.

![](../../../docs/en/figures/GatherToGatherV2FusionPass_1.png)

## Constraints

This fusion rule cannot be disabled.

## Applicable Products

<!-- npu="950" id1 -->
Ascend 950PR&950DT products
<!-- end id1 -->
