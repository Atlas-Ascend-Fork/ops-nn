# Conv2DFixPipeToExtendConv2DFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:38:34.204Z pushedAt=2026-11-16T06:28:26.332Z nl=2 -->

## Description

Fuses Conv2D+FixPipe into ExtendConv2D.

### Single-Output Fusion

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 1](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_1.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 2](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_2.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 3](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_3.png)

### Dual-Output Fusion

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 4](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_4.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 5](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_5.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 6](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_6.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 7](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_7.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 8](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_8.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 9](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_9.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 10](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_10.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 11](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_11.png)

![Conv2DFixPipeToExtendConv2DFusionPass fusion pattern 12](../../../docs/en/figures/Conv2DFixPipeToExtendConv2DFusionPass_12.png)

## Constraints

- Only the following Conv2D input/output Dtypes combinations are supported: int8 input with int32 output, or fp16 input with fp16 output.
- Only NCHW or NHWC is supported for Conv2D input/output Formats.

## Applicable Products

<!-- npu="950" id1 -->
Ascend 950PR&950DT products
<!-- end id1 -->
