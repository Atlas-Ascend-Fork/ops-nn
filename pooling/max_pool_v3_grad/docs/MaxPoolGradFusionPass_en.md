# MaxPoolGradFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:04:39.532Z pushedAt=2026-10-13T01:47:29.757Z -->

## Fusion Pattern

<!-- npu="950" id1 -->
The MaxPoolGrad operator in the network has no corresponding operator binary implementation on supported products. This fusion rule replaces the eligible MaxPoolGrad operators in the graph with MaxPoolV3Grad operators as a whole. The inputs and outputs remain in one-to-one correspondence, and attributes are rewritten with equivalent semantics. The attribute conversion rules used by MaxPoolGradFusionPass are the same as those used by MaxPoolFusionPass when converting MaxPool to MaxPoolV3. See the figure below.

![](../../../docs/en/figures/MaxPoolGradFusionPass_1.png)

<!-- end id1 -->

The input and output correspondence before and after fusion is as follows.

| **MaxPoolGrad Before Fusion** | **MaxPoolV3Grad After Fusion** | **Required/Optional** | **Description** |
| --- | --- | --- | --- |
| x1 (input 0) | orig_input (input 0) | Required | Original input of forward pooling. |
| x2 (input 1) | orig_output (input 1) | Required | Original output of forward pooling. |
| grad (input 2) | grad (input 2) | Required | Gradient passed in for backward propagation. |
| y (output 0) | out_grad (output 0) | Required | Gradient output for backward propagation. The output description is inherited from the node before fusion. |

The attribute correspondence before and after fusion is as follows. Attributes marked as Required must exist on the MaxPoolGrad node before fusion and meet the value requirements in the description. Otherwise, fusion is not performed.

| **MaxPoolV3Grad After Fusion** | **Value Source** | **Required/Optional** | **Description** |
| --- | --- | --- | --- |
| ksize | ksize attribute of MaxPoolGrad | Required | Both the attribute name and attribute value are passed through as-is from MaxPoolGrad to MaxPoolV3Grad. The number of elements must be 4. |
| strides | strides attribute of MaxPoolGrad | Required | Both the attribute name and attribute value are passed through as-is from MaxPoolGrad to MaxPoolV3Grad. The number of elements must be 4. |
| padding_mode | padding attribute of MaxPoolGrad | Required | Converted from the padding attribute of MaxPoolGrad: the attribute name is changed to padding_mode, and the attribute value remains unchanged. Only SAME and VALID are supported for padding. Fusion is not performed for other values. |
| pads | Fixed value `{0, 0, 0, 0}` | - | This attribute is absent in MaxPoolGrad and is added during fusion. It takes no effect when padding_mode is SAME or VALID. |
| data_format | data_format attribute of MaxPoolGrad | Required | Both the attribute name and attribute value are passed through as-is from MaxPoolGrad to MaxPoolV3Grad. Only NCHW and NHWC are supported. |
| global_pooling | Fixed value `false` | - | This attribute is absent in MaxPoolGrad and is added during fusion, set to non-global pooling. |
| ceil_mode | Fixed value `false` | - | This attribute is absent in MaxPoolGrad and is added during fusion, set to floor mode. |

> [!NOTE]
>
> The data_format attribute of the MaxPoolGrad node takes precedence. The physical Format in the node input description may be ND. Data layout is not determined based on this physical Format, and fusion is not rejected because the Format is ND.

## Constraints

- Fusion is performed in the stage after InferShape. The matched source operator type is MaxPoolGrad, and the target operator type after replacement is MaxPoolV3Grad.
- The data types supported by the full pipeline are FLOAT16, FLOAT32, and BFLOAT16.
- The matched graph structure is a single MaxPoolGrad node with three inputs (x1, x2, grad) and one output (y). The pattern does not match if the number of inputs or outputs does not meet the requirement.
- This fusion rule is enabled by default and currently cannot be disabled.

## Applicable Products

<!-- npu="950" id2 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id2 -->
