# DeformableOffsetsGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>      |    √     |
| <term>Atlas A2 products</term>       |    √     |

## Function

- Operator function: The DeformableOffsets operator collects the feature sampling points used for convolution according to offsets (offset values of the kernel sampling points) and reorganizes them, so that the Conv2d operator can perform the convolution computation conveniently. DeformableOffsetsGrad is the backward pass of this process: it takes as input the gradient `grad` with respect to the forward output `y` of DeformableOffsets obtained by convolution backpropagation, combines the forward input feature `x` and the offset values `offsets`, and computes and outputs the gradient of `x` (`grad_x`) and the gradient of `offsets` (`grad_offsets`).

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>The gradient with respect to the forward output y of DeformableOffsets. Its shape is [N, outH*K_H, outW*K_W, C], where outH/outW are calculated by the output size formula in the constraints section, and C is the same as the channel count of x.</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>The forward input feature (original input data) of DeformableOffsets. Its shape is [N, inH, inW, C].</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Offset values of the kernel sampling points. Its shape is [N, outH, outW, deformable_groups*K_H*K_W*3], and each sampling point contains three components: the x offset, the y offset, and the mask (when modulated=true). outH/outW are calculated by the output size formula in the constraints section.</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>Output</td>
      <td>The gradient of x. Its shape is the same as that of x: [N, inH, inW, C].</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>grad_offsets</td>
      <td>Output</td>
      <td>The gradient with respect to the input offsets. Its shape is the same as that of offsets: [N, outH, outW, deformable_groups*K_H*K_W*3].</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Required attribute</td>
      <td>Strides of the sliding window, a list containing 4 integers. When data_format is "NHWC", the value is (1, stridesH, stridesW, 1); when it is "NCHW", the value is (1, 1, stridesH, stridesW). stridesH/stridesW cannot be 0, and strides[0] of the N dimension must be 1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Required attribute</td>
      <td>Padding dimensions, a list containing 4 integers: [padTop, padBottom, padLeft, padRight].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Required attribute</td>
      <td>Size of the convolution kernel, a list containing 2 integers: (K_H, K_W).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>Optional attribute</td>
      <td>Scaling factor of each dimension, a list containing 4 integers. The default value is [1, 1, 1, 1]. dilations[0] of the N dimension must be 1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Data format. It only supports "NCHW" or "NHWC". The default value registered in OpDef is "NHWC", but the default value registered in the operator IR (proto) is "NCHW"; the two are inconsistent. Do not rely on the default value when invoking through GE graph mode; specify this attribute explicitly. This attribute only determines the subscript interpretation order of the strides/dilations list; the storage format of the input tensor is fixed to NHWC.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>Optional attribute</td>
      <td>Number of groups on the C axis. The default value is 1. It must be greater than 0 and must be equal to the offsets channel count /(K_H*K_W*3) (when modulated=true, each sampling point contains 3 offset components).</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>Optional attribute</td>
      <td>Boolean value specifying the version of DeformableConv2D. true indicates v2, and false indicates v1. The default value is true. Currently only true is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of the inputs `grad`, `x`, and `offsets` must be 4-D.
- **Empty tensors are not supported**: when the total number of elements of any of `grad`, `x`, and `offsets` is 0 (any dimension is 0), the operator rejects execution directly at the Tiling stage and reports an error through the log; no empty output is produced.
- The storage format of the input tensors is fixed to NHWC.
- The lengths of `strides`, `pads`, and `dilations` must be 4, and the length of `ksize` must be 2.
- The H/W of `offsets` is equal to the outH/outW calculated by the output size formula, respectively.
- The H/W of `grad` must be equal to the H/W of `offsets` multiplied by the corresponding elements of `ksize`, respectively. The H/W of the output `grad_offsets` must be consistent with the H/W of `offsets`.
- The batch dimension N of `grad`, `x`, `offsets`, and `grad_offsets` must be consistent.
- The outH/outW calculated by the output size formula below must be greater than 0. The channel dimension C of `x` must be greater than 0.
- Index path and capacity: when the total number of elements of any of `grad`/`x`/`offsets` does not exceed the INT32 upper limit (2147483647), the int32 index kernel is used; when any of them exceeds the limit, the int64 index kernel is automatically used. The maximum number of elements of a single tensor is subject to the device memory capacity.
- Inputs that violate any of the above constraints are rejected at the graph compilation/Tiling stage and reported through the error log. This operator has no independent public error code.

**Output size formula** (consistent with the [operator IR](./op_graph/deformable_offsets_grad_proto.h) definition, where `//` denotes integer division):

$$
outH = (inH + pads[0] + pads[1] - (dilationsH * (K_H - 1) + 1)) // stridesH + 1
$$

$$
outW = (inW + pads[2] + pads[3] - (dilationsW * (K_W - 1) + 1)) // stridesW + 1
$$

## Invocation Description

| Invocation method | Sample code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| GE graph mode | [test_geir_deformable_offsets_grad](./examples/arch35/test_geir_deformable_offsets_grad.cpp)   | Calls the DeformableOffsetsGrad operator by building a graph through the [operator IR](./op_graph/deformable_offsets_grad_proto.h). |

- The operator has registered InferShape/InferDataType inference callbacks (the shape/dtype of `grad_x` follows `x`, and `grad_offsets` follows `offsets`) and enabled dynamic shape/dynamic rank support, so it can be invoked through dynamic graph construction in GE.
- This operator has no public aclnn API entry (the build configuration is `ACLNNTYPE aclnn_exclude`). The L0-level interfaces delivered in the `op_api/` directory are only for internal cascading reuse of the operator and are not publicly committed.
