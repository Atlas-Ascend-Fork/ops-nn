# Centralization

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function

For each element of `x`, Centralization subtracts the corresponding mean computed over the axes specified by `axes`.

$$
y = x - \operatorname{mean}(x, \mathrm{axes})
$$

## Parameters

| Name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| `x` | Input | Tensor to center, with rank 1 to 8. | FLOAT, FLOAT16 | ND |
| `y` | Output | Centered tensor with the same shape and data type as `x`. | FLOAT, FLOAT16 | ND |
| `axes` | Attribute (optional in GE graphs) | List of reduction axes. Each axis must be in `[-rank(x), rank(x))`. Repeated axes count once. GE graphs use the last axis by default when this attribute is omitted. | LIST_INT | - |

When `axes=[]` is passed explicitly, the output is zero for each finite input element and NaN for each NaN or infinite input element.

## Constraints

- <term>Ascend 950PR&950DT products</term> support empty tensors and allow dynamic shapes or ranks at graph construction. All input dimensions must be known at execution time.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph | [test_geir_centralization](examples/test_geir_centralization.cpp) | Build the graph with `ge::op::Centralization`. |
| TensorFlow graph | - | Use the custom op `Centralization` and pass `axes` as a list-of-integers attribute. |
