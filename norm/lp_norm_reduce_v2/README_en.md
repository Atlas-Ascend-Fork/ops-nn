# LpNormReduceV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |    ×    |
|  <term>Atlas inference products</term>    |    ×    |
|  <term>Atlas training products</term>    |    ×    |


## Function

- Operator function: computes the reduction stage of the Lp norm. The absolute value of each element of the input `x` is taken first, then the reduction is performed along the axes specified by the attribute `axes`. The reduction rule is determined by the attribute `p`. This operator performs only the reduction, without taking the final $\frac{1}{p}$-th root; a complete Lp norm is obtained by chaining this operator with the matching LpNormUpdateV2 operator.
- Key difference from LpNormReduce (V1): the attribute `p` is of the **FLOAT** type (INT in V1) and supports ±inf and any finite floating-point order. The matching priority from top to bottom is:

| Priority | p Value | Formula |
|-------|--------|---------|
| 1 | p = +inf (including float aliases 2147483647.0/2147483648.0) | $y = \max\limits_{i \in S}{a_i}$ |
| 2 | p = −inf (including the float alias −2147483648.0) | $y = \min\limits_{i \in S}{a_i}$ |
| 3 | p = 0 | $y = \sum_{i \in S}{[a_i \neq 0]}$ (number of non-zero elements) |
| 4 | p = 1 | $y = \sum_{i \in S}{a_i}$ |
| 5 | p is an integer and 2 ≤ p ≤ 15 | $y = \sum_{i \in S}{a_i^{p}}$ (implemented by repeated multiplication) |
| 6 | Other finite floating-point p | $y = \sum_{i \in S}{\exp(p \cdot \ln a_i)}$ |

  where $a_i = \lvert x_i \rvert$ and $S$ is the normalized set of reduction axes (an empty `axes` means all axes; negative axes are folded by `axis + rank`; duplicate axes are rejected).

- Upcast semantics: FLOAT16 and BFLOAT16 inputs are converted to FLOAT32 for computation (both the absolute value and the reduction are performed in FLOAT32), then the result is written back to the original data type — FLOAT16 is rounded to nearest on conversion, and BFLOAT16 is written back with round-to-nearest-even. FLOAT32 inputs are computed directly. The p=±inf and p=0 branches are exact (bitwise identical to the reference implementation when the input contains no NaN).
- Example: with an x shape of `(4, 32, 16, 70)`, `axes = [1]`, `keepdim = true`, and `p = 2.0`, the output y shape is `(4, 1, 16, 70)`, and each output element equals the sum of squares of the 32 input elements along axis 1 at the output position $(i, 0, k, l)$ (the reduced axis 1 is retained with length 1 in the output).

## Parameters

| Parameter  | Input/Output/Attribute | Description | Data Type | Data Format |
| ------- | -------------- | ---- | -------- | -------- |
| x       | Required input       | The `x` in the formula, indicating the tensor to be reduced. The rank ranges from 0 to 8. | FLOAT, FLOAT16, BFLOAT16 | ND |
| y       | Required output       | The output `y` in the formula. The data type is the same as `x`, and the shape is determined by `axes` and `keepdim`. | FLOAT, FLOAT16, BFLOAT16 | ND |
| p       | Optional attribute       | The `p` in the formula, indicating the floating-point order of the norm. The default value is 2.0. The value follows the IEEE 754 range, supporting ±inf (including the fp32 storage aliases 2147483647.0/2147483648.0 and −2147483648.0) and any finite floating-point value; NaN is not supported. | FLOAT | - |
| axes    | Optional attribute       | The set of reduction axes $S$ in the formula. The default value is an empty list. An empty list means reduction over all axes; the value range is [-rank(`x`), rank(`x`)), and negative axes are supported; elements must not be duplicated and the count must not exceed rank(`x`). | LISTINT | - |
| keepdim | Optional attribute       | Determines whether the output tensor retains the reduction axes. If set to true, the reduction axes are retained with length 1; if set to false, the reduction axes are removed, and a full-axis reduction outputs a 0-dimensional scalar. The default value is false. | BOOL | - |
| epsilon | Optional attribute       | Reserved parameter, not used in computation (aligned with the TBE reference implementation; consumed by the subsequent LpNormUpdateV2). The default value is 1e-12. | FLOAT | - |

## Constraints

- This operator is the floating-point p evolution of LpNormReduce. It supports only GE graph mode and Kernel-compatible-path invocation, and does not provide the aclnn interface.
- The total number of input elements must not exceed $2^{31}-1$.
- NaN values in the input are not filtered: NaN participates in the computation per IEEE 754 semantics (counted as a non-zero element in the p=0 branch; in the p=±inf branch, a reduction containing NaN yields NaN).
- In the p∈[2,15] integer branch, the intermediate power $|x|^p$ may exceed the FLOAT32 upper limit and become inf; no saturation protection is performed (consistent with the reference implementation).
- When the reduction domain is empty for a finite p (for example, the shape contains a zero-length dimension), the output is 0; if the zero-length dimension is on a non-reduction axis, the output is an empty tensor.
- Output shape derivation: when keepdim=true, `y.shape[i] = (i ∈ axes) ? 1 : x.shape[i]`; when keepdim=false, the reduced dimensions are removed; when `axes` is empty and keepdim=false, the output is a 0-dimensional scalar.
- Known boundary (p=−inf branch): for the degenerate input where an entire row of $|x|$ is +inf, the output is FLT_MAX instead of +inf (the init sentinel of the underlying min-reduction instruction wins, which is indistinguishable from a mixed {FLT_MAX,+inf} set).

## Invocation

| Invocation Mode | Sample Code | Description |
| -------- | -------- | ---- |
| GE graph mode | -        | Build a graph with the [LpNormReduceV2 operator IR definition](op_graph/lp_norm_reduce_v2_proto.h); the in-repo [`../lp_norm_v2/op_api/lp_norm_reduce_v2.cpp`](../lp_norm_v2/op_api/lp_norm_reduce_v2.cpp) provides a graph-construction helper. |

## Build and Test

```bash
# Build the operator (specify the soc)
bash build.sh --ops=lp_norm_reduce_v2 --soc=ascend950

# Run the UT (op_host infershape/tiling + op_graph inferdtype, executed automatically after the build)
bash build.sh --ops=lp_norm_reduce_v2 --soc=ascend950 -u
```

- UT: `tests/ut/` (26 infershape cases + 90 arch35 tiling cases + 16 cross-core routing tiling cases + 6 op_graph InferDataType cases, executed via the shared in-repo Host UT runner).
- ST: `tests/st/` (`ttk_geir_lp_norm_reduce_v2.csv` with 31 cases + `arch35/ttk_kernel_lp_norm_reduce_v2.csv` with 93 cases, covering 6 p branches × 3 data types × multiple reduction layouts and cross-core scenarios). The third-party precision reference is in `tests/assets/golden.py`, based on the official torch Lp-norm API `torch.linalg.vector_norm` (ord=inf/-inf/0/1 maps directly to each branch; any p restores the power sum via `norm^p`).
