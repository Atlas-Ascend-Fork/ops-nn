# NormalizeBatch

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT series products</term> | √ |
| <term>Atlas A3 series products</term> | × |
| <term>Atlas A2 series products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference series products</term> | × |
| <term>Atlas training series products</term> | √ |

> Note: This operator registers compute units such as `ascend950`, `ascend960dt`, and `ascend960pr`, sharing the same arch35 AscendC implementation.

## Function Description

- Operator function: variable-length sequence batch normalization. For each `(n, c)` row (or `n` plane) of the input tensor `x` (of shape `(n, c, d)`), normalization is performed over the number of elements given by `seq_len[n]` for each batch. The output `y` has the same shape and dtype as `x`. Only the first `L` elements of each row are used to compute the statistics (mean/variance), but all `d` elements of the row are normalized.
- Calculation formula:

  $$
  L = \min(\max(seq\_len_n,\ 1),\ d)
  $$

  $$
  N = \begin{cases} L & \text{per\_feature (statistics computed independently for each (n, c) row)} \\ c \cdot L & \text{all\_features (all c rows of the same n share the statistics)} \end{cases}
  $$

  $$
  mean = \frac{1}{N}\sum_{k=0}^{L-1} x_{n,\cdot,k} \qquad var = \left|\,E(x^2) - E(x)^2\,\right| \cdot \frac{N}{N-1}
  $$

  $$
  y = \frac{x - mean}{\sqrt{var} + \epsilon}
  $$

  where `epsilon` is added after the `sqrt` (added to the standard deviation, not to the variance); the variance carries the unbiased correction `N/(N-1)` and takes the absolute value to guard against negatives; when `N < 2`, `var = 0`; `seq_len` is clamped to `[1, d]` inside the operator.

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input_x</td>
      <td>Input</td>
      <td>Input feature tensor to be normalized, i.e., <code>x</code> in the formula, of shape <code>(n, c, d)</code>.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_len</td>
      <td>Input</td>
      <td>Number of elements to normalize for each batch, i.e., <code>seq_len</code> in the formula, of shape <code>(n,)</code>; values are clamped to <code>[1, d]</code> inside the operator.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>Output</td>
      <td>Normalized output tensor, i.e., <code>y</code> in the formula; its shape and data type are the same as those of <code>input_x</code>.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normalize_type</td>
      <td>Attribute</td>
      <td>Statistics scope of normalization: <code>per_feature</code> computes statistics independently for each <code>(n, c)</code> row (<code>N = L</code>); <code>all_features</code> shares the statistics across all <code>c</code> rows of the same <code>n</code> (<code>N = c * L</code>). Required; no other values are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td>Lower bound that prevents division by zero, i.e., <code>epsilon</code> in the formula, added after the standard deviation. The default value is <code>1e-5</code>. The value must be greater than 0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- `input_x` supports only FLOAT32 and only the ND data format, with 3 dimensions (`n, c, d`), where `n >= 1`, `c >= 1`, and `d > 1` (the `d` dimension is the statistics length dimension and must be greater than 1); `output_y.shape = input_x.shape`.
- `seq_len` supports only INT32 and only the ND data format, with 1 dimension (`(n,)`); values are clamped at runtime to `[1, d]`.
- `normalize_type` is required and supports only `per_feature` or `all_features`.
- `epsilon` is of the FLOAT32 type and must be greater than 0; the default value is `1e-5`; it is added after `sqrt(var)`.
- On <term>Ascend 950PR&950DT series products</term>, this operator is implemented out-of-place: `input_x` remains unchanged after execution; the caller must ensure that the underlying memory of `output_y` does not overlap that of `input_x`.
- On <term>Ascend 950PR&950DT series products</term>, statistics and normalization are performed entirely in FLOAT32, using translation-domain statistics and compensated normalization; for near-constant inputs (`std << |mean|`), the accuracy is better than that of direct element-wise computation.

## Calling Method

| Calling Method | Sample | Description |
|---------|----|------|
| GE graph mode | [test_geir_normalize_batch](examples/test_geir_normalize_batch.cpp) | Calls the NormalizeBatch operator by building a graph with the [operator IR](op_graph/normalize_batch_proto.h). |
