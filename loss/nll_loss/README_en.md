# NllLoss

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

  - Description: computes the negative log-likelihood loss value.

  - Formula:

    When `reduction` is `none`:

    $$
    \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
    l_n = - w_{y_n} x_{n,y_n}, \quad
    w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not=   \text{ignoreIndex}\},
    $$

    Here $x$ is self, $y$ is target, $w$ is weight, and $N$ is the batch size. If `reduction` is not `'none'`, then:

    $$
    \ell(x, y) = \begin{cases}
        \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
        \text{if reduction} = \text{`mean';}\\
        \sum_{n=1}^N l_n,  &
        \text{if reduction} = \text{`sum'.}
    \end{cases}
    $$

    $$
    totalWeight = \sum_{n=1}^N w_{y_n}
    $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 910px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
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
        <td>x</td>
        <td>Input</td>
        <td><ul><li>The input parameter for the nll_loss computation, x in the formula.</li><li>The shape supports (C,), (N, C) or (N, C, H, W), where N indicates the batch size and C indicates the number of classes.</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>target</td>
        <td>Input</td>
        <td><ul><li>The input parameter for the nll_loss computation, target in the formula, indicating the ground-truth label.</li><li>The shape is (), (N,) or (N, H, W), corresponding to the dimensions of x.</li><li>Each element must be in the range [0, C-1]; elements equal to ignore_index do not participate in the loss computation.</li></ul></td>
        <td>INT32, INT64, UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Optional input</td>
        <td><ul><li>The input parameter for the nll_loss computation, weight in the formula, indicating the scaling weight of each class.</li><li>The shape is (C,).</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16 (same as self)</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td><ul><li>The output parameter of the nll_loss computation, L in the formula.</li><li>When reduction is none: the shape is (N,) for (N, C) input, (N, H, W) for (N, C, H, W) input, and () (a 0-D scalar) for (C,) input; when reduction is mean or sum, the shape is (1,).</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>total_weight</td>
        <td>Output</td>
        <td><ul><li>The output parameter of the nll_loss computation, totalWeight in the formula, i.e. the sum of class weights.</li><li>The shape is (1,). It is valid only when reduction is mean or sum; the value is undefined when reduction is none.</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>Optional attribute</td>
        <td><ul><li>Specifies the reduction applied to the output: none means no reduction; mean means the output is the loss sum divided by totalWeight; sum means the output is the sum of the per-element losses.</li><li>The default value is "mean".</li></ul></td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignore_index</td>
        <td>Optional attribute</td>
        <td><ul><li>Specifies a target value that is ignored and does not affect the loss computation.</li><li>The default value is -100; it should take an integer value outside [0, C-1] that is comparable with the target data type.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
      </tr>
    </tbody></table>

## Constraints

  - Each element of target should be in the range [0, C-1]; elements equal to ignore_index do not participate in the loss computation. The current implementation does not explicitly validate out-of-range values, so the behavior for out-of-range values is outside the public contract.
  - ignore_index is an integer with the default -100; it should take a value outside [0, C-1] that is comparable with the target data type, otherwise the loss of the corresponding class is skipped.
  - The input x supports only 1D (C,), 2D (N, C) or 4D (N, C, H, W); target is correspondingly 0D, 1D (N,) or 3D (N, H, W), and the dimensions must match those of x.
  - reduction takes the value none, mean or sum, with the default mean.
  - The output total_weight is valid only when reduction is mean or sum; its value is undefined when reduction is none.
  - Only the ND data format is supported.
  - Empty input scenario: when the number of elements of the input x is 0, reduction mean outputs NaN, sum outputs 0, and total_weight is 0; when reduction is none, the output is not written.
  - When the class dimension C of x is 0 and target is non-empty, the input is invalid.

## Invocation Description

  | Invocation method | Invocation sample                                                                   | Description                                                           |
  |--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
  | aclnn API | [test_aclnn_nll_loss](./examples/test_aclnn_nll_loss.cpp) | Invoke the NLLLoss operator through the [aclnnNLLLoss](./docs/aclnnNLLLoss_en.md) interface. |
  | aclnn API | [test_aclnn_nll_loss_2d](./examples/test_aclnn_nll_loss_2d.cpp) | Invoke the NLLLoss operator through the [aclnnNLLLoss2d](./docs/aclnnNLLLoss2d_en.md) interface. |
  | GE graph mode | - | Invoke the NLLLoss operator by constructing a graph through the [operator IR](./op_graph/nll_loss_proto.h). |

## UT Testing

  - Command for building host-side UTs (including the InferShape cases and the arch35 Tiling cases):

    ```bash
    ./build.sh -u --ophost --noexec --soc=ascend950 -O0 -j8
    ```

    `--soc` specifies the target SoC for operator compilation (ascend950, etc. are supported); if `--soc` is not specified, the arch35 Tiling cases (tests/ut/op_host/arch35/test_nll_loss_tiling.cpp) are not compiled or executed, and only host cases such as InferShape are run. Removing `--noexec` executes the cases directly after compilation.
