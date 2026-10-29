# cross_entropy_sum_exp_and_index_logit

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:16:16.342Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: This operator is a fused operator for the local computation stage of CrossEntropy in the vocab parallel (Tensor Parallel) use case. For the local logits of a TP rank split along the vocab dimension, after `all_reduce(MAX)` and before the `all_reduce(SUM)` of `predicted_logits` / `sum_exp_logits`, it completes in one pass the logits shift (subtracting the global maximum), target out-of-bounds mask determination, local target offset computation, gather of the logit corresponding to the target, and `exp` with local summation along the vocab dimension, reducing the overhead of small operator launch, GM read/write, and intermediate tensor materialization. It is designed for the large-vocabulary large-model training task.

- Formula:

  Assume that the local vocab shard of the current rank is `vocab_parallel_logits`, the global maximum logit is `global_logits_max`, and the vocab shard range is `[vocab_start_index, vocab_end_index)`. For each token `i` and each local vocab position `j`:

  $$
  target\_mask[i] = \begin{cases} 1, & target[i] < vocab\_start\_index \text{ or } target[i] \geq vocab\_end\_index \\ 0, & \text{otherwise} \end{cases}
  $$

  $$
  target\_offset[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ target[i] - vocab\_start\_index, & target\_mask[i] = 0 \end{cases}
  $$

  $$
  predicted\_logits[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ vocab\_parallel\_logits[i, target\_offset[i]] - global\_logits\_max[i], & target\_mask[i] = 0 \end{cases}
  $$

  $$
  exp\_logits[i, j] = \exp\big(vocab\_parallel\_logits[i, j] - global\_logits\_max[i]\big)
  $$

  $$
  sum\_exp\_logits[i] = \sum_{j=0}^{V\_local-1} exp\_logits[i, j]
  $$

## Function Prototype

```python
cann_ops_nn.cross_entropy_sum_exp_and_index_logit(vocab_parallel_logits, target, global_logits_max, vocab_start_index, vocab_end_index) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Mandatory

>**NOTE:**<br>
>
>- Parameter dimensions: N (`prod(target.shape)`) indicates the total number of flattened tokens, and V_local indicates the length of the local vocab shard on the current rank. The 2D input shape is `[N, V_local]`, and the 3D input shape is `[S, B, V_local]` (where `N = S * B`).

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>vocab_parallel_logits</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Local vocab shard logits of the current TP rank, corresponding to vocab_parallel_logits in the formula. Supports non-contiguous tensor, with data format ND. Empty tensor is not supported.</td>
        <td>float32, bfloat16</td>
        <td>2D [N, V_local] or 3D [S, B, V_local], shape[:-1] must be consistent with target.shape</td>
    </tr>
    <tr>
        <td>target</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Global vocab index, which is the target in the formula. Supports non-contiguous tensors, data format ND, and non-negative integer values. Does not support empty tensors.</td>
        <td>int32</td>
        <td>Same shape as vocab_parallel_logits.shape[:-1], N range is [1, 32K] after flattening</td>
    </tr>
    <tr>
        <td>global_logits_max</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Global maximum logit obtained after all_reduce(MAX), which is global_logits_max in the formula. Supports non-contiguous tensors, data format ND. Does not support empty tensors.</td>
        <td>Same as vocab_parallel_logits</td>
        <td>Same shape as target</td>
    </tr>
    <tr>
        <td>vocab_start_index</td>
        <td>int</td>
        <td>Required</td>
        <td>Start index (global) of the vocab shard on the current rank, corresponding to vocab_start_index in the formula. vocab_start_index must be greater than or equal to 0.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>vocab_end_index</td>
        <td>int</td>
        <td>Required</td>
        <td>End index (global) of the vocab shard on the current rank, corresponding to vocab_end_index in the formula. vocab_end_index must be greater than vocab_start_index, and vocab_end_index - vocab_start_index == V_local.</td>
        <td>-</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Returns

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Output Name</th>
    <th>Output Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>predicted_logits</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Logit corresponding to target minus global_max, which is predicted_logits in the formula. The corresponding position is 0 when target is not in the current rank shard.</td>
        <td>float32</td>
        <td>Same as target.shape</td>
    </tr>
    <tr>
        <td>sum_exp_logits</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Sum of local exp(logits - global_max) along the last dimension, which is sum_exp_logits in the formula.</td>
        <td>float32</td>
        <td>Same as target.shape</td>
    </tr>
    <tr>
        <td>exp_logits</td>
        <td>Tensor</td>
        <td>Mandatory</td>
        <td>exp(vocab_parallel_logits - global_logits_max), which is exp_logits in the formula.</td>
        <td>float32</td>
        <td>Same as vocab_parallel_logits.shape</td>
    </tr>
    <tr>
        <td>target_offset</td>
        <td>Tensor</td>
        <td>Mandatory</td>
        <td>target - vocab_start_index, which is target_offset in the formula. Set to 0 if target is not in the shard of the current rank.</td>
        <td>int32</td>
        <td>Same as target.shape</td>
    </tr>
    <tr>
        <td>target_mask</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Vocab out-of-bounds mask. 1 indicates that target is not in the current rank shard, and 0 indicates that it is. This is target_mask in the formula.</td>
        <td>int32</td>
        <td>Same as target.shape</td>
    </tr>
</tbody>
</table>

## Constraints

- This API supports use in training scenarios.
- This API can be called in single-operator mode and TorchAir graph mode.
- `vocab_parallel_logits` supports only 2D `[N, V_local]` or 3D `[S, B, V_local]`, and its `shape[:-1]` must be exactly the same as `target.shape` and `global_logits_max.shape`.
- V_local alignment constraint: for `bfloat16` input, V_local must be a multiple of 16; for `float32` input, V_local must be a multiple of 8 (to ensure UB 32-byte alignment).
- N (`prod(target.shape)`) range: [1, 32K]; V_local range: [16, 200K].
- The operator does not support empty tensor input.

## Deterministic/Batch Consistency

- Defaults to a deterministic computation.
- Batch consistency is supported by default.

## Example

- Call in single-operator mode (eager)

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    # Parameter settings (2D [N, V_local])
    N = 4096
    V_local = 37888
    vocab_start_index = 0
    vocab_end_index = V_local

    # Generate random data and send to NPU
    vocab_parallel_logits = torch.randn(N, V_local, dtype=torch.bfloat16).npu()
    target = torch.randint(vocab_start_index, vocab_end_index, (N,), dtype=torch.int32).npu()
    # global_logits_max usually comes from cross-rank all_reduce(MAX), simulated with local max here
    global_logits_max = vocab_parallel_logits.to(torch.float32).max(dim=-1)[0].to(torch.bfloat16).contiguous()

    # Call the cross_entropy_sum_exp_and_index_logit operator
    predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask = \
        cann_ops_nn.cross_entropy_sum_exp_and_index_logit(
            vocab_parallel_logits,
            target,
            global_logits_max,
            vocab_start_index,
            vocab_end_index,
        )
    ```

- Call in graph mode (torchair)

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_nn


    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
            self,
            vocab_parallel_logits,
            target,
            global_logits_max,
            vocab_start_index,
            vocab_end_index,
        ):
            return cann_ops_nn.cross_entropy_sum_exp_and_index_logit(
                vocab_parallel_logits,
                target,
                global_logits_max,
                vocab_start_index,
                vocab_end_index,
            )


    def cross_entropy_sum_exp_and_index_logit_test():
        # Parameter settings (2D [N, V_local])
        N = 256
        V_local = 16
        vocab_start_index = 0
        vocab_end_index = V_local

        vocab_parallel_logits = torch.randn(N, V_local, dtype=torch.float32).npu()
        target = torch.randint(vocab_start_index, vocab_end_index, (N,), dtype=torch.int32).npu()
        global_logits_max = vocab_parallel_logits.max(dim=-1)[0].contiguous()

        model = NetModel()
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask = model(
            vocab_parallel_logits,
            target,
            global_logits_max,
            vocab_start_index,
            vocab_end_index,
        )


    if __name__ == "__main__":
        cross_entropy_sum_exp_and_index_logit_test()
    ```
