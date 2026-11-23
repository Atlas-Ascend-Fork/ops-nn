# SituGluGroupQuant

## Applicable Products

| Product                                                      | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT series products</term>               |     √     |
| <term>Atlas A3 series products</term>                         |     ×     |
| <term>Atlas A2 series products</term>                         |     ×     |
| <term>Atlas 200I/500 A2 inference products</term>             |     ×     |
| <term>Atlas inference series products</term>                  |     ×     |
| <term>Atlas training series products</term>                   |     ×     |

## Function

- Operator function: for MoE grouped scenarios, fuses the sparse SiTU gated activation with MXFP8 (e4m3fn, blocksize=32) tail-axis dynamic quantization into a single operator. The last dimension of the input `x` concatenates the gate/up halves (2\*EPR); the activation result stays on-chip and is never written back to GM, and is then quantized per the MX algorithm into `y`(FLOAT8_E4M3FN) + `y_scale`(FLOAT8_E8M0). Within valid rows, the output is bit-identical to chaining the two small operators `npu_situ_and_mul_sparse -> npu_dynamic_mx_quant(dst=e4m3fn)`.
- Activation formulas (the gate/up selection direction is decided by the attribute `activate_left`):

  $$
  g = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \frac{1}{1+e^{-gate}},\qquad
  u = linear\_beta \cdot \tanh\left(\frac{up}{linear\_beta}\right)
  $$

  $$
  act = g \times u
  $$

  The whole chain is computed in the fp32 domain (with a single exit rounding) and has no intermediate low-precision round-trips; when the optional input `weight` is present, the per-row scaling applies: $act = fp32(act) \times w[row]$.

- Quantization description: along `axis` (currently only -1, the tail axis), every 32 elements form one MX block; the FLOAT8_E8M0 scale factor s is derived from the exponent field of the block's maximum absolute value Amax (OCP MX semantics); each element of the block is divided by s, rounded according to `round_mode`, and saturated into the FLOAT8_E4M3FN range to produce y. Special-value semantics: NaN elements in the activation result are individually set to +0 (a block is not zeroed as a whole due to partial NaN); ±Inf and finite overflow are clamped to the maximum finite value of the input type before quantization; a block whose elements are all NaN has Amax = 0 and is zeroed as a whole (y=+0, y_scale=0). When the group count is odd, each y_scale row ends with 1 padding byte that is not written out; callers should not read it.
- Grouping semantics (when `group_index` is present): the i-th element of group_index is the number of token rows assigned to expert i (count mode); the kernel accumulates the elements to obtain the effective row count validRows = min(sum(group_index), totalRows), processes only the first validRows rows, and leaves the remaining padding rows unwritten; when `group_index` is absent, whole-tensor mode applies (all rows are processed).

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
      <td>x</td>
      <td>Input</td>
      <td>Data to be activated and quantized; the last dimension concatenates the gate/up halves (2*EPR) and must be a positive even number. A zero-size leading dim degenerates to 0-row computation and is not rejected.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td>Per-token (row) scale factor; the element count equals the flattened total row count of x, indexed by the global row number. When present, the activation result is scaled per row in the fp32 domain before quantization; when absent, no scaling is applied.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td>Number of token rows assigned to each expert. Absent = whole-tensor mode; present = grouped sparse semantics (validRows = min(sum, totalRows), padding rows not written out).</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Attribute</td>
      <td>SiTU activation parameter of the gate branch; must be greater than 0.</td>
      <td>double</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>Attribute</td>
      <td>SiTU activation parameter of the up branch; must be greater than 0.</td>
      <td>double</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td>gate/up direction selection: true takes gate from the first half of x; false (default) takes gate from the second half.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Quantization axis; currently only -1 (the tail axis) is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Quantization mode; currently only 0 (MX quantization) is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Target quantized data type; currently only 36 (FLOAT8_E4M3FN) is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Attribute</td>
      <td>Numeric upper bound of the target data type; must be >= 0; currently not used in computation.</td>
      <td>double</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>Rounding mode; currently only rint is supported.</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>Attribute</td>
      <td>Semantic mode of group_index: 0 = cumsum mode (reserved, rejected when passed); 1 = count mode (current semantics, must be passed explicitly).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Quantized result, shape = x.shape[:-1] + [EPR].</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>Output</td>
      <td>MX quantization scale factor, shape = x.shape[:-1] + [ceil(ceil(EPR/32)/2), 2], one scale pair per group of 32 elements.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- x.shape[-1] must be a positive even number; EPR = x.shape[-1] / 2 must be a multiple of 32 within [128, 3072] (enforced by tiling validation; out-of-range inputs correspond to known defects of the original operator: EPR &lt; 128 silently loses y_scale, EPR &gt; 3072 overruns UB via the whole-row copy, EPR % 32 != 0 leaves tail-group garbage).
- beta > 0. linear_beta <= 0 disables the up-branch soft clamp (up passthrough); > 0 enables it (amplitude clamped to ±linear_beta); any value is valid.
- x supports rank >= 1 (no 8-dim cap check) and empty tensors are not rejected (a zero-size leading dim degenerates to 0 rows, consistent with the original operator); supported formats are ND/NCL/NCHW/NCDHW (contiguous storage treated as ND).
- **The default value of group_list_type is 0 (cumsum reserved mode, which is rejected); calling with the default signature fails even if group_index is not passed. It must be explicitly passed as 1 (count mode).**

## Invocation Description

| Invocation method | Invocation sample | Description |
|----------|----------|------|
| aclnn invocation | [test_aclnn_situ_glu_group_quant](./examples/arch35/test_aclnn_situ_glu_group_quant.cpp) | Invokes the SituGluGroupQuant operator through the aclnnSituGluGroupQuant API; see [aclnn_situ_glu_group_quant.h](./op_host/op_api/aclnn_situ_glu_group_quant.h) for the full signature and parameters. |
| Graph mode (C++ GE API) | [test_geir_situ_glu_group_quant](./examples/arch35/test_geir_situ_glu_group_quant.cpp) | Builds the graph via the op_graph proto and executes through GE graph compilation, covering grouped/zero-token/padding scenarios. |
| torchAPI invocation | [situ_glu_group_quant](./torch_extension/situ_glu_group_quant.py) | Invokes the operator through the torchAPI; runnable directly after installing the torch extension wheel; see [torchapi_situ_glu_group_quant](./docs/torchapi_situ_glu_group_quant.md) (zh-CN) for usage. |

Build / test / TTK commands:

```bash
bash build.sh -u --ophost   --ops=situ_glu_group_quant --soc=ascend950   # host UT
bash build.sh -u --opkernel --ops=situ_glu_group_quant --soc=ascend950   # kernel UT
bash build.sh --opkernel    --ops=situ_glu_group_quant --soc=ascend950   # kernel binary
bash build.sh --pkg         --ops=situ_glu_group_quant --soc=ascend950   # build package
bash build.sh --run_example situ_glu_group_quant eager cust --soc=ascend950  # onboard example (aclnn eager)
bash build.sh --run_example situ_glu_group_quant graph --soc=ascend950       # onboard example (C++ GE graph mode)
bash build.sh --torch_extension --ops=situ_glu_group_quant                   # torch extension whl
