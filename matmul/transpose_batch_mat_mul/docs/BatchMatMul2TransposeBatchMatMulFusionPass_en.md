# BatchMatMul2TransposeBatchMatMulFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T16:54:13.177Z pushedAt=2026-10-13T01:47:29.716Z -->

## Fusion Patterns

**Pattern 1:**

When the Transpose (optional), BatchMatMul, and Transpose nodes are connected in the sequence shown in the following figure, they can be fused into a TransposeBatchMatMul operator node.

![](../../../docs/en/figures/BatchMatMul2TransposeBatchMatMulFusionPass_1.png)

**Pattern 2:**

When the x1 input shape of the BatchMatMul/BatchMatMulV2 node is \[B2,B1,1,K\] and the x2 input shape is \[1,B1,K,N\]/\[1,B1,N,K\], the nodes are fused into the TransposeBatchMatMul operator: x1 and x2 are first merged along axes via Reshape into \[B2,B1,K\] and \[B1,K,N\]/\[B1,N,K\] respectively, then computed by TransposeBatchMatMul to obtain \[B2,B1,N\], and finally restored via Reshape to \[B2,B1,1,N\].

![](../../../docs/en/figures/BatchMatMul2TransposeBatchMatMulFusionPass_2.png)

**Pattern 3:**

When the Transpose (optional), Transpose (optional), BatchMatMul, Transpose, Reshape, Reshape, and Transpose nodes are connected in the sequence shown in the following figure, they can be fused into a TransposeBatchMatMul operator node.

![](../../../docs/en/figures/BatchMatMul2TransposeBatchMatMulFusionPass_3.png)

## Constraints

**Pattern 1 and Pattern 3:**

- Supported data types of input x1, x2, and output out (inputs and outputs must correspond): BFLOAT16, FLOAT16, FLOAT32.
- In Mode 3, the output of out is \[B1,M,B/B1\*N\], and B%B1=0.
- The adj\_x1 and adj\_x2 attributes of BatchMatMul/BatchMatMulV2 nodes must both be false.
- Only the static graph mode is supported.
- The bias input is not supported.
  <!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>: The x1 input supports \[B,M,K\] or \[M,B,K\], and the x2 input supports only \[B,K,N\]. The attributes of the corresponding TransposeBatchMatMul are perm\_x1=\[0,1,2\]/\[1,0,2\] and perm\_x2=\[0,1,2\].
  <!-- end id1 -->

  <!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: When the input data type is BFLOAT16 or FLOAT16, k and n must be aligned to 128. B\*K<65536 or B\*K\>=65536 must be satisfied, and k<65536.
  <!-- end id2 -->

  <!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: When the input data type is FLOAT32 and the input transpose node is not empty, there is no alignment constraint, but B\*K<65536 must still be satisfied.
  <!-- end id3 -->

  <!-- npu="A3" id4 -->
- <term>Atlas A3 products</term>: The x1 input supports \[B,M,K\] or \[M,B,K\], and the x2 input supports only \[B,K,N\]. The attributes of the corresponding TransposeBatchMatMul are perm\_x1=\[0,1,2\]/\[1,0,2\] and perm\_x2=\[0,1,2\].
  <!-- end id4 -->

  <!-- npu="A3" id5 -->
- <term>Atlas A3 products</term>: When the input data type is BFLOAT16 or FLOAT16, k and n are aligned to 128, B\*K<65536 or B\*K\>=65536 must be satisfied, and k<65536.
  <!-- end id5 -->

  <!-- npu="A3" id6 -->
- <term>Atlas A3 products</term>: When the input data type is FLOAT32 and the transpose node in the input is not empty, there is no alignment requirement, but B\*K<65536 must still be satisfied.
  <!-- end id6 -->

  <!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: The input of x1 is \[B,M,K\] or \[M,B,K\], the input of x2 is \[B,K,N\] or \[B,N,K\], and the attributes of the corresponding TransposeBatchMatMul are perm\_x1=\[0,1,2\]/\[1,0,2\] and perm\_x2=\[0,1,2\]/\[0,2,1\].
  <!-- end id7 -->

  <!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: No fusion is performed when the input shape can use the matmultomul or iterbatch optimization template.
  <!-- end id8 -->

**Pattern 2:**

- The input of x1 is \[B2,B1,1,K\], the input of x2 is \[1,B1,K,N\] or \[1,B1,N,K\], and the attributes of the corresponding BatchMatMul are adj\_x1=false and adj\_x2=false/true.
- Supported data types of input x1, x2, and output out (the inputs and output must correspond): FLOAT32.
  <!-- npu="910b" id9 -->
- <term>Atlas A2 products</term>: Enabling HF32 is not supported.
  <!-- end id9 -->

  <!-- npu="A3" id10 -->
- <term>Atlas A3 products</term>: Enabling HF32 is not supported.
  <!-- end id10 -->

- Only the static graph mode is supported.
- The bias input is not supported.
  <!-- npu="910b" id11 -->
- <term>Atlas A2 products</term>: The input must satisfy B1\*K<65536.
  <!-- end id11 -->

  <!-- npu="A3" id12 -->
- <term>Atlas A3 products</term>: The input must satisfy B1\*K<65536.
  <!-- end id12 -->

## Applicable Products

<!-- npu="910b" id13 -->
Atlas A2 products
<!-- end id13 -->

<!-- npu="A3" id14 -->
Atlas A3 products
<!-- end id14 -->

<!-- npu="950" id15 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id15 -->
