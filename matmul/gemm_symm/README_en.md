# GemmSymm

English | [简体中文](./README.md)

`Y = alpha * (A @ B) + beta * C`. The caller guarantees that `A @ B` is symmetric. The CATLASS headers used by this operator live in `op_kernel/arch35/catlass` and do not depend on an external catlass library. The kernel supports float16, bfloat16, and Ascend950. In one call, a, b, c, and y must share the same dtype.

The two-phase aclnn interfaces are `aclnnGemmSymmGetWorkspaceSize` and `aclnnGemmSymm`. Parameter descriptions are in [aclnnGemmSymm](./docs/aclnnGemmSymm_en.md). The invocation sample is [test_aclnn_gemm_symm](./examples/arch35/test_aclnn_gemm_symm.cpp).

Torch:

```python
torch.ops.cann_ops_nn.gemm_symm(a, b, c, alpha=1.0, beta=1.0, transpose_a=False, transpose_b=False)
```

`c` may be `None`, in which case `beta` must be 0. When `k == 0`, the matmul is skipped and the result is `beta * C`.
