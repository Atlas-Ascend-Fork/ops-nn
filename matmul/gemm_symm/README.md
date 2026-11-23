# GemmSymm

简体中文 | [English](./README_en.md)

`Y = alpha * (A @ B) + beta * C`。调用方保证 `A @ B` 对称。用到的 CATLASS 头文件放在 `op_kernel/arch35/catlass`，不依赖外部 catlass 库。当前 kernel 支持 float16、bfloat16 和 Ascend950，同一调用里 a、b、c、y 的 dtype 必须一致。

aclnn 两段式接口是 `aclnnGemmSymmGetWorkspaceSize` 和 `aclnnGemmSymm`，参数说明见 [aclnnGemmSymm](./docs/aclnnGemmSymm.md)。调用示例见 [test_aclnn_gemm_symm](./examples/arch35/test_aclnn_gemm_symm.cpp)。

Torch：

```python
torch.ops.cann_ops_nn.gemm_symm(a, b, c, alpha=1.0, beta=1.0, transpose_a=False, transpose_b=False)
```

`c` 可以是 `None`，此时 `beta` 必须为 0。`k == 0` 时不做矩阵乘，结果是 `beta * C`。
