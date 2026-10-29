# ConfidenceHeadFused测试

## UT

| 目录 | 用例数 | 覆盖范围 |
| --- | --- | --- |
| [ut/op_api](ut/op_api/test_aclnn_confidence_head_fused.cpp) | 16 | FP16/BF16、Ascend950、可选输入、零K、空输出，以及指针、类型、维度和shape校验。 |
| [ut/op_host](ut/op_host/test_confidence_head_fused_tiling.cpp) | 12 | 两种tiling key、2D/3D、片上容量边界、AIC/AIV比例、展平溢出和空输出。 |

在仓库根目录运行：

```bash
bash build.sh --ophost --ops=confidence_head_fused -u
bash build.sh --opapi --ops=confidence_head_fused -u
```

## ST

用例沿用仓库TTK CSV格式，由对应测试入口加载 [st/arch35](st/arch35) 下的用例表。

| 通路 | 用例数 | Golden |
| --- | --- | --- |
| ACLNN | 16 | [confidence_head_fused_aclnn_golden.py](assets/confidence_head_fused_aclnn_golden.py) |
| Torch | 15 | [confidence_head_fused_e2e_golden.py](assets/confidence_head_fused_e2e_golden.py) |
| Kernel | 13 | [confidence_head_fused_kernel_golden.py](assets/confidence_head_fused_kernel_golden.py) |

三条通路共保留44条代表用例，覆盖FP16/BF16、2D/3D、N/K尾块、hidden/markov多轮K、batch bias、markov-only、零K、温度零值与负值、单侧与双侧截断、空输出。ACLNN和Torch补充非连续输入，ACLNN补充非连续输出。

公共计算逻辑位于 [Kernel Golden](assets/confidence_head_fused_kernel_golden.py)，ACLNN和Torch Golden复用该逻辑，先按输入精度生成数据，再以FP32计算投影与后处理。多轮K的bias回归用例使用零特征、非零bias，检查bias是否漏加或重复加入。

设备在温度缩放和clamp后使用默认的非精确Div；FusedMatMul的存量Sigmoid使用精确Div，两者不要求逐位一致。三条ST通路均沿用CSV中的`precision_tolerances=((0.001, 0.001),)`，分别表示`rtol=1e-3`和允许不匹配比例`ptol=1e-3`，常规用例的`absolute_precision=1e-4`表示`atol=1e-4`。逐元素按`abs(actual-golden) <= atol + rtol*abs(golden)`比较，再按`ptol`判定通过比例。`high_only_large_exponent`用例保留`atol=1e-30`，检查单侧clamp的执行顺序。
