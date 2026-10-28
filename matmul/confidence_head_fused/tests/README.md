# ConfidenceHeadFused 测试

## UT

| 目录 | 用例数 | 覆盖范围 |
| --- | --- | --- |
| [ut/op_api](ut/op_api/test_aclnn_confidence_head_fused.cpp) | 16 | FP16/BF16、Ascend950/Ascend350、可选输入、零 K、空输出，以及指针、类型、维度和 shape 校验。 |
| [ut/op_host](ut/op_host/test_confidence_head_fused_tiling.cpp) | 12 | 两种 tiling key、2D/3D、片上容量边界、AIC/AIV 比例、展平溢出和空输出。 |

在仓库根目录运行：

```bash
bash build.sh --ophost --ops=confidence_head_fused -u
bash build.sh --opapi --ops=confidence_head_fused -u
```

## ST

用例沿用仓库 TTK CSV 格式，由对应测试入口加载 [st/arch35](st/arch35) 下的用例表。

| 通路 | 用例数 | Golden |
| --- | --- | --- |
| ACLNN | 16 | [confidence_head_fused_aclnn_golden.py](assets/confidence_head_fused_aclnn_golden.py) |
| Torch | 15 | [confidence_head_fused_e2e_golden.py](assets/confidence_head_fused_e2e_golden.py) |
| Kernel | 13 | [confidence_head_fused_kernel_golden.py](assets/confidence_head_fused_kernel_golden.py) |

三条通路共保留 44 条代表用例，覆盖 FP16/BF16、2D/3D、N/K 尾块、hidden/markov 多轮 K、batch bias、markov-only、零 K、温度零值与负值、单侧与双侧截断、空输出。ACLNN 和 Torch 补充非连续输入，ACLNN 补充非连续输出。

公共计算逻辑位于 [Kernel Golden](assets/confidence_head_fused_kernel_golden.py)，ACLNN 和 Torch Golden 复用该逻辑，先按输入精度生成数据，再以 FP32 计算投影与后处理。多轮 K 的 bias 回归用例使用零特征、非零 bias，检查 bias 是否漏加或重复加入。

设备在温度缩放和 clamp 后使用默认的非精确 Div；FusedMatMul 的存量 Sigmoid 使用精确 Div，两者不要求逐位一致。三条 ST 通路均沿用 CSV 中的 `precision_tolerances=((0.001, 0.001),)`，分别表示 `rtol=1e-3` 和允许不匹配比例 `ptol=1e-3`，常规用例的 `absolute_precision=1e-4` 表示 `atol=1e-4`。逐元素按 `abs(actual-golden) <= atol + rtol*abs(golden)` 比较，再按 `ptol` 判定通过比例。`high_only_large_exponent` 用例保留 `atol=1e-30`，检查单侧 clamp 的执行顺序。
