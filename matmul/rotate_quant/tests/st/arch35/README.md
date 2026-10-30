# RotateQuant 普通模式 TTK 回归

本目录的 Ascend950 普通回归只覆盖 ACLNN 直调、kernel 直调和 Torch eager 单算子，
groupList 缺省，不涉及 GEIR 或图模式。每 32 个元素共享 scale 是普通 MX 量化规则，
不属于 M 轴分组功能。

## 测试资产

| 文件 | 内容 | 用例数 |
| --- | --- | ---: |
| ttk_kernel_rotate_quant_st.csv | 普通 RotateQuant kernel | 30 |
| ttk_aclnn_rotate_quant_st.csv | aclnnRotateQuant V1 | 33 |
| ttk_e2e_rotate_quant_st.csv | 算子仓接口 43 条、原 torch_npu 接口 43 条 | 86 |
| ttk_kernel_rotate_quant_dav_920_series_st.csv | DAV_9201/DAV_9202 的普通 MX 对照，可在 Ascend950 执行 | 12 |
| ttk_aclnn_rotate_quant_dav_920_series_st.csv | 对应 ACLNN V1 对照 | 12 |

三类 golden 入口位于 ../../assets：

- rotate_quant_kernel_golden.py：旋转、clamp、MX 参考计算，以及 Y/scale 比较函数。
- rotate_quant_aclnn_golden.py：V1 参数适配。
- rotate_quant_e2e_golden.py：按 api_name 分别适配 torch_npu.npu_rotate_quant 和
  torch.ops.cann_ops_nn.rotate_quant。两者复用数学计算，分别处理输出存储表示。

TTK 根据 CSV 的 api_name 和 golden 的同名 __spec__ 注册执行，无需算子专用框架分支。
两套 Torch 用例采用相同规格，各自与 golden 比较；不同用例名不保证自动共用同一份输入。

## 覆盖范围

- FLOAT16/BFLOAT16 输入，MXFP4-E2M1、MXFP8-E4M3FN、MXFP8-E5M2 输出。
- K=32/64/128/256，包括 FP8-E5M2 的 K=256；共享及逐 N 段 rotation。
- x 的一维、二维、三维和七维代表场景，覆盖高维展平、尾块及 scale padding。
- alpha 缺省、有效裁剪、边界值，以及合法的 roundMode/scaleAlg/dstTypeMax 分支。
- ACLNN/E2E 包含 M=0、N=0 和非连续输入；kernel 直调按连续输入契约设计。
- Torch 包含 FP4 打包、FP8 dtype 别名、可选属性 None 和非连续视图。

DAV_9201/DAV_9202 对照表使用 K=32/64/128，覆盖二维/三维 rotation、
M=17 和 M=32769，检查多轮调度及尾块。该平台不使用 K=256 和分组用例。
HiFloat4 不在此 MX golden 的覆盖范围内。

## 执行

Linux 测试机需安装配套 CANN、包含当前算子的算子包、torch/torch_npu 和 cann_ops_nn。
golden 依赖 NumPy、ml_dtypes、en_dtypes（至少 0.0.4）及提供 torch.float8_e8m0fnu
等类型的配套 Torch。原接口用例还要求 torch_npu 提供 npu_rotate_quant。

在 ops-test-kit 根目录执行，OPS_NN_PATH 替换为算子仓绝对路径：

~~~bash
export OPS_NN_PATH=/path/to/ops-nn
python3 -m ttk e2e \
  -i "$OPS_NN_PATH/matmul/rotate_quant/tests/st/arch35/ttk_e2e_rotate_quant_st.csv" \
  --plugin "$OPS_NN_PATH/matmul/rotate_quant/tests/assets" \
  --backend npu --devices 0 --pc 1 --seed 42 --task-prof false \
  --dump-on-fail -o rotate_quant_e2e_result.csv
~~~

ACLNN/kernel 使用对应子命令和同目录 CSV。仅验证新接口时选择对应 api_name 的行。
--plugin 指向测试资产，被测代码来自当前生效的算子包和 Torch 扩展。
不要启用 GEIR、torch.compile 或 aclgraph；--no-prof 会跳过设备执行，不能代替
--task-prof false。必须检查结果 CSV 的精度状态，不能只看进程退出码。

## 输出与精度

- kernel 输入槽位为 x、rotation、alpha、group_list，group_list 使用 None 占位。
  V1 ACLNN 和原 Torch 接口没有 group_list 参数。
- Torch FP4 的 Y 为 uint8[...,N/2]，低四位存放第一个逻辑元素；FP8 的 Y 保持原 shape。
- 算子仓 Torch scale 为 uint8[...,ceil(N/64),2]，保存原始 E8M0 编码；
  原 Torch scale 按 E8M0 语义比较，也兼容其 uint8 存储载体。
- Y 按目标格式相邻可表示值比较；scale 按编码精确比较，包含奇数块的末尾 padding。
  precision_tolerances 第一组第二项只控制 Y 的允许坏点比例。
- 参考计算保持 FP32 矩阵乘、Fixpipe BF16 舍入、BF16 clamp/倒数乘法、
  最终 FP4/FP8 转换及 E8M0 编码的顺序。

参考实现不模拟 MMAD 并行累加顺序。常规 CSV 使用有限输入；
NaN/Inf、溢出及特殊值转换仍需在 A5 上对照验证。

golden 自测位于 tests/ut/golden，在算子仓根目录执行：

~~~bash
python3 -m unittest discover -s matmul/rotate_quant/tests/ut/golden -p 'test_rotate_quant*.py' -v
~~~

分组回归已移至 ../grouped，使用其 README 中的显式命令执行，
不由本目录的普通 ttk_*.csv 扫描收集。
