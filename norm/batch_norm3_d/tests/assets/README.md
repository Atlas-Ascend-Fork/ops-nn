# BatchNorm3D回归测试

这些资产用于<term>Ascend 950PR&950DT系列产品</term>上的回归测试，使用本算子的`golden.py`。

| 资产 | 用途 |
| --- | --- |
| [ttk_kernel_batch_norm3_d.csv](../st/arch35/ttk_kernel_batch_norm3_d.csv) | 常规ST的18条小shape用例；Kernel和GEIR共用，单例全部输入合计不足50KiB。 |
| `generate_nonfinite_cases.py` | 按需生成36例非有限值固定输入，用于AR、RA和分核归约专项诊断，不纳入常规ST。 |
| `generate_precision_cases.py` | 按需生成50例固定输入，用于近零抵消、均值低位、极端数值及五条推理路径专项诊断，不纳入常规ST。 |

归档CSV覆盖FP16/FP32、NCDHW/NDHWC、训练/推理的全部8种组合，并包含可选输入与属性省略、M=1、epsilon为0、归约和通道尾部、4096归约分块边界、极值常量、大方差、大scale、近零抵消和Inf传播。非法格式拒绝、内存插桩及大shape路径需单独验证，不计入这份正向集合。

两个专项生成脚本接受`--output`指定输出目录，生成固定输入和`cases.csv`；常规ST无需运行它们。专项集合规模较大，使用前需单独评估内存和时长。

## 准备

源码与`ops-test-kit`位于同一任务目录。先按[编译安装指南](../../../../docs/zh/install/compile.md)加载目标CANN及待测算子包环境；构建有效并行度不超过8，TTK执行始终显式使用`--pc=1`。

```bash
export BN3D_TASK_ROOT=/path/to/task
export BN3D_ASSETS="$BN3D_TASK_ROOT/ops-nn/norm/batch_norm3_d/tests/assets"
export BN3D_CASES="$BN3D_TASK_ROOT/ops-nn/norm/batch_norm3_d/tests/st/arch35/ttk_kernel_batch_norm3_d.csv"
export BN3D_RESULTS="$BN3D_TASK_ROOT/reports/batch_norm3d_st"
mkdir -p "$BN3D_RESULTS"
cd "$BN3D_TASK_ROOT/ops-test-kit"
export TILINGKEY_PAR_COMPILE=1 CMAKE_BUILD_PARALLEL_LEVEL=8 MAX_JOBS=8
```

CSV中的`op_name=batch_norm3_d`对应安装入口及golden注册键；`BatchNorm3D`是GE算子类型名。

## 执行

以下命令使用NPU输出与CPU Golden的两方`stat_rel_err`比对，覆盖全部五个输出。插件默认声明`cross_check`，需保留命令行中的比较方式。Golden使用float64中间计算；epsilon先转换为接口声明的float32属性值，再提升为float64参与计算。

```bash
python3 -m ttk kernel --pc=1 --devices=0 --no-memory-check -d=false -b=release \
  --run 1 --plugin "$BN3D_ASSETS/golden.py" --compare stat_rel_err \
  -i "$BN3D_CASES" -o "$BN3D_RESULTS/kernel.csv"

python3 -m ttk geir --pc=1 --devices=0 --no-memory-check \
  --run 1 --plugin "$BN3D_ASSETS/golden.py" --compare stat_rel_err \
  -i "$BN3D_CASES" -o "$BN3D_RESULTS/geir.csv"
```

预编译包验证检查`bin_compile_s=BINARY_MATCH`、`bin_precision`和非空精度指标；GEIR检查`cst_compile_s`、`cst_precision`及实际输出比较。`COMPILE_ONLY`、`SUPPRESSED`或仅有进程退出码不能作为数值通过依据。

跨版本对照时，应先保存输入和golden，再通过`--manual-data-dirs`回放同一份数据。`--no-prof`仅用于准备冻结数据，验收执行必须移除。full_reduce的UB子缓冲安全仍需已插桩JIT kernel与mssanitizer验证，普通GM哨兵检查不能替代。
