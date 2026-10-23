# QuantMatmulActivationQuant测试

本目录同时覆盖SwiGLU和SiLU的MXFP8通路，以及GELU（tanh/erf）的MXFP8、MXFP4通路。设备用例和CPU golden均位于仓内，不依赖桌面脚本或个人目录。

| 目录 | 用途 |
| --- | --- |
| `ut/op_api` | ND与WeightNz接口分别使用独立的C++/CSV，包含151条ND和88条WeightNz用例 |
| `ut/op_host` | InferShape和Host tiling回归 |
| `ut/op_kernel` | CSV保存用例、tiling参数和期望输出，公共工具头负责加载、校验及执行；`arch35/test_quant_matmul_activation_quant.cpp`统一四组dtype/layout测试 |
| `assets/spec.py` | 自包含的NumPy golden、FP8/FP4/E8M0编解码、比较器及TTK kernel/ACLNN/PyTorch E2E适配 |
| `st/arch35` | 124条kernel（含18条Inf/NaN、2条FP4量化下溢及3条erf/累加边界）、143条ACLNN回归（36条ND、107条WeightNZ）、17条E2E配置；SwiGLU和SiLU覆盖FP8 ND与WeightNZ |

## Kernel UT接入

本仓从算子根`CMakeLists.txt`调用`AddOpTestCase`，沿用`cmake/ut.cmake`的源码拷贝、tiling stub和`MULTI_KERNEL_TARGET`路径，并显式声明`quant_batch_matmul_v3`依赖。同一个测试源文件编译为四组dtype/layout对象，使用独立的kernel符号和GoogleTest suite，共30条用例。

`arch35/quant_matmul_activation_quant_cpu_debug_stub.h`仅在CPU调试测试中加载：复用ops-tensor的Blaze兼容定义，将GM-to-L1拷贝转发到模拟器实现，并分别映射UB、L1、L0A/B/C、bias和MX scale buffer的地址。测试执行复用ops-tensor的`kernel_ut_runner.h`检查模拟器子进程错误，PV符号只链接一份。

这里不需要`tiling_registry_stub.cpp`：NN的`AddOpTestCase`检测到现有`quant_matmul_activation_quant_tiling_def.h`后会直接强制包含该文件，跳过注册表生成tiling头的流程；`GET_TILING_DATA_WITH_STRUCT`按指定类型拷贝生产POD，与入口的`REGISTER_NONE_TILING`配合使用，无须另外维护同一结构的注册字段。

算子使用独立的tiling POD：batch结构为104字节，非batch结构为56字节。Host在x1、x2的batch总数均为1时选择`TPL_WITHOUT_BATCH`，包括二维矩阵以及前导batch维全为1的输入。Kernel UT按CSV的batch维度选择对应入口及结构大小，现有用例中27条走非batch、3条走batch；Host UT另外检查序列化大小、batch分发及前导维全为1时的bias/广播形状。

用例由`ut/op_kernel/test_quant_matmul_activation_quant.csv`驱动，组织方式参考`block_attn_res_update`：公共`test_*_utils.h`返回用例列表和加载错误，统一测试入口按`socVersion` / `kernelUtTarget`选择用例并注册参数化测试。缺文件、错误表头、错列、重复名称、非法参数或没有匹配用例都会使CSV加载测试失败。

| CSV字段 | 含义 |
| --- | --- |
| `kernelUtTarget` | `QMMAQ_E4M3` / `QMMAQ_E5M2` / `QMMAQ_E5M2_GELU` / `QMMAQ_E4M3_WEIGHT_NZ`，分别对应17 / 3 / 5 / 5条用例及其编译dtype |
| `m,n,k,batchA,batchB` | 矩阵尺寸和四维batch；batch维度用空格分隔 |
| `activationType,scaleAlg,x1Dtype,x2Dtype,yDtype,transposeX2,fullLoad,biasMode,numBlocks` | 激活、量化、dtype和执行参数；dtype使用GE编码35/36，biasMode为0无bias、1 `[N]`、2 `[B,1,N]` |
| `baseM,baseN,baseK,kL1,scaleKL1,nBufferNum,dbL0C,mTailTile,nTailTile` | 构造tiling的显式字段；A/B共用kL1，尾轮tile直接由CSV控制，避免在C++测试中克隆并篡改case，也避免将整个结构体的ABI字节串固化到表格 |
| `dataPattern` | `batch_constant`保留原确定性数据；`coordinate`随batch、M/N/K坐标及K组变化；`subnormal_bias`用正常FP32 bias构造非零BF16 subnormal激活值；`gate_subnormal`检查FP32 subnormal gate在SiLU的FTZ Div中清零；`activation_bias`用零矩阵乘及固定bias隔离零值、负值与sigmoid饱和区间 |
| `expectedY,expectedYScale` | 原始字节用空格分隔；`coordinate`按batch/M顺序保存全部行，其他模式每个batch保存一行，检查时覆盖全部M行 |

输入由工具头中的固定规则生成，期望字节按`assets/spec.py`的`reference`计算。修改尺寸、输入规则或计算属性时须同步更新期望值。无CPU模拟器宏时仍可编译CSV加载部分，kernel执行测试明确标记为跳过。

在具备CANN、Ascend950 tikicpulib、匹配ops-tensor/Blaze的Linux构建环境运行：

```bash
bash build.sh -u --opkernel --ops=quant_matmul_activation_quant --soc=ascend950
```

Kernel UT保留GELU基础回归，以及SwiGLU的N=128、WeightNZ FF/FT和subnormal gate覆盖；新增12条SiLU用例覆盖M=1/N=1、负值与零值、N/K尾块、OCP/BLAS、全载/非全载、多tile、batch/bias、ND转置和WeightNZ非对齐存储。原始N不满足64对齐的SwiGLU直调用例已移除；接口、图推导和tiling CSV将这些尺寸作为拒绝用例。设备ST继续覆盖输出32列尾块、五/六维广播、K尾块、WeightNZ、bias、两种FP8输出和全载/非全载路径。

## TTK ST

`ttk_kernel_quant_matmul_activation_quant_st.csv`包含18条名称以`swiglu_nonfinite_`开头的SwiGLU特殊值回归：E5M2输入或FP32 bias为`+Inf`、`-Inf`、`NaN`，两种scale算法，E4M3FN/E5M2输出，组内混合有限值与特殊值，以及输出32列时的scale padding。混合用例通过CSV的`attributes['bias']`指定数据，零矩阵乘隔离激活与量化；相邻的有限值组必须保持正常输出。E4M3FN没有Inf编码，因此Inf由E5M2输入或FP32 bias构造。

SiLU的kernel ST包含32条ND组合（4种输入dtype组合 × FF/FT/TF/TT × OCP/BLAS）、8条WeightNZ组合（2种x1 dtype × FF/FT × OCP/BLAS），以及ND batch、NZ batch/bias和多N tile共3条边界用例。ACLNN和E2E另行覆盖接口装配及输出形状，避免在C++中按业务shape重复添加独立测试。InferShape CSV覆盖动态N、batch输出及属性/dtype拒绝路径；Host/API CSV检查输出保持完整N及scale形状。

WeightNZ的ACLNN表另包含98条SiLU形状回归，名称以`silu_weight_nz_`开头，覆盖FF/FT、非对齐N/K和不同batch广播；每条均显式填写NZ物理存储。SwiGLU在kernel与ACLNN表中新增`swiglu_weight_nz_t0_n128_k67`和`swiglu_weight_nz_t1_n128_k67`，分别检查K=67时FF/FT的物理布局及golden解码。

NZ配置须区分测试通路：

- ACLNN的`tensor_view_shapes`保存逻辑shape，`tensor_storage_shapes`保存实际分配的物理shape。设置`tensor_formats='FRACTAL_NZ'`不会自动补齐storage；即使元素数恰好对齐，也应填写完整物理shape以满足WeightNZ接口的storage rank要求。
- 对FP8权重，若逻辑shape为`(*batch, R, C)`，物理shape为`(*batch, ceil(C/32), ceil(R/16), 16, 32)`。FF时`(R,C)=(K,N)`，FT时`(R,C)=(N,K)`；batch维保持不变。其他ND tensor的storage项可填写`None`。
- kernel的`input_shapes`直接填写NZ物理shape，`input_ori_shapes`填写原始逻辑shape，`input_ori_formats`填写ND。这两列也提供给golden解码，不能用ACLNN的字段替代。
- 本目录E2E表使用ND逻辑视图，WeightNZ的物理存储由kernel与ACLNN表覆盖。

例如SiLU的M=3、N=66、K=66、FF用例，`x2`逻辑shape为`(66,66)`，NZ物理shape为`(3,5,16,32)`；七个ACLNN tensor对应的`tensor_storage_shapes`应为`(None,(3,5,16,32),None,None,None,None,None)`。缺失该列时只分配4356个元素，而NZ解码需要7680个，会报`WeightNZ storage has ... elements, expected at least ...`。另需使用升精度时保留完整底层storage及view几何的TTK版本；CSV正确分配与golden升精度保留storage是两项独立要求。

ND基础CSV另有`silu_nd_e4_e4_ff_s1_m1n32769k65_full_load`：以单行、宽N和K尾块补充A全载回归。实际是否选择全载取决于平台与Host tiling，应从执行日志核对tiling key的`KERNELTYPE=5`；边界批次不能仅凭shape名称宣称已覆盖全载。

`assets/spec.py`顶部通过`__spec__`注册四个入口，注册名分别与CSV的`op_name` / `api_name`完全一致：

| 通路 | 注册名 | TestSpec |
| --- | --- | --- |
| kernel | `quant_matmul_activation_quant` | `QuantMatmulActivationQuantTestSpec` |
| ACLNN | `aclnnQuantMatmulActivationQuant` | `QuantMatmulActivationQuantAclnnTestSpec` |
| ACLNN WeightNZ | `aclnnQuantMatmulActivationQuantWeightNz` | `QuantMatmulActivationQuantAclnnTestSpec` |
| E2E | `cann_ops_nn.ops.quant_matmul_activation_quant` | `QuantMatmulActivationQuantE2ETestSpec` |

E2E使用`cann_ops_nn.ops`导出的函数。不要写成`cann_ops_nn.ops.matmul.quant_matmul_activation_quant.quant_matmul_activation_quant`：`matmul`包已将同名属性导出为函数，TTK执行器逐级访问属性时会报`'function' object has no attribute 'quant_matmul_activation_quant'`。

完整测试需要Linux、CANN、Ascend950，以及包含本次修改的算子包和对应ops-tensor；E2E还需匹配的`torch`、`torch_npu`、TorchAir和`cann_ops_nn`扩展。TTK的FP8/FP4/E8M0输入和设备输出转换需要`ml-dtypes`、`en-dtypes>=0.0.4`。若E2E校验在导入扩展时出现`API_PARSE_FAIL`，且堆栈报告`DataType`缺少`DT_FLOAT8_E5M2`，应检查所用Python环境的TorchAir是否包含FP8类型定义；此时尚未执行算子。先加载目标CANN环境，再从ops-test-kit源码目录执行；将以下两个路径替换为Linux环境中的实际路径：

新增激活的ACLNN验证需要Host/tiling与kernel二进制同时更新。仅安装`--jit`源码包时，运行时可能仍选择CANN内置的旧二进制，报新增tiling key不存在。可用以下命令构建本算子的完整二进制包；`--no_force`只限制待编译的kernel范围，Host/API依赖仍按构建系统生成：

```bash
source /usr/local/Ascend/cann/set_env.sh
cd /path/to/ops-nn
bash build.sh --pkg --no_force --ops=quant_matmul_activation_quant --soc=ascend950 --vendor_name=custom -j16
bash build_out/cann-ops-nn-custom_linux-x86_64.run --quiet --install-path="$ASCEND_HOME_PATH/opp"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
```

```bash
cd /path/to/ops-test-kit
OP_TESTS=/path/to/ops-nn/matmul/quant_matmul_activation_quant/tests
python3 -m ttk info
python3 -m ttk kernel -i "$OP_TESTS/st/arch35/ttk_kernel_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk aclnn -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk aclnn -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_weight_nz_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk e2e -i "$OP_TESTS/st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
```

无上板设备时可追加`--backend npusim`作模拟器验证，但不能作为上板通过结果。检查结果CSV中的每一条`precision_status`都为`PASS`；仅看TTK进程退出码不足以判断成功，`N/A`同样不算通过。

本地检查的ops-test-kit支持keyword-only tensor：E2E CSV的tensor顺序为`x1,x2,x2_scale,x1_scale,bias`，后两者按Python API签名装配为关键字参数。E2E显式传入x1_scale，绝不使用隐式单位scale；FT和WeightNZ的物理存储场景由kernel/ACLNN用例覆盖。三条通路的y dtype来源不同：kernel用attrs的`y_dtype`（GE 35/36/40），ACLNN用attrs的`yDtype`（GE 35/36/40，必须显式携带——TTK会把yOut提升为float32占位张量，golden无法从张量本身取回MX类型），E2E的FP8用attrs的`output_dtype`（PyTorch 23/24或ACL偏移编码291/292），FP4则以`uint8`双nibble打包并显式传入`output_dtype/x1_dtype/x2_dtype=296`；三者不能混用。

原生Windows不能直接执行当前版本TTK的完整CLI：入口依赖Linux的`resource`模块。`--validate`也会经过该入口；它只校验用例配置，不代表设备执行通过。`--cpu`不适用于本算子的NPU扩展实现。

## Golden与精度口径

golden按K的32元素MX组反量化，FP64保存参考操作数及组内点积，避免合法FP8/FP4 × E8M0在反量化时提前溢出。累加器以FP32 bias初始化（无bias时为零），每加入一个MX组的点积便舍入回FP32；bias从首组参与累加，不再于完整MatMul之后相加。该模型保留首轮bias和FP32累计舍入，但不声称逐指令复刻Cube内部归约树；32是MX量化分组，不是把kernel的MMAD K tile固定为32。SwiGLU、SiLU和两种GELU均保留原epilogue的BF16中间结果；SwiGLU与SiLU的舍入位置在激活完成后、MX scale选择之前。FP8使用RNE，OCP和BLAS独立生成scale。FP4支持`rint`、`floor`、`round`，并分别建模OCP与`scaleAlg=2`的动态dtype range/cuBLAS分支。全零组及全padding半组的E8M0字节为0（数值`2^-127`）。SwiGLU与SiLU激活链按950 VF默认`--cce-ftz=true`建模：Exp/Div深刷subnormal，最终Mul与BF16舍入保留subnormal。

`reference`返回MX数据码和E8M0 scale原始字节。kernel/ACLNN的FP4结果使用原生`float4_e2m1`数组，PyTorch E2E的FP4结果按低nibble在前打包为`uint8`；FP8结果保持相应FP8 dtype。NumPy输入返回NumPy数组，Torch输入返回CPU Torch张量，避免golden依赖TTK内部模块或CPU E8M0张量支持。

FP4量化在激活值乘以scale倒数之后，还需保留一次BF16 RNE舍入，再按`round_mode`转FP4。这对应`QuantizeFp4Vf`中的BF16 `Mul`；若直接将FP64归一化结果转FP4，极小负数在`floor`下会被错误量化为`-0.5`，而kernel中的BF16乘积已经舍入成负零。主kernel CSV的两条`gelu_erf_fp4_floor_*_bf16_mul_underflow`用例分别覆盖OCP与`scale_alg=2, dst_type_max=8`，bias均使用FP32正常数，以隔离前置Cube对subnormal的处理。

`gelu_erf`默认采用arch35 `gelu.h::ErfCallee`的实现一致性模型：保留原始系数位模式、分段条件、FP32 Horner FMA、Exp FTZ及`(1 + erf) * (0.5 * x)`运算顺序。负输入使erf接近-1时，`1 + erf`会放大近似误差，甚至舍入为零；该模型用于匹配当前kernel，并不代表数学GELU精度有所提升，也没有添加按输入阈值清零的特判。`activate(values, "gelu_erf", kernel_erf=False)`保留原`math.erf`计算路径供诊断；评估数学函数误差时可另用FP64的稳定形式`0.5 * x * erfc(-x / sqrt(2))`。若kernel的近似算法改变，应同步核对该模型。FP32组累加和CPU Exp仍是参考模型，不保证所有输入与硬件逐位相同，精度验收规则保持不变。

主CSV中的`gelu_erf_e4_s0_negative_tail_kernel_model`覆盖负尾部及Inf/NaN等边界；`mx_fp32_accum_e5_bias_cancel`验证首轮bias参与累计，`mx_fp32_accum_e5_overflow_cancel`验证中间FP32溢出对量化scale的影响。这三条均使用显式数据，便于复现计算顺序差异。

FP8输出量化按kernel的特殊值链路处理：激活后的某个32元素组含有Inf或NaN时，该组的E8M0 scale为`0xFF`，内部倒数为NaN；组内所有元素乘该倒数后均为NaN，最终FP8 `SAT` Cast在默认`CTRL[50]=0`下写为`0x00`。因此，不能将该组的量化数据写成`0x7F`，也不能仅将原来为Inf/NaN的位置清零。邻接的有限值组不受影响，未使用的scale padding仍为`0x00`。这一规则以激活后的分组值为准，不以输入张量是否含特殊值直接判定。

ST通过`assets/spec.py`的自定义`compare`统一检查GELU、SwiGLU和SiLU：`y`（FP8或FP4）与`y_scale`（E8M0）分别要求单点误差不超过1 ULP，超过1 ULP的元素数量各自不超过该输出元素总数的千分之一。ULP按对应dtype相邻可表示数值的间隔计数，FP4打包输出按解包后的元素计数。Kernel、ACLNN和E2E使用同一标准。

输出shape必须一致；正负零视为相等，NaN位置和Inf符号必须匹配，NaN payload不作要求。Kernel UT对确定性输入使用严格比较。
