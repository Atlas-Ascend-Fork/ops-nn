# RotateQuant 分组专项回归

本目录保留已有的 20 条 aclnnRotateQuantV2 分组用例和对应 golden。
它独立于 arch35 下的普通 RotateQuant 回归，不进入普通 CSV 的默认收集范围。
仅在 Ascend950 环境显式运行，不用于 DAV_9201/DAV_9202。

在 ops-test-kit 根目录执行：

~~~bash
export OPS_NN_PATH=/path/to/ops-nn
python3 -m ttk aclnn \
  -i "$OPS_NN_PATH/matmul/rotate_quant/tests/st/grouped/arch35/ttk_aclnn_rotate_quant_v2_st.csv" \
  --plugin "$OPS_NN_PATH/matmul/rotate_quant/tests/st/grouped/assets" \
  --pc 1 --seed 42 --run 1 --task-prof false
~~~

专项 golden 复用 tests/assets 中的普通参考计算，保留 count/cumsum、
三维/四维 rotation、空组和未覆盖输出行的原有处理。CPU 自测已移至 tests/ut/golden。
