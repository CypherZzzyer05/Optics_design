# Optics_design

当前分支：`codex/train-au100-metal8-1000`。已完成生成器 1.0 的 1000 样本逆向训练，固定划分 800/100/100；[训练结果](Metaformer_As2Se3/results/inverse_au100_1000/report.md)包含数据核对、参数误差、图表和使用方式。

模型保持 3,998,087 个参数，使用训练集光谱标准化。100 条测试 MSE 为 0.033422；在 88 个此前未出现的相同测试结构上，比旧 100 样本模型低 34.1%。五个尺寸参数明显改善，两个角仍约 21°。

已完成[一个新测试目标的真实 FDTD 回仿真](Metaformer_As2Se3/results/inverse_roundtrip_207/report.md)：网络直接输出的结构保留双谷形状，谷位置约 771.1/877.2 nm，相对目标偏移约 1.0/4.0 nm；第二谷仍超出 2 nm 参考标准。

此前的[100 样本配置对照](Metaformer_As2Se3/docs/inverse_optimization.md)及旧权重保留。训练和推理入口位于 [Metaformer_As2Se3](Metaformer_As2Se3/README.md)。
Optic design
