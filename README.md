# Optics_design

本分支 `codex/inverse-feature-optimization` 在独立工作树中使用现有 1000 条光谱，比较小型 Transformer、斜率/曲率多尺度编码、保留相位的 Fourier token 和镜像等价联合损失。原始数据与历史训练代码、权重保持原样。

[本轮实验与结果](Metaformer_As2Se3/results/inverse_features1000/report.md)。新入口为 `optimize_spectral_inverse.py` 和 `predict_spectral_inverse.py`；PyCharm 使用配置 11/12。所有方案只训练逆向映射。

七个方案、每个三个种子的验证选中 `compact_raw_mirror`：320,552 参数，约比原模型少 92%；同一镜像等价口径的平均测试 MSE 从 0.026834 降至 0.021085（降低 21.4%）。原始标签周期 MSE 从 0.032750 变为 0.033056（略升 0.9%），两种评价同时保留。最终模型使用原始光谱和镜像联合损失，未选用导数/Fourier 特征分支。

**材料更正：1000 样本服务器数据也沿用了已知几乎丢失 Au 欧姆损耗的拟合。以下结果用于数值流程和模型对照，真实金器件性能需要修正材料后重新验证。** [证据](Metaformer_As2Se3/results/inverse_roundtrip_207/report.md)

此前分支：`codex/train-au100-metal8-1000`。已完成生成器 1.0 的 1000 样本逆向训练，固定划分 800/100/100；[训练结果](Metaformer_As2Se3/results/inverse_au100_1000/report.md)包含数据核对、参数误差、图表和使用方式。

模型保持 3,998,087 个参数，使用训练集光谱标准化。100 条测试 MSE 为 0.033422；在 88 个此前未出现的相同测试结构上，比旧 100 样本模型低 34.1%。五个尺寸参数明显改善，两个角仍约 21°。

已完成[一个新测试目标的真实 FDTD 回仿真](Metaformer_As2Se3/results/inverse_roundtrip_207/report.md)：网络直接输出的结构保留双谷形状，谷位置约 771.1/877.2 nm，相对目标偏移约 1.0/4.0 nm；第二谷仍超出 2 nm 参考标准。

此前的[100 样本配置对照](Metaformer_As2Se3/docs/inverse_optimization.md)及旧权重保留。训练和推理入口位于 [Metaformer_As2Se3](Metaformer_As2Se3/README.md)。
Optic design
